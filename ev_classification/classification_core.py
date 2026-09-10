"""Pure, leakage-safe EV classification primitives. All physical features use kW."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import percentile_filter
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score)

DATES = ['datum_unterschrift_earliest', 'geplanter_baustart_earliest',
         'uebergabe_latest', 'inbetrieb_datum_latest']
LOADS = [f'net_load_{d}_{h:02d}_{m:02d}_kwh'
         for d in ('mon','tue','wed','thu','fri','sat','sun')
         for h in range(24) for m in (0,15,30,45)]
PROHIBITED = {'gp_nr', 'week_start', 'year', 'plz', 'ort', 'kanton', 'split',
    'ev_label', 'temporal_target', 'eligible_for_supervised_training',
    'training_exclusion_reason', 'label_date_ambiguous', 'pv_label',
    'pv_capacity_kwp', 'heat_pump_label', 'heat_pump_boiler_label',
    'battery_storage_label', *DATES}


def temporal_labels(frame, separate_bounds=False, washout=0):
    dates = frame[DATES].apply(pd.to_datetime)
    lo = dates[DATES[:2]].min(axis=1) if separate_bounds else dates.min(axis=1)
    hi = dates[DATES[2:]].max(axis=1) if separate_bounds else dates.max(axis=1)
    start = pd.to_datetime(frame.week_start)
    target = pd.Series(np.nan, index=frame.index, name='temporal_target')
    target.loc[frame.ev_label.eq(0)] = 0
    dated_ev = frame.ev_label.eq(1) & lo.notna() & hi.notna() & lo.le(hi)
    target.loc[dated_ev & (start + pd.Timedelta(days=7)).le(lo)] = 0
    target.loc[dated_ev & start.ge(hi + pd.Timedelta(days=washout))] = 1
    return target, lo, hi


def supervised(frame):
    return frame.loc[frame.eligible_for_supervised_training.eq(1)
                     & frame.temporal_target.notna()].copy()


def split_customers(frame, seed=20260910):
    customers = frame.groupby('gp_nr', sort=True).ev_label.first().to_frame()
    usable = set(supervised(frame).gp_nr)
    customers['usable'] = customers.index.isin(usable)
    rng = np.random.default_rng(seed)
    customers['split'] = ''
    for _, group in customers.groupby(['usable', 'ev_label'], sort=True):
        ids = rng.permutation(group.index.to_numpy())
        n = len(ids)
        ndev, nval = int(round(.6*n)), int(round(.2*n))
        customers.loc[ids[:ndev], 'split'] = 'development'
        customers.loc[ids[ndev:ndev+nval], 'split'] = 'validation'
        customers.loc[ids[ndev+nval:], 'split'] = 'test'
    return customers.reset_index()


def primary_cohort(frame, cap=12, minimum=2):
    eligible = supervised(frame)
    state = eligible.loc[eligible.ev_label.eq(0) | eligible.temporal_target.eq(1)]
    recent = state.sort_values(['gp_nr','week_start']).groupby('gp_nr').tail(cap)
    counts = recent.groupby('gp_nr').size()
    cohort = frame.groupby('gp_nr', sort=True).ev_label.first().to_frame('target')
    cohort['weeks'] = counts.reindex(cohort.index, fill_value=0)
    cohort['status'] = np.where(cohort.weeks.ge(minimum), 'ok', 'insufficient_history')
    return recent.loc[recent.gp_nr.isin(cohort.index[cohort.status.eq('ok')])], cohort.reset_index()


def customer_weights(frame):
    # Equal total weight per customer; scale to mean one for regularized models.
    w = 1 / frame.groupby('gp_nr').gp_nr.transform('size').to_numpy(float)
    return w / w.mean()


def kw(values):
    return np.asarray(values, dtype=float) * 4.0


def compact_features(frame):
    x = kw(frame[LOADS])
    q = np.quantile(x, [.05,.25,.5,.75,.95,.99], axis=1)
    ramp = np.abs(np.diff(x, axis=1))
    out = {f'stat_q{p}': v for p,v in zip((5,25,50,75,95,99),q)}
    out.update(stat_range=q[4]-q[0], stat_mean=x.mean(1),
        stat_load_factor=x.mean(1)/np.maximum(np.abs(x).max(1), .1),
        stat_ramp50=np.quantile(ramp,.5,axis=1),
        stat_ramp95=np.quantile(ramp,.95,axis=1),
        stat_ramp99=np.quantile(ramp,.99,axis=1))
    hours = np.tile(np.arange(96)/4, 7)
    night = (hours<6)|(hours>=22)
    out['stat_night_kwh'] = x[:,night].sum(1)/4
    out['stat_day_kwh'] = x[:,~night].sum(1)/4
    out['stat_weekday_daily_kwh'] = x[:,:480].sum(1)/20
    out['stat_weekend_daily_kwh'] = x[:,480:].sum(1)/8
    for level in (1.5,3,5,7,11):
        out[f'stat_hours_above_{level}'] = (x>level).sum(1)/4
    return pd.DataFrame(out,index=frame.index)


def shape_features(frame):
    x = kw(frame[LOADS])
    centered = x - np.median(x,axis=1)[:,None]
    scale = np.maximum(np.quantile(x,.95,axis=1)-np.quantile(x,.05,axis=1),.1)
    z = centered/scale[:,None]
    blocks = z.reshape(-1,7,4,24).mean(axis=3)
    fft = np.abs(np.fft.rfft(z, axis=1))/672
    out = {f'shape_day{d}_block{b}':blocks[:,d,b] for d in range(7) for b in range(4)}
    out.update({f'fourier_{k}':fft[:,k] for k in (1,2,3,7,14,21,28)})
    return pd.DataFrame(out,index=frame.index)


def check_features(features):
    assert not (set(features.columns) & PROHIBITED), 'Prohibited feature column'
    assert all(c.startswith(('stat_','event_','shape_','fourier_')) for c in features), 'Non-signal feature'
    assert np.isfinite(features.to_numpy()).all(), 'Non-finite features'


def baseline_power(x, method):
    if method == 'daily_q20':
        return np.repeat(np.quantile(x.reshape(-1,7,96), .2, axis=2),96,axis=1)
    if method == 'rolling_q20':
        return percentile_filter(x,percentile=20,size=(1,33),mode='nearest')
    raise ValueError(method)


def extract_events(frame, baseline='daily_q20', amplitude=3.0):
    x = kw(frame[LOADS]); base = baseline_power(x, baseline); residual=x-base
    records=[]
    for row, power, res in zip(frame.index, x, residual):
        high=res>=amplitude
        merged=high.copy()
        merged[1:-1] |= high[:-2]&high[2:]
        changes=np.diff(np.r_[False,merged,False].astype(int))
        for start,end in zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1)):
            if end-start<4: continue
            y=res[start:end]; median=float(np.median(y)); mad=float(np.median(np.abs(y-median)))
            rise=float(power[start]-power[start-1]) if start else 0.
            fall=float(power[end-1]-power[end]) if end<672 else 0.
            symmetry=min(max(rise,0),max(fall,0))/max(abs(rise),abs(fall),.1)
            normalized=y/max(median,.1)
            rect=np.mean(np.abs(normalized-1))
            taper=np.mean(np.abs(normalized-np.linspace(1.2,.8,len(y))))
            records.append((row,start,end,(end-start)/4,median,float(np.quantile(y,.9)),
                float(np.maximum(y,0).sum()/4),mad, float((y[-1]-y[0])/max(len(y)-1,1)),
                float(np.mean(np.abs(y-median)<=max(.5,.2*median))),rise,fall,symmetry,
                start//96, (end-1)//96 != start//96,
                float(np.mean(power[start:end]<0)),
                float(np.mean((power[start:end]<0)&(np.arange(start,end)%96>=40)&(np.arange(start,end)%96<60))),
                float(np.mean(~high[start:end])),
                float(np.max(np.abs(np.diff(power[max(0,start-4):min(672,end+4)])))),
                1/(1+min(rect,taper)),float(start%96/4),float(end%96/4)))
    columns=['row','start','end','duration','power','power90','energy','mad','slope','plateau',
        'rise','fall','symmetry','day','cross_midnight','negative_share','midday_export_share',
        'interrupted_share','nearby_ramp','template','start_hour','end_hour']
    return pd.DataFrame(records,columns=columns)


def event_features(frame, events, config):
    e=events.loc[events.duration.ge(config.get('duration',1.5))].copy()
    family=config.get('family','block')
    if family in ('plateau','repeat_plateau'):
        e=e.loc[(e.mad/np.maximum(e.power,.1)<=config.get('variation',.3))
                & e.rise.ge(config.get('edge',.8)) & e.fall.ge(config.get('edge',.8))
                & e.symmetry.ge(config.get('symmetry',.3))]
    tolerance=config.get('tolerance',.75)
    # Repetition is within the observed week: no future/customer-history fitting.
    repeats={}
    for row,g in e.groupby('row',sort=False):
        powers=g.power.to_numpy(); days=g.day.to_numpy()
        repeats[row]=max((len(set(days[np.abs(powers-p)<=tolerance])) for p in powers),default=0)
    columns=['count','days','duration','energy','longest','power','plateau','paired',
             'repetition','template','night','negative','interruptions']
    out=pd.DataFrame(0.,index=frame.index,columns=['event_'+c for c in columns])
    if not e.empty:
        e['paired']=(e.rise.gt(.8)&e.fall.gt(.8)&e.symmetry.gt(.3)).astype(float)
        e['night']=((e.start_hour<6)|(e.start_hour>=22)).astype(float)
        g=e.groupby('row')
        stats={'count':g.size(),'days':g.day.nunique(),'duration':g.duration.sum(),
            'energy':g.energy.sum(),'longest':g.duration.max(),'power':g.power.median(),
            'plateau':g.plateau.mean(),'paired':g.paired.mean(),'repetition':pd.Series(repeats),
            'template':g.template.mean(),'night':g.night.mean(),'negative':g.negative_share.mean(),
            'interruptions':g.interrupted_share.mean()}
        for c,v in stats.items(): out.loc[v.index,'event_'+c]=v
    count=out.event_count.to_numpy()
    score=1-np.exp(-count/config.get('min_events',2))
    if family.startswith('repeat'):
        score*=np.minimum(out.event_repetition.to_numpy()/config.get('repeat_days',2),1)
    if family=='template': score*=out.event_template.to_numpy()
    return out, score, e


def aggregate(weeks, scores, rule='top3', event_frame=None):
    work=weeks[['gp_nr','week_start','temporal_target']].copy()
    work['score']=np.asarray(scores)
    records=[]
    for gp,g in work.groupby('gp_nr',sort=True):
        ordered=g.sort_values('score',ascending=False,kind='stable')
        if rule=='max': value=float(g.score.max())
        elif rule=='top3': value=float(ordered.score.head(3).mean())
        elif rule=='evidence':
            if event_frame is None:
                value=float(1-np.exp(-g.score.sum()/3))
            else:
                f=event_frame.loc[g.index]
                value=float(min(1-np.exp(-f.event_count.sum()/3),1-np.exp(-f.event_days.sum()/2)))
        else: raise ValueError(rule)
        records.append((gp,int(g.temporal_target.iloc[0]),value,len(g),
                        ordered.week_start.head(3).astype(str).tolist()))
    return pd.DataFrame(records,columns=['gp_nr','target','score','weeks','evidence_weeks'])


def metrics(y,p,threshold=.5):
    y=np.asarray(y,dtype=int); p=np.asarray(p,float); pred=p>=threshold
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    both=len(np.unique(y))==2
    return {'customers_or_weeks':len(y),'positives':int(y.sum()),
        'pr_auc':float(average_precision_score(y,p)) if y.sum() else None,
        'balanced_accuracy':float(balanced_accuracy_score(y,pred)) if both else None,
        'f1':float(f1_score(y,pred,zero_division=0)),
        'recall':float(recall_score(y,pred,zero_division=0)),
        'precision':float(precision_score(y,pred,zero_division=0)),
        'specificity':float(tn/(tn+fp)) if tn+fp else None,
        'roc_auc':float(roc_auc_score(y,p)) if both else None,
        'brier':float(brier_score_loss(y,p)), 'confusion_matrix':[[int(tn),int(fp)],[int(fn),int(tp)]]}


def choose_threshold(y,p):
    candidates=np.r_[0.,np.unique(p),np.nextafter(1.,2.)]
    # Balanced accuracy, then F1, then higher threshold.
    return float(max(candidates,key=lambda t:(balanced_accuracy_score(y,np.asarray(p)>=t),
                      f1_score(y,np.asarray(p)>=t,zero_division=0),t)))


def bootstrap(y,p,threshold,seed=20260910,n=1000,other=None):
    y=np.asarray(y,int); p=np.asarray(p,float)
    rng=np.random.default_rng(seed); values=[]
    for _ in range(n):
        idx=rng.integers(0,len(y),len(y))
        if len(np.unique(y[idx]))<2: continue
        ap=average_precision_score(y[idx],p[idx])
        if other is not None:
            values.append(ap-average_precision_score(y[idx],np.asarray(other)[idx]))
        else:
            values.append([ap,balanced_accuracy_score(y[idx],p[idx]>=threshold),
                f1_score(y[idx],p[idx]>=threshold,zero_division=0),brier_score_loss(y[idx],p[idx])])
    if other is not None:
        return {'delta_pr_auc':float(average_precision_score(y,p)-average_precision_score(y,other)),
                'ci95':np.quantile(values,[.025,.975]).tolist(),'draws':len(values)}
    intervals=np.quantile(values,[.025,.975],axis=0).T
    return {k:v.tolist() for k,v in zip(('pr_auc','balanced_accuracy','f1','brier'),intervals)}
