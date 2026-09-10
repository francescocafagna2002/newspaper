#!/usr/bin/env python3
"""Staged, resumable experiment runner. Final test requires a persisted freeze.

Run audit, baselines, heuristics, models, calibrate, freeze, final in that order.
Existing experiment directories are never overwritten. Notebook reads caches.
"""
from __future__ import annotations
import argparse
from datetime import datetime, UTC
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import pickle
import resource
import shutil
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, precision_recall_curve

try:
    from .classification_core import *
except ImportError:
    from classification_core import *

ROOT=Path(__file__).resolve().parent
CONFIG={'seed':20260910,'split':[.6,.2,.2],'history_cap':12,'minimum_history':2,
 'primary':'customer_average_precision','threshold_metric':'balanced_accuracy',
 'bootstrap_draws':1000,'privacy_min_cell':5,'minimum_validation_positives':8,
 'baselines':['daily_q20','rolling_q20'],'amplitudes_kw':[1.5,3.,5.],
 'durations_hours':[1.,1.5], 'min_events':[2,3],
 'aggregations':['max','top3','evidence'],
 'plateau_variations':[.2,.4],'edge_kw':[.5,1.], 'edge_symmetry':.3,
 'repeat_tolerance_kw':[.5,1.], 'repeat_days':[2,3],
 'lgbm_grid':[
  {'num_leaves':7,'learning_rate':.05,'n_estimators':250,'min_child_samples':100,'subsample':.8,'colsample_bytree':.9,'reg_alpha':.1,'reg_lambda':5.},
  {'num_leaves':15,'learning_rate':.03,'n_estimators':350,'min_child_samples':150,'subsample':.8,'colsample_bytree':.8,'reg_alpha':.5,'reg_lambda':10.},
  {'num_leaves':7,'learning_rate':.03,'n_estimators':350,'min_child_samples':200,'subsample':1.,'colsample_bytree':1.,'reg_alpha':1.,'reg_lambda':15.}],
 'early_stopping_rounds':30,'max_experiments':12,
 'date_bounds':'min/max all four; half-open weeks; eligibility gate first'}


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,default=lambda x: x.item() if isinstance(x,np.generic) else str(x),allow_nan=False)+'\n')


def read_json(path): return json.loads(path.read_text())


def source_identity():
    source=ROOT/'output/customer_week_table.parquet'
    if not source.exists(): raise FileNotFoundError(f'Required Parquet dataset absent: {source}')
    files=[{'name':str(p.relative_to(ROOT/'output')),'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}
           for p in sorted(source.rglob('*.parquet'))]
    metadata={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
              (ROOT/'output/run_metadata.json',ROOT/'output/run_summary.csv') if p.exists()}
    return {'files':files,'metadata_hashes':metadata,'hash':digest([files,metadata])}


class Study:
    def __init__(self,smoke=False):
        self.smoke=smoke
        self.out=ROOT/('classification_smoke' if smoke else 'classification_output')
        self.out.mkdir(exist_ok=True)
        self.journal=self.out/'results.md' if smoke else ROOT/'results.md'
        self.config=CONFIG.copy()
        if smoke: self.config['bootstrap_draws']=50
        self.data=pd.read_parquet(ROOT/'output/customer_week_table.parquet').sort_values(['gp_nr','week_start']).reset_index(drop=True)
        if smoke:
            ids=self.data[['gp_nr','ev_label']].drop_duplicates().sort_values('gp_nr').groupby('ev_label').head(35).gp_nr
            self.data=self.data.loc[self.data.gp_nr.isin(ids)].reset_index(drop=True)
        self.data['temporal_target'],self.data['earliest_known_date'],self.data['latest_known_date']=temporal_labels(self.data)
        if (self.out/'run_manifest.json').exists():
            manifest=read_json(self.out/'run_manifest.json')
            assert manifest['source']['hash']==source_identity()['hash'], 'Source changed; refuse to compare runs'
            assert manifest['config']==self.config, 'Configuration changed; create a new study'
        if (self.out/'split_manifest.parquet').exists():
            manifest=pd.read_parquet(self.out/'split_manifest.parquet').set_index('gp_nr')
            self.data['split']=self.data.gp_nr.map(manifest.split)
            assert self.data.split.notna().all()
        self.events_cache={}

    def append(self,text):
        with self.journal.open('a') as f: f.write(text+'\n')

    def decision(self,text):
        path=self.out/'decisions.json'
        logs=read_json(path) if path.exists() else []
        logs.append({'time':datetime.now(UTC).isoformat(),'decision':text});write_json(path,logs)
        self.append('\nDecision: '+text+'\n')

    def audit(self):
        if (self.out/'audit.json').exists(): return
        assert not (self.out/'run_manifest.json').exists(), 'Interrupted audit: inspect before resuming'
        code={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.glob('*.py'))}
        versions={m:importlib.metadata.version(m) for m in ('numpy','pandas','scipy','scikit-learn','pyarrow','matplotlib')}
        for m in ('lightgbm','nbformat','nbclient','ipykernel'):
            try: versions[m]=importlib.metadata.version(m)
            except importlib.metadata.PackageNotFoundError: versions[m]='unavailable'
        manifest={'created_utc':datetime.now(UTC).isoformat(),'source':source_identity(),
            'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'working_tree_hash':digest(code),'code_hashes':code,'config':self.config,'python':sys.version,
            'versions':versions,'smoke':self.smoke,'test_status':'sealed'}
        manifest['run_identity']=digest(manifest)
        write_json(self.out/'run_manifest.json',manifest)
        self.journal.write_text('# EV classification experiment results\n\n## Run identity and evaluation protocol\n\n'
            f"Run `{manifest['run_identity']}`; source `{manifest['source']['hash']}`. "
            f"Created {manifest['created_utc']}; commit `{manifest['commit']}`; code hash `{manifest['working_tree_hash']}`. "
            'User waived unavailable grill-me prerequisite. Source data is immutable. '
            'Seed 20260910; customer split 60/20/20; minimum 2 weeks, latest 12 state-specific weeks. '
            'Primary metric: customer average precision (PR-AUC); operating threshold maximizes balanced accuracy, then F1. '
            'All intervals are 95% customer bootstrap (1,000 paired draws). Minimum publishable slice size is 5. '
            'Project-span labels are weak labels, not verified EV commissioning dates. Test predictions remain sealed until FINAL FREEZE.\n\n'
            '## Data audit\n\nPending E00.\n\n## Validation leaderboard\n\n<!-- LEADERBOARD -->\nPending.\n<!-- END LEADERBOARD -->\n\n## Experiment log\n')
        try:
            d=self.data
            assert [c for c in d if c.startswith('net_load_')]==LOADS
            assert not d.duplicated(['gp_nr','week_start']).any()
            assert np.isfinite(d[LOADS].to_numpy()).all()
            assert d.groupby('gp_nr').ev_label.nunique().max()==1
            assert d.ev_label.isin([0,1]).all()
            assert d.eligible_for_supervised_training.isin([0,1]).all()
            assert d.week_start.dt.dayofweek.eq(0).all()
            assert d.observed_timestamp_count.add(d.imputed_timestamp_count).eq(672).all()
            assert d.imputed_timestamp_count.le(12).all() and d.longest_missing_run.le(4).all()
            assert np.allclose(d.coverage_ratio,d.observed_timestamp_count/672)
            assert not ((d.eligible_for_supervised_training==1)&d.training_exclusion_reason.notna()).any()
            from zoneinfo import ZoneInfo
            from datetime import timedelta
            tz=ZoneInfo('Europe/Zurich')
            assert all(t.to_pydatetime().replace(tzinfo=tz).utcoffset()==
                (t.to_pydatetime().replace(tzinfo=tz)+timedelta(days=7)).utcoffset() for t in d.week_start.unique())
            metadata=read_json(ROOT/'output/run_metadata.json')
            assert metadata['register_convention']['formula']=='import_kwh - export_kwh'
            assert not any('.tmp.' in p for p in metadata['finalized_monthly_exports'])
            split=split_customers(d,self.config['seed']);split.to_parquet(self.out/'split_manifest.parquet',index=False)
            assert split.gp_nr.is_unique and set(split.split)=={'development','validation','test'}
            d['split']=d.gp_nr.map(split.set_index('gp_nr').split)
            label_columns=['gp_nr','week_start',*DATES,'earliest_known_date','latest_known_date','temporal_target',
                'ev_label','eligible_for_supervised_training','training_exclusion_reason','split']
            d[label_columns].to_parquet(self.out/'temporal_label_audit.parquet',index=False)
            counts={}
            for part in ('development','validation','test'):
                rows,cohort=primary_cohort(d.loc[d.split.eq(part)],self.config['history_cap'],self.config['minimum_history'])
                cohort.to_parquet(self.out/f'cohort_{part}.parquet',index=False)
                rows[['gp_nr','week_start']].to_parquet(self.out/f'cohort_weeks_{part}.parquet',index=False)
                counts[part]={'all_customers':len(cohort),'primary_customers':int(cohort.status.eq('ok').sum()),
                    'primary_positive':int(cohort.loc[cohort.status.eq('ok'),'target'].sum()),
                    'positive_without_post_week':int(((cohort.target==1)&(cohort.weeks==0)).sum()),
                    'insufficient_history':int(cohort.status.ne('ok').sum()),
                    'supervised_weeks':len(supervised(d.loc[d.split.eq(part)]))}
            if not self.smoke:
                assert counts['validation']['primary_positive']>=self.config['minimum_validation_positives'], 'Too few validation positives; grouped outer CV required'
            audit={'rows':len(d),'customers':d.gp_nr.nunique(),'load_columns':len(LOADS),
                'excluded_rows':int(d.eligible_for_supervised_training.ne(1).sum()),
                'unknown_eligible_rows':int((d.eligible_for_supervised_training.eq(1)&d.temporal_target.isna()).sum()),
                'target_counts':{str(k):int(v) for k,v in supervised(d).temporal_target.value_counts().items()},
                'date_field_completeness':{c:int(d.drop_duplicates('gp_nr')[c].notna().sum()) for c in DATES},
                'splits':counts,'quality_pass':True,'units':'kWh / 15 min; kW = 4*kWh',
                'input_summary':pd.read_csv(ROOT/'output/run_summary.csv').to_dict('records')[0]}
            if not self.smoke:
                assert audit['input_summary']['eligible_week_count']==len(d)
                coverage=pd.read_parquet(ROOT/'output/customer_coverage.parquet')
                assert set(d.gp_nr)==set(coverage.loc[coverage.eligible_week_count.gt(0),'gp_nr'])
                assert coverage.eligible_week_count.sum()==len(d)
            write_json(self.out/'audit.json',audit)
            self.append('\n### E00 — Data and label audit\n\nAll schema, load, duplicate, quality, DST, register-sign and source-count checks passed.\n\n```json\n'+json.dumps(audit,indent=2)+'\n```\n\n'
                'Only 331 of 878 source customers have retained weeks. Eligibility excludes contradictory labels. '
                'Validation has few positives; uncertainty and cohort selection limit generalization.\n')
            self.decision('E00 passed; run prevalence and compact logistic baselines before adding detector complexity.')
        except Exception as exc:
            self.append(f'\n### E00 — FAILED\n\n{type(exc).__name__}: {exc}. No input data changed; modelling stopped.\n')
            raise

    def rows(self,part): return supervised(self.data.loc[self.data.split.isin(part if isinstance(part,list) else [part])])

    def cohort_rows(self,rows):
        result,_=primary_cohort(rows,self.config['history_cap'],self.config['minimum_history'])
        return result

    def get_events(self,rows,config):
        key=(digest(rows.index.tolist()),config['baseline'],config['amplitude'])
        if key not in self.events_cache:
            path=self.out/f'events_cache_{digest(key)[:16]}.parquet'
            if path.exists(): self.events_cache[key]=pd.read_parquet(path)
            else:
                events=extract_events(rows,config['baseline'],config['amplitude'])
                events.to_parquet(path,index=False);self.events_cache[key]=events
        return self.events_cache[key]

    def features(self,rows,kind,heuristic=None):
        x=compact_features(rows)
        if kind in ('events','shape','fourier'):
            ef,_,_=event_features(rows,self.get_events(rows,heuristic),heuristic)
            x=pd.concat([x,ef],axis=1)
        if kind in ('shape','fourier'):
            shapes=shape_features(rows)
            if kind=='fourier': shapes=shapes.filter(like='fourier_')
            x=pd.concat([x,shapes],axis=1)
        check_features(x);return x

    def select(self,rows,p,event_frame=None,rules=None):
        cohort=self.cohort_rows(rows)
        selected=np.asarray(p)[rows.index.get_indexer(cohort.index)]
        choices=[]
        for rule in rules or self.config['aggregations']:
            customers=aggregate(cohort,selected,rule,event_frame)
            threshold=choose_threshold(customers.target,customers.score)
            m=metrics(customers.target,customers.score,threshold)
            choices.append((m['pr_auc'],m['balanced_accuracy'],m['f1'],-m['brier'],rule,threshold,customers))
        best=max(choices,key=lambda t:t[:4])
        return {'aggregation':best[4],'threshold':best[5],'customers':best[6],
                'search':[{'aggregation':v[4],'threshold':v[5],'pr_auc':v[0],'balanced_accuracy':v[1]} for v in choices]}

    def finish(self,eid,family,hypothesis,config,rows,p,selection,started,model=None,features=None,events=None,extra=None):
        directory=self.out/eid
        assert not directory.exists(),f'{eid} immutable: already exists'
        directory.mkdir()
        cfg={**config,'aggregation':selection['aggregation'],'threshold':selection['threshold'],
             'study_config':self.config,'source_identity':source_identity()['hash'],
             'code_hash':digest({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')})}
        write_json(directory/'config.json',cfg)
        customers=selection['customers']; customers.to_parquet(directory/'customer_validation.parquet',index=False)
        weekly=rows[['gp_nr','week_start','temporal_target']].copy();weekly['score']=p
        weekly.to_parquet(directory/'weekly_validation.parquet',index=False)
        result={'customer':metrics(customers.target,customers.score,cfg['threshold']),
            'week':metrics(rows.temporal_target,p,cfg['threshold']),
            'ci95':bootstrap(customers.target,customers.score,cfg['threshold'],n=self.config['bootstrap_draws']),
            'runtime_seconds':time.perf_counter()-started,
            'peak_process_rss_mb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            'aggregation_search':selection['search']}
        if extra: result.update(extra)
        write_json(directory/'metrics_validation.json',result)
        names=[] if features is None else list(features.columns)
        write_json(directory/'feature_manifest.json',{'features':names,'hash':digest(names),
            'prohibited_columns':sorted(PROHIBITED),'prohibited_check_passed':not bool(set(names)&PROHIBITED)})
        if model is not None:
            with (directory/'model.pkl').open('wb') as f:pickle.dump(model,f)
            if hasattr(model,'booster_'):model.booster_.save_model(str(directory/'model.txt'))
        if events is not None:
            events=events.merge(rows[['gp_nr','week_start']],left_on='row',right_index=True,how='left')
            events.to_parquet(directory/'events_validation.parquet',index=False)
        self.diagnostics(directory,customers,weekly,rows,cfg,events)
        leaderboard=read_json(self.out/'leaderboard.json') if (self.out/'leaderboard.json').exists() else []
        champion=read_json(self.out/'champion.json')['id'] if (self.out/'champion.json').exists() else None
        comparison=None
        if champion:
            old=pd.read_parquet(self.out/champion/'customer_validation.parquet')
            assert customers.gp_nr.tolist()==old.gp_nr.tolist(), 'Comparison cohort changed'
            comparison=bootstrap(customers.target,customers.score,cfg['threshold'],n=self.config['bootstrap_draws'],other=old.score)
        promoted=champion is None or comparison['ci95'][0]>0
        if promoted:write_json(self.out/'champion.json',{'id':eid,'reason':'first pipeline' if champion is None else comparison})
        entry={'id':eid,'family':family,'hypothesis':hypothesis,'metrics':result['customer'],
            'ci95':result['ci95'],'runtime':result['runtime_seconds'],'promoted':promoted,'paired_vs_champion':comparison}
        leaderboard.append(entry);write_json(self.out/'leaderboard.json',leaderboard)
        m=result['customer'];ci=result['ci95']['pr_auc']
        self.append(f'\n### {eid} — {family}\n\n- Hypothesis: {hypothesis}\n'
            f'- Inputs/split: eligible labelled development / validation; fixed {len(customers)}-customer cohort, 12-week cap.\n'
            f'- Method and parameters: `{json.dumps(config,default=str)}`; aggregation `{cfg["aggregation"]}`, threshold {cfg["threshold"]:.6g}.\n'
            f'- Result (validation): AP {m["pr_auc"]:.4f} [{ci[0]:.4f}, {ci[1]:.4f}], balanced accuracy {m["balanced_accuracy"]:.4f}, F1 {m["f1"]:.4f}, Brier {m["brier"]:.4f}.\n'
            f'- Paired comparison: {json.dumps(comparison)}.\n'
            f'- Decision: {"promoted" if promoted else "retained earlier simpler champion; AP gain not established beyond paired uncertainty"}.\n'
            f'- Error slices/observations: [suppressed-small-cell slices](classification_output/{eid}/slices.csv), [diagnostics](classification_output/{eid}/diagnostics.png).\n'
            f'- Artifacts: [configuration](classification_output/{eid}/config.json), [metrics](classification_output/{eid}/metrics_validation.json).\n')
        self.update_leaderboard(leaderboard)
        print(eid,json.dumps({'AP':m['pr_auc'],'CI':ci,'BA':m['balanced_accuracy'],'promoted':promoted,'paired':comparison}),flush=True)
        return cfg,result

    def update_leaderboard(self,entries):
        lines=['| Rank | ID | Family | Key change | Customers | PR-AUC | Balanced accuracy | F1 | Brier | Runtime | Status |',
               '|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|']
        for i,e in enumerate(sorted(entries,key=lambda e:-e['metrics']['pr_auc']),1):
            m=e['metrics'];lines.append(f'| {i} | {e["id"]} | {e["family"]} | {e["hypothesis"]} | {m["customers_or_weeks"]} | {m["pr_auc"]:.4f} | {m["balanced_accuracy"]:.4f} | {m["f1"]:.4f} | {m["brier"]:.4f} | {e["runtime"]:.1f}s | validation |')
        text=self.journal.read_text();before,tail=text.split('<!-- LEADERBOARD -->');_,after=tail.split('<!-- END LEADERBOARD -->')
        self.journal.write_text(before+'<!-- LEADERBOARD -->\n'+'\n'.join(lines)+'\n<!-- END LEADERBOARD -->'+after)

    def diagnostics(self,directory,customers,weekly,rows,config,events=None):
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        precision,recall,_=precision_recall_curve(customers.target,customers.score)
        axes[0].plot(recall,precision);axes[0].set(xlabel='Recall',ylabel='Precision',title='Validation customers')
        obs,pred=calibration_curve(customers.target,customers.score,n_bins=5,strategy='quantile')
        axes[1].plot(pred,obs,'o-');axes[1].plot([0,1],[0,1],'--');axes[1].set(xlabel='Score',ylabel='Observed fraction',title='Reliability (scores may be uncalibrated)')
        fig.tight_layout();fig.savefig(directory/'diagnostics.png');plt.close(fig)
        meta=rows.groupby('gp_nr').first()
        joined=customers.set_index('gp_nr').join(meta[['pv_label','heat_pump_label','battery_storage_label']])
        slices=[]
        for col in ('pv_label','heat_pump_label','battery_storage_label','weeks'):
            for value,g in joined.groupby(col):
                if len(g)<5 or min(g.target.sum(),len(g)-g.target.sum())<5:continue
                slices.append({'slice':col,'value':str(value),**metrics(g.target,g.score,config['threshold'])})
        detail=weekly.copy();detail['season']=detail.week_start.dt.month.map(lambda m:('winter','spring','summer','autumn')[(m%12)//3])
        detail['year']=detail.week_start.dt.year
        detail['state']=np.where(detail.temporal_target.eq(1),'post_positive','negative')
        detail['imputed']=rows.imputed_timestamp_count.gt(0).to_numpy()
        for col in ('season','year','state','imputed'):
            for value,g in detail.groupby(col):
                if g.gp_nr.nunique()<5:continue
                slices.append({'slice':'week_'+col,'value':str(value),**metrics(g.temporal_target,g.score,config['threshold'])})
        pd.DataFrame(slices).to_csv(directory/'slices.csv',index=False)
        # Representative examples use pseudonyms, never raw identifiers.
        examples=[]
        for target,predicted,name in ((1,1,'TP'),(0,0,'TN'),(0,1,'FP'),(1,0,'FN')):
            candidates=customers.loc[customers.target.eq(target)&customers.score.ge(config['threshold']).eq(bool(predicted))]
            if candidates.empty:continue
            gp=candidates.iloc[0].gp_nr
            evidence=weekly.loc[weekly.gp_nr.eq(gp)].sort_values('score',ascending=False).iloc[0]
            row=rows.loc[rows.gp_nr.eq(gp)&rows.week_start.eq(evidence.week_start)].iloc[0]
            power=kw(row[LOADS].to_numpy(float));method=config.get('heuristic',config).get('baseline','daily_q20')
            base=baseline_power(power[None,:],method)[0]
            fig,ax=plt.subplots(figsize=(12,3));ax.plot(np.arange(672)/4,power,label='Net kW');ax.plot(np.arange(672)/4,base,label='Baseline');ax.plot(np.arange(672)/4,power-base,alpha=.4,label='Residual')
            if events is not None:
                ev=events.loc[events.gp_nr.eq(gp)&events.week_start.eq(row.week_start)]
                for _,event in ev.iterrows():ax.axvspan(event.start/4,event.end/4,alpha=.15,color='green')
            ax.set(title=f'{name}: anonymized example',xlabel='Hours from Monday',ylabel='kW');ax.legend();fig.tight_layout();fig.savefig(directory/f'example_{name}.png');plt.close(fig)
            examples.append({'category':name,'pseudonym':digest(str(gp))[:10],'score':float(evidence.score)})
        write_json(directory/'examples.json',examples)

    def baselines(self):
        train=self.rows('development');val=self.rows('validation')
        for eid,kind in [('E01_dummy','dummy'),('E02_logistic','logistic')]:
            if (self.out/eid).exists():continue
            started=time.perf_counter();x=self.features(train,'compact');v=self.features(val,'compact')
            if kind=='dummy':
                prevalence=float(np.average(train.temporal_target,weights=customer_weights(train)))
                p=np.full(len(val),prevalence);model=None;config={'kind':kind,'prevalence':prevalence,'features':'compact'}
            else:
                model=make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=2000,random_state=self.config['seed']))
                model.fit(x,train.temporal_target,logisticregression__sample_weight=customer_weights(train))
                p=model.predict_proba(v)[:,1];config={'kind':kind,'C':1.,'features':'compact'}
            self.finish(eid,kind,'Establish customer-balanced prevalence floor' if kind=='dummy' else 'Compact signal statistics improve on prevalence',
                config,val,p,self.select(val,p),started,model,x)
            self.decision(f'{eid} completed; '+('run logistic baseline next.' if kind=='dummy' else 'inspect baseline uncertainty, then test sustained charging events.'))

    def heuristics(self):
        val=self.rows('validation')
        best_id=None;stale=0
        for eid,family in [('E10_sustained_block','block'),('E11_plateau_edges','plateau'),('E12_repeated_level','repeat')]:
            if (self.out/eid).exists():
                if best_id is None:best_id=eid
                else:best_id=self.heuristic_champion([best_id,eid])
                continue
            started=time.perf_counter();search=[];winner=None
            if family=='block':
                configs=[{'family':family,'baseline':b,'amplitude':a,'duration':dur,'min_events':count}
                    for b,a,dur,count in itertools.product(self.config['baselines'],self.config['amplitudes_kw'],self.config['durations_hours'],self.config['min_events'])]
            elif family=='plateau':
                base=read_json(self.out/'E10_sustained_block/config.json')
                configs=[{k:base[k] for k in ('baseline','amplitude','duration','min_events')} | {'family':family,'variation':v,'edge':e,'symmetry':.3}
                    for v,e in itertools.product(self.config['plateau_variations'],self.config['edge_kw'])]
            else:
                base=read_json(self.out/best_id/'config.json')
                configs=[{k:v for k,v in base.items() if k in ('baseline','amplitude','duration','min_events','variation','edge','symmetry')}
                    | {'family':'repeat_plateau' if base['family']=='plateau' else 'repeat','tolerance':tol,'repeat_days':days}
                    for tol,days in itertools.product(self.config['repeat_tolerance_kw'],self.config['repeat_days'])]
            for config in configs:
                events=self.get_events(val,config);ef,p,chosen=event_features(val,events,config)
                selection=self.select(val,p,ef);c=selection['customers'];m=metrics(c.target,c.score,selection['threshold'])
                key=(m['pr_auc'],m['balanced_accuracy'],-m['brier'])
                search.append({'parameters':config,'metrics':m,'aggregations':selection['search']})
                if winner is None or key>winner[0]:winner=(key,config,p,selection,ef,chosen)
            _,config,p,selection,ef,chosen=winner
            self.finish(eid,family,{'block':'Repeated sustained residual loads distinguish chargers',
                'plateau':'Paired edges and flatness reject heating-like blocks','repeat':'Repeated power levels across days reject isolated appliance loads'}[family],
                config,val,p,selection,started,features=ef,events=chosen,extra={'bounded_search':search})
            previous=best_id;best_id=self.heuristic_champion([best_id,eid] if best_id else [eid])
            stale=stale+1 if best_id==previous else 0
            write_json(self.out/'heuristic_champion.json',{'id':best_id,'consecutive_non_improvements':stale})
            self.decision(f'{eid} complete; heuristic champion {best_id}. '+('Test repetition next.' if family=='plateau' else 'Continue mandatory families before optional complexity.'))
        if stale>=2:
            self.decision('E13 template-only detector and E14 high-pass/wavelet skipped: two successive full heuristic extensions failed to improve the simpler detector beyond paired uncertainty. Template evidence remains available in E21 features.')
        else:
            self.decision('E13 template similarity retained as E21 evidence only; three heuristic families complete. E14 skipped: no demonstrated drifting-baseline error pattern to justify another detector.')

    def heuristic_champion(self,ids):
        champion=ids[0]
        for eid in ids[1:]:
            a=pd.read_parquet(self.out/eid/'customer_validation.parquet');b=pd.read_parquet(self.out/champion/'customer_validation.parquet')
            delta=bootstrap(a.target,a.score,.5,n=self.config['bootstrap_draws'],other=b.score)
            if delta['ci95'][0]>0:champion=eid
        return champion

    def fit_model(self,train,x,parameters,val=None,v=None):
        from lightgbm import LGBMClassifier,early_stopping,log_evaluation
        model=LGBMClassifier(objective='binary',random_state=self.config['seed'],n_jobs=2,
            verbosity=-1,subsample_freq=1,**parameters)
        kwargs={}
        if val is not None:
            kwargs={'eval_set':[(v,val.temporal_target)],'eval_sample_weight':[customer_weights(val)],
                    'callbacks':[early_stopping(self.config['early_stopping_rounds'],verbose=False),log_evaluation(0)]}
        model.fit(x,train.temporal_target,sample_weight=customer_weights(train),**kwargs)
        return model

    def models(self):
        train=self.rows('development');val=self.rows('validation')
        heuristic=read_json(self.out/read_json(self.out/'heuristic_champion.json')['id']/'config.json')
        heuristic={k:v for k,v in heuristic.items() if k not in ('study_config','source_identity','code_hash','aggregation','threshold')}
        for eid,kind in [('E20_lgbm_compact','compact'),('E21_lgbm_events','events'),('E22_lgbm_shape','shape')]:
            if (self.out/eid).exists():continue
            started=time.perf_counter();winner=None;search=[]
            kinds=['fourier','shape'] if kind=='shape' else [kind]
            for feature_kind in kinds:
                x=self.features(train,feature_kind,heuristic);v=self.features(val,feature_kind,heuristic)
                for parameters in self.config['lgbm_grid']:
                    model=self.fit_model(train,x,parameters,val,v);p=model.predict_proba(v)[:,1]
                    selection=self.select(val,p);c=selection['customers'];m=metrics(c.target,c.score,selection['threshold'])
                    key=(m['pr_auc'],m['balanced_accuracy'],-m['brier'])
                    search.append({'features':feature_kind,'parameters':parameters,'best_iteration':model.best_iteration_,'metrics':m,'aggregation_search':selection['search']})
                    if winner is None or key>winner[0]:winner=(key,model,p,selection,x,feature_kind,parameters)
            _,model,p,selection,x,feature_kind,parameters=winner
            config={'kind':'lgbm','features':feature_kind,'heuristic':heuristic,'parameters':parameters,
                'refit_parameters':parameters|{'n_estimators':model.best_iteration_ or parameters['n_estimators']}}
            train_p=model.predict_proba(x)[:,1];tc=self.select(train,train_p,rules=[selection['aggregation']])['customers']
            self.finish(eid,'LightGBM '+feature_kind,{'compact':'Nonlinear interactions improve compact features',
                'events':'Charging-event evidence adds value beyond compact statistics',
                'shape':'Normalized Fourier and time-of-week shape adds transferable evidence'}[kind],
                config,val,p,selection,started,model,x,extra={'bounded_search':search,'training_customer_ap':float(average_precision_score(tc.target,tc.score))})
            importance=pd.DataFrame({'feature':x.columns,'gain':model.booster_.feature_importance('gain'),'split':model.booster_.feature_importance('split')})
            importance.sort_values('gain',ascending=False).to_csv(self.out/eid/'importance.csv',index=False)
            self.decision(f'{eid} finished; '+('test explicit event features next.' if kind=='compact' else 'compare normalized temporal shape with feature-family ablation.' if kind=='events' else 'major model families complete; inspect errors and compare calibration.'))

    def calibrate(self):
        eid='E24_calibration'
        if (self.out/eid).exists():return
        entries=read_json(self.out/'leaderboard.json')
        models=[e for e in entries if e['id'].startswith(('E02','E20','E21','E22'))]
        # Simpler-first paired selection among supervised models, separate from global dummy champion.
        base_id=self.heuristic_champion([e['id'] for e in models])
        config=read_json(self.out/base_id/'config.json')
        train=self.rows('development');val=self.rows('validation');started=time.perf_counter()
        x=self.features(train,config['features'],config.get('heuristic'))
        oof=np.zeros(len(train));groups=train.gp_nr.to_numpy()
        splitter=StratifiedGroupKFold(3,shuffle=True,random_state=self.config['seed'])
        for ti,vi in splitter.split(x,train.temporal_target,groups):
            assert not (set(groups[ti])&set(groups[vi]))
            if config['kind']=='logistic':
                model=make_pipeline(StandardScaler(),LogisticRegression(C=config['C'],max_iter=2000,random_state=self.config['seed']))
                model.fit(x.iloc[ti],train.temporal_target.iloc[ti],logisticregression__sample_weight=customer_weights(train.iloc[ti]))
            else:model=self.fit_model(train.iloc[ti],x.iloc[ti],config['refit_parameters'])
            oof[vi]=model.predict_proba(x.iloc[vi])[:,1]
        calibrator=LogisticRegression(C=100.,random_state=self.config['seed'])
        calibrator.fit(self.logits(oof),train.temporal_target,sample_weight=customer_weights(train))
        raw=pd.read_parquet(self.out/base_id/'weekly_validation.parquet').score.to_numpy()
        calibrated=calibrator.predict_proba(self.logits(raw))[:,1]
        searches=[];winner=None
        for method,p in [('uncalibrated',raw),('sigmoid',calibrated)]:
            selection=self.select(val,p);c=selection['customers'];m=metrics(c.target,c.score,selection['threshold'])
            searches.append({'method':method,'metrics':m,'aggregation_search':selection['search']})
            # Monotone calibration tied on AP: Brier breaks exact ties.
            key=(m['pr_auc'],-m['brier'])
            if winner is None or key>winner[0]:winner=(key,method,p,selection)
        _,method,p,selection=winner
        config={k:v for k,v in config.items() if k not in ('study_config','source_identity','code_hash','threshold','aggregation')}
        config.update(base_experiment=base_id,calibration=method,calibration_fit='3-fold grouped development OOF; final refit uses development+validation OOF')
        self.finish(eid,'calibration','Grouped out-of-fold sigmoid calibration improves probability reliability',config,val,p,selection,started,calibrator if method=='sigmoid' else None,x,
            extra={'calibration_search':searches,'isotonic':'skipped: fewer than 100 validation customers / 20 positives'})
        self.decision(f'E24 selected {method} on {base_id}; isotonic skipped for insufficient calibration population. E23 alternative tree skipped because LightGBM is available. No additional algorithm search justified.')

    def sensitivity(self):
        """Predeclared date-label sensitivity; does not change the primary result."""
        path=self.out/'date_sensitivity.json'
        if path.exists():return
        val=self.data.loc[self.data.split.eq('validation')].copy()
        config=read_json(self.out/'E02_logistic/config.json')
        with (self.out/'E02_logistic/model.pkl').open('rb') as f:model=pickle.load(f)
        p=model.predict_proba(self.features(val,'compact'))[:,1]
        val['score']=p;findings=[]
        definitions=[('primary',False,0),('early_late_fields',True,0),
                     ('washout_7_days',False,7),('washout_28_days',False,28)]
        for name,separate,washout in definitions:
            target,lo,hi=temporal_labels(val,separate_bounds=separate,washout=washout)
            current=val.copy();current['temporal_target']=target
            current=supervised(current)
            cohort=self.cohort_rows(current)
            scores=current.loc[cohort.index,'score'].to_numpy()
            customers=aggregate(cohort,scores,config['aggregation'])
            m=metrics(customers.target,customers.score,config['threshold'])
            findings.append({'definition':name,'customers':len(customers),
                'positive_customers':int(customers.target.sum()),'metrics':m,
                'unknown_eligible_weeks':int((val.eligible_for_supervised_training.eq(1)&target.isna()).sum())})
        write_json(path,{'hypothesis':'If boundary label noise dominates, 7/28-day washouts materially change validation performance.',
                         'result':findings,'decision':'Sensitivity only; primary pipeline and threshold unchanged.'})
        self.append('\n### E30 — Date-label sensitivity (targeted analysis)\n\n'
            '- Hypothesis: If project-boundary label noise dominates, early/late-only bounds or 7/28-day washouts materially change validation performance.\n'
            '- Method: frozen E02 weekly model and threshold; recompute only temporal targets and fixed-rule customer aggregation.\n'
            '- Result: [complete metrics](classification_output/date_sensitivity.json).\n'
            '- Decision: sensitivity analysis only; no feature, model, aggregation, or threshold was changed. Large differences are treated as date-label uncertainty.\n')
        self.decision('E30 date sensitivity completed. No post-hoc model change; freeze the simpler validation champion under uncertainty.')

    @staticmethod
    def logits(p):
        p=np.clip(p,1e-6,1-1e-6);return np.log(p/(1-p)).reshape(-1,1)

    def freeze(self):
        if (self.out/'final_freeze.json').exists():return
        eid=read_json(self.out/'champion.json')['id']
        baselines=['E01_dummy','E02_logistic',read_json(self.out/'heuristic_champion.json')['id']]
        ids=list(dict.fromkeys([eid,*baselines]))
        frozen={i:{'config':read_json(self.out/i/'config.json'),'feature_manifest':read_json(self.out/i/'feature_manifest.json')} for i in ids}
        record={'created_utc':datetime.now(UTC).isoformat(),'selected':eid,'pipelines':frozen,
            'source_identity':source_identity()['hash'],'minimum_history':self.config['minimum_history'],
            'history_cap':self.config['history_cap'],'stop_reason':'Major heuristic and LightGBM families evaluated; prefer simpler pipelines within paired AP uncertainty.',
            'test_status':'unopened: no test predictions computed','code_hash':digest({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')})}
        record['freeze_hash']=digest(record);write_json(self.out/'final_freeze.json',record)
        self.append('\n## Decision log\n\n'+ '\n'.join('- '+e['decision'] for e in read_json(self.out/'decisions.json'))+'\n\n## Final freeze\n\nFINAL FREEZE — written before test prediction.\n\n```json\n'+json.dumps(record,indent=2)+'\n```\n')
        print('FROZEN',eid,record['freeze_hash'],flush=True)

    def refit_predict(self,config,train,target):
        kind=config.get('kind','heuristic')
        if kind=='dummy':
            prevalence=float(np.average(train.temporal_target,weights=customer_weights(train)))
            return np.full(len(target),prevalence),{'prevalence':prevalence},None
        if kind=='heuristic':
            ef,p,_=event_features(target,self.get_events(target,config),config)
            return p,config,ef
        x=self.features(train,config['features'],config.get('heuristic'))
        v=self.features(target,config['features'],config.get('heuristic'))
        def fit(ti):
            if kind=='logistic':
                model=make_pipeline(StandardScaler(),LogisticRegression(C=config['C'],max_iter=2000,random_state=self.config['seed']))
                model.fit(x.iloc[ti],train.temporal_target.iloc[ti],logisticregression__sample_weight=customer_weights(train.iloc[ti]));return model
            return self.fit_model(train.iloc[ti],x.iloc[ti],config['refit_parameters'])
        model=fit(np.arange(len(train)));p=model.predict_proba(v)[:,1];calibrator=None
        if config.get('calibration')=='sigmoid':
            oof=np.zeros(len(train));groups=train.gp_nr.to_numpy()
            for ti,vi in StratifiedGroupKFold(3,shuffle=True,random_state=self.config['seed']).split(x,train.temporal_target,groups):
                assert not (set(groups[ti])&set(groups[vi]));oof[vi]=fit(ti).predict_proba(x.iloc[vi])[:,1]
            calibrator=LogisticRegression(C=100.,random_state=self.config['seed']).fit(self.logits(oof),train.temporal_target,sample_weight=customer_weights(train))
            p=calibrator.predict_proba(self.logits(p))[:,1]
        return p,{'model':model,'calibrator':calibrator},None

    def final(self):
        frozen=read_json(self.out/'final_freeze.json')
        assert frozen['source_identity']==source_identity()['hash']
        directory=self.out/'final_test'
        assert not directory.exists(), 'Final test already started/evaluated: refuse to reopen'
        directory.mkdir();write_json(directory/'opened.json',{'time':datetime.now(UTC).isoformat(),'freeze_hash':frozen['freeze_hash']})
        train=self.rows(['development','validation']);test=self.rows('test');result={};predictions={}
        for eid,record in frozen['pipelines'].items():
            config=record['config'];p,model,ef=self.refit_predict(config,train,test)
            cohort=self.cohort_rows(test);positions=test.index.get_indexer(cohort.index)
            customers=aggregate(cohort,p[positions],config['aggregation'],ef)
            customers['decision']=customers.score.ge(config['threshold']).astype(int)
            predictions[eid]=customers
            customers.to_parquet(directory/f'{eid}_customers.parquet',index=False)
            weekly=test[['gp_nr','week_start','temporal_target']].copy();weekly['score']=p
            weekly.to_parquet(directory/f'{eid}_weeks.parquet',index=False)
            result[eid]={'customer':metrics(customers.target,customers.score,config['threshold']),
                'week':metrics(test.temporal_target,p,config['threshold']),
                'ci95':bootstrap(customers.target,customers.score,config['threshold'],n=self.config['bootstrap_draws'])}
            if eid==frozen['selected']:
                best=self.out/'best';best.mkdir();write_json(best/'config.json',config)
                write_json(best/'feature_manifest.json',record['feature_manifest'])
                with (best/'pipeline.pkl').open('wb') as f:pickle.dump(model,f)
                if isinstance(model,dict) and hasattr(model.get('model'),'booster_'):model['model'].booster_.save_model(str(best/'model.txt'))
                customers.to_parquet(best/'customer_predictions.parquet',index=False)
                weekly.to_parquet(best/'weekly_predictions.parquet',index=False)
        chosen=predictions[frozen['selected']]
        for eid,customers in predictions.items():
            assert customers.gp_nr.tolist()==chosen.gp_nr.tolist()
            result[eid]['selected_minus_baseline']=bootstrap(chosen.target,chosen.score,.5,n=self.config['bootstrap_draws'],other=customers.score)
        write_json(directory/'metrics_test.json',result);write_json(self.out/'best/metrics_test.json',result)
        write_json(self.out/'best/metrics.json',result)
        selected=frozen['selected'];selected_dir=self.out/selected
        shutil.copy2(selected_dir/'slices.csv',self.out/'best/metrics_by_slice.csv')
        if (selected_dir/'model.txt').exists():shutil.copy2(selected_dir/'model.txt',self.out/'best/model.txt')
        # Stable, inspectable feature and event caches for reproducibility.
        feature_cfg=frozen['pipelines'][selected]['config']
        all_features=self.features(self.data,feature_cfg.get('features','compact'),feature_cfg.get('heuristic'))
        pd.concat([self.data[['gp_nr','week_start','split','temporal_target']],all_features],axis=1).to_parquet(self.out/'weekly_features.parquet',index=False)
        heuristic_cfg=read_json(self.out/read_json(self.out/'heuristic_champion.json')['id']/'config.json')
        write_json(self.out/'best/heuristic_config.json',heuristic_cfg)
        all_events=self.get_events(self.data,heuristic_cfg).merge(self.data[['gp_nr','week_start']],left_on='row',right_index=True,how='left')
        all_events.to_parquet(self.out/'heuristic_events.parquet',index=False)
        self.append('\n## Final test\n\nOne-time held-out customer evaluation after refitting on development plus validation.\n\n```json\n'+json.dumps(result,indent=2)+'\n```\n')
        self.exploratory(frozen)
        manifest=read_json(self.out/'run_manifest.json');manifest['test_status']='evaluated_once';write_json(self.out/'run_manifest.json',manifest)
        self.append('\n## Recommendation and limitations\n\n'
            f'Selected pipeline: `{frozen["selected"]}`; deployable local configuration and fitted objects are in [best/](classification_output/best/). '
            'Treat this as a small-cohort research result, not verified ownership. Customer AP is the ranking criterion; the threshold optimizes validation balanced accuracy. '
            'A charger can be unused; heating, boilers and other loads can mimic charging; PV and batteries can mask it. '
            'Project dates are not confirmed commissioning dates. Only a subset of labelled customers has usable data. '
            'Repeated validation searches and the small number of positive customers create substantial selection uncertainty. '
            'Use weekly/event evidence for review; insufficient-history customers abstain, and excluded/unknown rows are exploratory only. '
            'No pipeline changes were made in response to final-test outcomes.\n')
        print('FINAL',json.dumps(result),flush=True)

    def exploratory(self,frozen):
        config=frozen['pipelines'][frozen['selected']]['config']
        unknown=self.data.loc[~(self.data.eligible_for_supervised_training.eq(1)&self.data.temporal_target.notna())]
        # Apply already fitted pipeline; no refitting or supervised metrics on these rows.
        with (self.out/'best/pipeline.pkl').open('rb') as f:bundle=pickle.load(f)
        def predict(rows):
            kind=config.get('kind','heuristic')
            if kind=='dummy':return np.full(len(rows),bundle['prevalence'])
            if kind=='heuristic':return event_features(rows,self.get_events(rows,config),config)[1]
            p=bundle['model'].predict_proba(self.features(rows,config['features'],config.get('heuristic')))[:,1]
            if bundle.get('calibrator') is not None:p=bundle['calibrator'].predict_proba(self.logits(p))[:,1]
            return p
        out=unknown[['gp_nr','week_start','eligible_for_supervised_training','temporal_target','training_exclusion_reason']].copy()
        out['score']=predict(unknown);out['status']='exploratory';out.to_parquet(self.out/'best/exploratory_predictions.parquet',index=False)
        labelled=supervised(self.data)
        assert len(labelled)+len(out)==len(self.data)
        write_json(self.out/'best/prediction_reconciliation.json',{'source_rows':len(self.data),'supervised_rows':len(labelled),'exploratory_rows':len(out),'reconciles':True})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['audit','baselines','heuristics','models','calibrate','sensitivity','freeze','final'])
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();study=Study(args.smoke);getattr(study,args.stage)()


if __name__=='__main__':main()
