"""Resumeable full-export sweep and annual, interpretable WPB evidence.

Run after the initial noise gate and labelled inspection:
python -m newspaper.models.heat_pump_boiler.pipeline --workers 4
The final prediction refers to the latest January/July pair. Earlier years
and all monthly profiles are retained for temporal stability assessment.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import time
from types import SimpleNamespace
import numpy as np
import pandas as pd
from . import config as C, wpb, wpb_core as W
from ..pv.io import gp_to_mpid


def profile_month(ym):
    out=Path(str(wpb.PROFILES).format(ym=ym))
    start=time.monotonic()
    cached=False
    if out.exists():
        with np.load(out) as z:
            cached=('profile_version' in z and int(z['profile_version'])==2
                    and 'sample_every' in z and int(z['sample_every'])==1
                    and not bool(z['labelled_only']))
    if not cached:
        wpb.build_profiles(ym,keep_raw=2000 if ym.endswith('-07') else 0)
    z=wpb.load_profiles(ym)
    rows=[]
    for i,mp in enumerate(z['mp_id']):
        ps=W.find_plateaus(z['p10'][i],z['p50'][i])
        best=next((p for p in ps if W.classify(p)=='wpb_candidate'),None)
        resistance=next((p for p in ps if W.classify(p)=='resistance_boiler'),None)
        rows.append(dict(mp_id=mp,plz=z['plz'][i],month=ym,n_days=int(z['n_days'][i]),
            night_noise_kwh_slot=float(z['noise'][i]),wpb_candidate=best is not None,
            amplitude_kw=best.amplitude_kw if best else np.nan,
            duration_h=best.duration_h if best else np.nan,
            start_slot=best.start_slot if best else np.nan,
            daily_kwh=best.energy_kwh if best else np.nan,
            resistance_amplitude_kw=resistance.amplitude_kw if resistance else np.nan))
    d=pd.DataFrame(rows)
    d.to_pickle(C.ARTIFACTS/f'wpb_monthly_{ym}.pkl')
    return dict(month=ym,n_meters=len(d),n_candidates=int(d.wpb_candidate.sum()),
                n_resistance=int(d.resistance_amplitude_kw.notna().sum()),
                quiet_fraction=float((d.night_noise_kwh_slot<.0625).mean()),
                cached=cached,seconds=round(time.monotonic()-start,1))


def household_rollup(d):
    links=gp_to_mpid()[['gp_nr','mp_id']].dropna().drop_duplicates()
    # Ambiguous meter->GP links cannot identify one household; report exclusions.
    counts=links.groupby('mp_id').gp_nr.nunique()
    ambiguous=set(counts[counts>1].index)
    usable=links[~links.mp_id.isin(ambiguous)]
    merged=d.merge(usable,on='mp_id',how='inner',validate='one_to_one')
    best=merged.sort_values(['wpb_score','mp_id'],ascending=[False,True]).drop_duplicates('gp_nr')
    out=best[['gp_nr','mp_id','plz','wpb_score','reason']].rename(columns={'mp_id':'evidence_mp_id'})
    out=out.merge(merged.groupby('gp_nr').size().rename('n_meters'),on='gp_nr')
    out['wpb_flag']=out.wpb_score>=.8
    out.to_csv(C.ARTIFACTS/'wpb_household_predictions.csv',index=False)
    return dict(n_households=len(out),n_flagged=int(out.wpb_flag.sum()),
                n_unmapped_meters=int((~d.mp_id.isin(links.mp_id)).sum()),
                n_ambiguous_meters=int(d.mp_id.isin(ambiguous).sum()))


def finish():
    from . import build_baserate
    months=sorted(e.ym for e in C.discover_monthly_exports())
    years=sorted(set(m[:4] for m in months if m.endswith('-07') and m[:4]+'-01' in months))
    annual=[]
    for year in years:
        args=SimpleNamespace(summer=year+'-07',winter=year+'-01',threshold=.8,mark_labelled=True)
        wpb.cmd_detect(args)
        d=pd.read_csv(wpb.PREDICTIONS,dtype={'mp_id':str,'plz':str},low_memory=False)
        d['year']=int(year)
        annual.append(d)
        d.to_csv(C.ARTIFACTS/f'wpb_predictions_{year}.csv',index=False)
        shutil.copyfile(wpb.EVIDENCE,C.ARTIFACTS/f'wpb_evidence_{year}.json')
    all_years=pd.concat(annual,ignore_index=True)
    all_years.to_csv(C.ARTIFACTS/'wpb_predictions_annual.csv',index=False)
    paired=all_years[all_years.has_winter_month].copy()
    paired['flag']=paired.wpb_score>=.8
    stability=paired.groupby('mp_id').agg(n_paired_years=('year','size'),n_flagged_years=('flag','sum'))
    stability['fraction_flagged']=stability.n_flagged_years/stability.n_paired_years
    stability.to_csv(C.ARTIFACTS/'wpb_temporal_stability.csv')
    rollup=household_rollup(annual[-1])
    (C.ARTIFACTS/'wpb_household_coverage.json').write_text(json.dumps(rollup,indent=2))
    latest=years[-1]
    args=SimpleNamespace(summer=latest+'-07',winter=latest+'-01')
    wpb.cmd_labelled(args)
    wpb.cmd_inject(SimpleNamespace(month=latest+'-07',amplitudes=[.3,.5,.8,1.2],n_hosts=150))
    wpb.cmd_figures(args)
    build_baserate.compare()
    print(json.dumps(rollup,indent=2))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--finish-only',action='store_true')
    args=ap.parse_args()
    gate=C.ARTIFACTS/'wpb_noise_floor_stage0.csv'
    if not gate.exists() or pd.read_csv(gate).frac_meters_below_half_wpb.iloc[0]<=.5:
        raise RuntimeError('A passing stage-0 noise gate is required before the full sweep')
    if not args.finish_only:
        rows=[]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs={pool.submit(profile_month,e.ym):e.ym for e in C.discover_monthly_exports()}
            for job in as_completed(jobs):
                row=job.result(); rows.append(row); print(row,flush=True)
                pd.DataFrame(rows).sort_values('month').to_csv(C.ARTIFACTS/'wpb_sweep_summary.csv',index=False)
    finish()

if __name__=='__main__': main()
