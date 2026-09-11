"""Independent postcode comparison against BFS GWR hot-water equipment.

GWAERZW1=7610 is the primary hot-water heat generator, labelled exactly
"Wärmepumpe". GWR has no Wärmepumpenboiler code at all: its only boiler
codes are 7650 "Zentraler Elektroboiler" and 7651 "Kleinboiler", both
resistance. So 7610 alone is not a WPB count -- in AG, 96% of 7610
buildings also have GWAERZH1 in {7410, 7411}, i.e. one heat pump serving
both space heating and hot water.

Hence two rates are produced:
  gwr_hp_rate           all 7610 -- the broad category, ~21% of buildings
  gwr_separate_wpb_rate 7610 AND space heating NOT a heat pump -- a
                        dedicated hot-water heat pump must exist, ~0.6%

The second is the like-for-like comparator. It is still a lower bound: a
WPB in a building that also has a space-heating heat pump is coded 7610 /
7410 and is absorbed into the combined bucket.
GENW1 describes the energy source, not the generator. Neither is GENH1.
"""
from __future__ import annotations
import csv
import hashlib
import json
import zipfile
from datetime import datetime, timezone

import pandas as pd
import requests
from scipy.stats import spearmanr
from . import config as C

URL = 'https://public.madd.bfs.admin.ch/ag.zip'
CATALOG = 'https://www.housing-stat.ch/catalog/data/4.3/de.html'
BASE_RATE = C.ARTIFACTS / 'wpb_gwr_baserate_plz.csv'


def build():
    cache = C.ARTIFACTS / 'gwr_ag.zip'
    if not cache.exists():
        response = requests.get(URL, timeout=180)
        response.raise_for_status()
        cache.write_bytes(response.content)
    with zipfile.ZipFile(cache) as z:
        def read(name, cols=None):
            return pd.read_csv(z.open(name), sep='\t', dtype=str,
                               quoting=csv.QUOTE_NONE, usecols=cols).fillna('')
        codes = read('kodes_codes_codici.csv')
        hp = codes[(codes.CMERKM == 'GWAERZW1') & (codes.CECODID == '7610')]
        if hp.empty or not hp.CODTXTLD.str.contains('Wärmepumpe').all():
            raise ValueError('GWR generator code no longer matches the verified catalogue')
        boiler = codes[(codes.CMERKM == 'GWAERZW1') & codes.CODTXTLD.str.contains('boiler', case=False)]
        if boiler.CODTXTLD.str.contains('Wärmepumpe').any():
            raise ValueError('GWR now has a heat-pump boiler code; use it directly')
        b = read('gebaeude_batiment_edificio.csv',
                 ['EGID','GDEKT','GSTAT','GKAT','GWAERZH1','GWAERZW1','GENW1','GEXPDAT'])
        e = read('eingang_entree_entrata.csv', ['EGID','EDID','DPLZ4'])
    # One building contributes once, even if it has several entrances.
    postal_counts = e.groupby('EGID').DPLZ4.nunique()
    ambiguous = set(postal_counts[postal_counts > 1].index)
    e = e[~e.EGID.isin(ambiguous)].sort_values(['EGID','EDID']).drop_duplicates('EGID')
    b = b[(b.GDEKT == 'AG') & (b.GSTAT == '1004') & b.GKAT.isin(['1020','1030','1040'])]
    b = b.merge(e[['EGID','DPLZ4']], on='EGID', validate='one_to_one')
    b = b[b.DPLZ4.str.fullmatch(r'\d{4}')].copy()
    b['heat_pump'] = b.GWAERZW1.eq('7610')
    # Space heating by the same heat pump -> not a standalone water heater.
    b['separate_wpb'] = b.heat_pump & ~b.GWAERZH1.isin(['7410','7411'])
    b['small_air_boiler'] = b.GWAERZW1.eq('7651') & b.GENW1.eq('7501')
    b['known'] = ~b.GWAERZW1.isin(['','7699'])
    out = b.groupby('DPLZ4').agg(n_buildings=('EGID','size'),
        n_hotwater_hp=('heat_pump','sum'), n_separate_wpb=('separate_wpb','sum'),
        n_known=('known','sum'),
        n_small_air_boiler=('small_air_boiler','sum')).rename_axis('plz').reset_index()
    out['gwr_hp_rate'] = out.n_hotwater_hp / out.n_buildings
    out['gwr_separate_wpb_rate'] = out.n_separate_wpb / out.n_buildings
    out['gwr_hp_rate_known'] = out.n_hotwater_hp / out.n_known.replace(0,float('nan'))
    out['gwr_hp_rate_including_small_air'] = (out.n_hotwater_hp + out.n_small_air_boiler)/out.n_buildings
    out.to_csv(BASE_RATE,index=False)
    meta = dict(source=URL, catalogue=CATALOG,
        accessed_utc=datetime.now(timezone.utc).isoformat(),
        sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
        export_dates=sorted(b.GEXPDAT.unique()),
        n_existing_residential_buildings=len(b), n_postcodes=len(out),
        n_ambiguous_postcode_buildings_excluded=len(ambiguous),
        generator_field='GWAERZW1',generator_code=7610,
        generator_code_label=hp.CODTXTLD.iloc[0],
        n_hotwater_hp=int(b.heat_pump.sum()),
        n_separate_wpb=int(b.separate_wpb.sum()),
        share_of_7610_that_is_combined=float(1 - b.separate_wpb.sum()/max(b.heat_pump.sum(),1)),
        caution='GWR has no Wärmepumpenboiler code. 7610 is plain "Wärmepumpe" and mostly marks '
                'combined space-heating/DHW machines; gwr_separate_wpb_rate is the like-for-like '
                'comparator and is itself a lower bound. Ecological comparison, not WPB accuracy.')
    (C.ARTIFACTS/'wpb_gwr_provenance.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta,indent=2))
    return out


def compare(predictions=None, threshold=0.8):
    base = pd.read_csv(BASE_RATE,dtype={'plz':str}) if BASE_RATE.exists() else build()
    if predictions is None:
        predictions = pd.read_csv(C.ARTIFACTS/'wpb_predictions.csv',dtype={'plz':str},
                                  low_memory=False)
    # Use meters with both seasons to avoid confusing rollout with absence.
    d=predictions[predictions.has_winter_month.astype(str).str.lower().eq('true')].copy()
    d['flag'] = d.wpb_score >= threshold
    rate=d.groupby('plz').agg(n_meters=('flag','size'),n_flagged=('flag','sum')).reset_index()
    rate['flag_rate']=rate.n_flagged/rate.n_meters
    out=rate.merge(base,on='plz',validate='one_to_one')
    out.to_csv(C.ARTIFACTS/'wpb_gwr_comparison.csv',index=False)
    results=[]
    for min_n in [1,30,100]:
        sub=out[out.n_meters>=min_n]
        for field in ['gwr_hp_rate','gwr_separate_wpb_rate','gwr_hp_rate_known',
                      'gwr_hp_rate_including_small_air']:
            valid=sub[['flag_rate',field]].dropna()
            r=spearmanr(valid.flag_rate,valid[field]) if len(valid)>2 else (float('nan'),float('nan'))
            results.append(dict(min_meters=min_n,reference=field,n_postcodes=len(valid),rho=float(r[0]),pvalue=float(r[1])))
    pd.DataFrame(results).to_csv(C.ARTIFACTS/'wpb_gwr_correlations.csv',index=False)
    print(pd.DataFrame(results).to_string(index=False))
    return out

if __name__=='__main__':
    build()
    if (C.ARTIFACTS/'wpb_predictions.csv').exists(): compare()
