"""Regression tests and a real-format, isolated end-to-end fixture."""
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
from . import wpb, wpb_core as W


def test_energy_conservation_across_multiple_midnights():
    donor=np.full((12,96),.05)
    donor[:,90:96] += .75
    p=W.find_plateaus(*W.day_quantile_profiles(donor)[:2])[0]
    host=np.zeros((12,96))
    result=wpb._retrofit(host,donor,p,.1,np.random.default_rng(0))
    np.testing.assert_allclose(result.sum(axis=1),4.5)


def test_noise_bands_with_identical_values():
    assert wpb.noise_bands(pd.Series([0.,0.,0.])).astype(str).tolist()==['quiet']*3


def test_hash_sample_is_process_stable():
    code='import zlib; print([str(i) for i in range(1000) if zlib.crc32(str(i).encode()) % 20 == 0])'
    values=[subprocess.check_output([sys.executable,'-c',code],env={**os.environ,'PYTHONHASHSEED':seed}) for seed in ['1','42']]
    assert values[0]==values[1]


def test_all_six_stages_and_slides(tmp_path):
    # This file never touches the real exports or artifacts.
    root=tmp_path
    data=root/'newspaper'/'data_addition'; data.mkdir(parents=True)
    pd.DataFrame({'GP-Nr':['1','2','3','4'],'Wärmepumpenboiler':['x']*4}).to_csv(data/'HackDays2026 - GIGI - annotated.csv',sep=';',index=False,encoding='utf-8-sig')
    pd.DataFrame(dict(gp_nr=['1','2','3','4'],mp_id=['1','2','3','4'],plz=['5000']*4,ort=['']*4,kanton=['AG']*4)).to_csv(data/'gigi_augmented.csv',index=False)
    rng=np.random.default_rng(19)
    for ym,month_name in [('2025-07','Juli'),('2025-01','Januar')]:
        folder=root/'store'/'input_data'/'2025'/f'{month_name} 2025'; folder.mkdir(parents=True)
        with (folder/'LG_AIM2Hackerdays_kWh_fixture.csv').open('w') as f:
            # Deliberately missing OBIS header; data columns must stay positional.
            f.write(';'.join(['MP ID','Datum','PLZ',*[str(i) for i in range(96)]])+';\n')
            for mp in range(1,25):
                for day in range(1,29):
                    vals=np.clip(.05+rng.normal(0,.005,96),0,None)
                    if mp<=4: vals[np.arange(92,108)%96]+=.125
                    if mp==5: vals[np.arange(92,98)%96]+=.75
                    for obis,v in [('1-1:1.29.0*255',vals),('1-1:2.29.0*255',np.full(96,99.))]:
                        f.write(';'.join([str(mp),obis,f'{day:02d}.{ym[-2:]}.2025','5000',*[f'{x:.6f}' for x in v]])+';\n')
    env={**os.environ,'PV_WORK_ROOT':str(root)}
    repo=Path(__file__).resolve().parents[2]
    def run(*args):
        subprocess.run([sys.executable,'-m','models.heat_pump_boiler.'+args[0],*args[1:]],cwd=repo,env=env,check=True,capture_output=True,text=True)
    run('wpb','profiles','--month','2025-07','--month','2025-01')
    run('wpb','noise','--month','2025-07')
    for stage in ['labelled','detect','figures']:
        run('wpb',stage,'--summer','2025-07','--winter','2025-01')
    run('wpb','inject','--month','2025-07')
    run('wpb','figures','--summer','2025-07','--winter','2025-01')
    run('wpb_slides')
    artifacts=root/'newspaper'/'models'/'heat_pump_boiler'/'artifacts'
    predictions=pd.read_csv(artifacts/'wpb_predictions.csv')
    assert (predictions[predictions.mp_id<=4].wpb_score>.8).all()
    assert len(predictions)==24
    assert (artifacts/'wpb_slides.pptx').stat().st_size>1000
    z=np.load(artifacts/'wpb_profiles_2025-07.npz')
    assert z['p50'].max()<1  # 99 kWh export channel did not leak.
