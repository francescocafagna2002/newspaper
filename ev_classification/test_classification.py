import unittest
import numpy as np
import pandas as pd

from ev_classification.classification_core import (
    DATES, LOADS, aggregate, check_features, customer_weights, kw, metrics,
    split_customers, supervised, temporal_labels,
)


class ClassificationContracts(unittest.TestCase):
    def base(self):
        return pd.DataFrame({
            'gp_nr':['a','a','b','b'], 'week_start':pd.to_datetime(['2024-01-01','2024-03-04']*2),
            'ev_label':[1,1,0,0], 'eligible_for_supervised_training':[1,1,1,1],
            DATES[0]:pd.to_datetime(['2024-02-01']*4), DATES[1]:pd.NaT,
            DATES[2]:pd.to_datetime(['2024-03-01']*4), DATES[3]:pd.NaT,
        })

    def test_temporal_labels_and_unknown_boundaries(self):
        d=self.base();target,lo,hi=temporal_labels(d)
        self.assertEqual(target.tolist(),[0,1,0,0])
        d.loc[0,'week_start']=pd.Timestamp('2024-01-29')
        self.assertTrue(np.isnan(temporal_labels(d)[0].iloc[0]))
        d.loc[0,DATES[0]]=pd.NaT;d.loc[0,DATES[2]]=pd.NaT
        self.assertTrue(np.isnan(temporal_labels(d)[0].iloc[0]))

    def test_gate_excludes_no_train_rows(self):
        d=self.base();d['temporal_target']=temporal_labels(d)[0]
        d.loc[0,'eligible_for_supervised_training']=0
        self.assertNotIn(0,supervised(d).index)

    def test_split_is_deterministic_and_disjoint(self):
        d=pd.concat([self.base().assign(gp_nr=str(i),ev_label=i%2) for i in range(20)],ignore_index=True)
        d['temporal_target']=temporal_labels(d)[0]
        one=split_customers(d,7);two=split_customers(d,7)
        pd.testing.assert_frame_equal(one,two)
        self.assertTrue(one.gp_nr.is_unique)
        self.assertEqual(set(one['split']),{'development','validation','test'})

    def test_feature_exclusion(self):
        check_features(pd.DataFrame({'stat_mean':[1.]}))
        with self.assertRaises(AssertionError):check_features(pd.DataFrame({'ev_label':[1]}))
        with self.assertRaises(AssertionError):check_features(pd.DataFrame({'mystery':[1]}))

    def test_kwh_to_kw(self):
        np.testing.assert_allclose(kw([.25,1.,-2.]),[1.,4.,-8.])

    def test_customer_balanced_weights(self):
        d=pd.DataFrame({'gp_nr':['a','a','a','b']})
        w=customer_weights(d)
        self.assertAlmostEqual(w[:3].sum(),w[3:].sum())
        self.assertAlmostEqual(w.mean(),1.)

    def test_aggregation_and_scoring(self):
        weeks=pd.DataFrame({'gp_nr':['a','a','a','b','b'],
            'week_start':pd.date_range('2024-01-01',periods=5,freq='7D'),
            'temporal_target':[1,1,1,0,0]})
        result=aggregate(weeks,[.2,.9,.7,.1,.3],'top3')
        self.assertEqual(result.gp_nr.tolist(),['a','b'])
        self.assertAlmostEqual(result.loc[0,'score'],.6)
        self.assertEqual(metrics(result.target,result.score,.5)['confusion_matrix'],[[1,0],[0,1]])

    def test_load_column_order(self):
        self.assertEqual(len(LOADS),672)
        self.assertEqual(LOADS[0],'net_load_mon_00_00_kwh')
        self.assertEqual(LOADS[-1],'net_load_sun_23_45_kwh')


if __name__=='__main__':unittest.main()
