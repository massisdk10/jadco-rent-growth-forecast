"""Tests de contrat Model A et mutations synthétiques du futur."""
from pathlib import Path
import unittest
from unittest.mock import patch
import subprocess
import numpy as np
import pandas as pd
from src.features.model_a_features import (build_model_a_features, assert_foundation_reference,
    select_model_a_features, DEFERRED_FEATURES)
from src.features.model_dataset import ADMISSIBLE_FEATURES, OCCURRENCE_FEATURES, FORBIDDEN_FEATURES


def fixture():
    rows=[]
    for unit in ['001','002']:
        for year in [2022,2023]:
            rows.append({'sPropCode':'P1','sUnitCode':unit,'sLeaseFrom':f'{year}-07-01',
                'sSignDate':f'{year}-06-15','sLeaseTo':f'{year+1}-06-30',
                'sRent':1000 if year==2022 else 1100,
                'sRentEffective':900 if year==2022 else 990,
                'sBuilding':'B1','sState':'Quebec','sBeds':2,'sBaths':1,
                'sSqft':800,'sFloor':2,'sUnitSubtype':'A','sTermMonths':12,
                'sTermSeq':year-2021,'sRenewal':1,'sConcession':1})
    return pd.DataFrame(rows)


class ModelAFeaturesTests(unittest.TestCase):
    def test_reference_and_frozen_sources(self):
        root=Path(__file__).resolve().parents[1]
        raw=pd.read_csv(root/'data/raw/equinoxe_lease_history.csv',dtype={'sPropCode':'string','sUnitCode':'string'})
        result=assert_foundation_reference(raw)
        self.assertEqual(result.candidates.tolist(),[405,645,856,931])
        self.assertEqual(result.observed_units.iloc[:3].tolist(),[401,634,773])
        for f in ['docs/modeling_contract.md','src/features/model_dataset.py',
                  'src/evaluation/backtest.py','src/same_unit.py','src/rent_economics.py','src/external_data.py']:
            frozen=subprocess.run(['git','show',f'foundation-v1:{f}'],capture_output=True,cwd=root,check=True).stdout
            self.assertEqual(frozen,(root/f).read_bytes(),f)

    def test_correct_origin_and_schema(self):
        a=build_model_a_features(fixture(),2023)
        self.assertTrue(a.audit.prediction_origin.eq(pd.Timestamp('2022-12-31')).all())
        self.assertEqual(tuple(a.X.columns),OCCURRENCE_FEATURES)
        self.assertEqual(tuple(a.X_growth.columns),ADMISSIBLE_FEATURES)
        self.assertFalse(set(a.X)&set(FORBIDDEN_FEATURES))
        self.assertFalse(set(a.X)&set(a.audit))
        with self.assertRaises(ValueError):build_model_a_features(fixture(),2023,prediction_origin='2023-01-01')

    def test_future_mutations_do_not_affect_any_feature(self):
        raw=fixture();a=build_model_a_features(raw,2023)
        mutated=raw.copy();mask=mutated.sLeaseFrom.ge('2023-01-01')
        for col in ['sRent','sRentEffective','sSqft','sConcession','sRenewal','sTermMonths']:
            mutated.loc[mask,col]=999999
        mutated.loc[mask,'sBuilding']='FUTUR';mutated.loc[mask,'sLeaseTo']='2099-01-01'
        b=build_model_a_features(mutated,2023)
        pd.testing.assert_frame_equal(a.X,b.X)
        pd.testing.assert_frame_equal(a.audit,b.audit)
        past=raw.loc[~mask]
        pd.testing.assert_frame_equal(a.X,build_model_a_features(past,2023).X)

    def test_labels_separate_and_fraction_convention(self):
        raw=fixture();a=build_model_a_features(raw,2023,include_targets=True)
        self.assertEqual(a.labels.transition_occurred.tolist(),[1,1])
        np.testing.assert_allclose(a.labels.conditional_growth,1.1**(365.25/365)-1)
        self.assertFalse(set(a.labels)&set(a.X))
        no_future=build_model_a_features(raw.loc[raw.sLeaseFrom.lt('2023')],2023,include_targets=True)
        self.assertEqual(no_future.labels.transition_occurred.tolist(),[0,0])
        self.assertTrue(no_future.labels.conditional_growth.isna().all())
        pd.testing.assert_frame_equal(a.X,no_future.X)

    def test_no_label_engine_called_without_targets(self):
        with patch('src.features.model_a_features.build_model_dataset',side_effect=AssertionError('Futur consulté')):
            result=build_model_a_features(fixture(),2023)
            self.assertIsNone(result.labels)

    def test_unqualified_economic_features_never_constructed(self):
        a=build_model_a_features(fixture(),2023)
        for feature in DEFERRED_FEATURES:
            self.assertNotIn(feature,a.X)
            with self.assertRaises(ValueError):select_model_a_features(a,[feature])
        # Rent position/historiques/dynamiques ne peuvent fuiter : ils restent
        # absents, même si des historiques et prix effectifs sont présents.
        select_model_a_features(a,['expiry_month'],purpose='occurrence')
        with self.assertRaises(ValueError):select_model_a_features(a,['expiry_month'],purpose='growth')

    def test_external_optional_verified_only_and_never_added_to_X(self):
        external=pd.DataFrame({'available_date':['2022-12-31','2023-01-01',None,'2022-01-01'],
            'availability_verified':[True,True,True,False],'value':[1,2,3,4]})
        a=build_model_a_features(fixture(),2023,external=external)
        self.assertEqual(a.external_context.index.tolist(),[0])
        pd.testing.assert_frame_equal(a.X,build_model_a_features(fixture(),2023).X)

    def test_2026_without_target_or_future_information(self):
        raw=fixture().iloc[:2].copy()
        raw['sLeaseFrom']=['2024-07-01','2025-07-01'];raw['sSignDate']=['2024-06-15','2025-06-15']
        raw['sLeaseTo']=['2025-06-30','2026-06-30']
        a=build_model_a_features(raw,2026)
        self.assertEqual(len(a.X),1);self.assertIsNone(a.labels)
        future=raw.iloc[-1:].copy();future['sLeaseFrom']='2026-07-01';future['sSignDate']='2026-06-15'
        pd.testing.assert_frame_equal(a.X,build_model_a_features(pd.concat([raw,future]),2026).X)
        with self.assertRaises(ValueError):build_model_a_features(raw,2026,include_targets=True)

    def test_missing_features_keep_candidates(self):
        raw=fixture().drop(columns=['sFloor','sSqft'])
        a=build_model_a_features(raw,2023)
        self.assertEqual(len(a.X),2)
        self.assertTrue(a.X.floor.isna().all());self.assertTrue(a.X.sqft.isna().all())

    def test_deterministic_and_input_unchanged(self):
        raw=fixture();before=raw.copy(deep=True)
        a=build_model_a_features(raw,2023,include_targets=True)
        b=build_model_a_features(raw.iloc[::-1],2023,include_targets=True)
        pd.testing.assert_frame_equal(a.X,b.X);pd.testing.assert_frame_equal(a.audit,b.audit)
        pd.testing.assert_frame_equal(a.labels,b.labels);pd.testing.assert_frame_equal(raw,before)

    def test_enriched_core_is_identical(self):
        from src.features.model_a_features import CORE_FEATURES, ENRICHED_FEATURES
        raw=fixture()
        core=build_model_a_features(raw,2023,include_targets=True)
        enriched=build_model_a_features(raw,2023,include_targets=True,feature_set='enriched')
        self.assertEqual(tuple(enriched.X),ENRICHED_FEATURES)
        pd.testing.assert_frame_equal(core.X,enriched.X.loc[:,list(CORE_FEATURES)])
        pd.testing.assert_frame_equal(core.audit,enriched.audit)
        pd.testing.assert_frame_equal(core.labels,enriched.labels)
        self.assertEqual(enriched.X_growth.shape[1],16)
        with self.assertRaises(ValueError):build_model_a_features(raw,2023,feature_set='inconnu')

    @staticmethod
    def history_fixture():
        rows=[]
        for i in range(6):
            for year in [2020,2021,2022]:
                row=fixture().iloc[0].to_dict()
                row.update(sUnitCode=f'{i:03}',sLeaseFrom=f'{year}-01-01',
                    sSignDate=f'{year-1}-12-15',sLeaseTo=f'{year}-12-31',
                    sRent=1000+year-2020,sRentEffective=900*(1.1**(year-2020)))
                if year==2022: row['sLeaseTo']='2023-12-31'
                rows.append(row)
            # Transition mûre récente indépendante : début 2022, fin 2022,
            # pour démontrer un support récent non nul sans utiliser le bail
            # courant dont la fin est future. Ancien bail et nouveau terminés.
            row=rows[-1].copy();row.update(sPropCode='REF',sLeaseFrom='2022-01-01',sLeaseTo='2022-12-31')
            old=row.copy();old.update(sLeaseFrom='2021-01-01',sSignDate='2020-12-15',sLeaseTo='2021-12-31',sRentEffective=900)
            rows.extend([old,row])
        return pd.DataFrame(rows)

    def test_each_enriched_feature_invariant_to_future_injection(self):
        from src.features.model_a_features import EXPERIMENTAL_FEATURES
        raw=self.history_fixture()
        a=build_model_a_features(raw,2023,feature_set='enriched')
        self.assertEqual(len(a.X),6)
        for name in EXPERIMENTAL_FEATURES:self.assertTrue(a.X[name].notna().all(),name)
        future=raw.iloc[:1].copy()
        future['sLeaseFrom']='2024-01-01';future['sSignDate']='2023-12-15'
        future['sLeaseTo']='2024-12-31';future['sRentEffective']=999999
        future['sBuilding']='FUTUR'
        b=build_model_a_features(pd.concat([raw,future],ignore_index=True),2023,feature_set='enriched')
        for name in EXPERIMENTAL_FEATURES:
            with self.subTest(feature=name):pd.testing.assert_series_equal(a.X[name],b.X[name])
        pd.testing.assert_frame_equal(a.X,build_model_a_features(raw.iloc[::-1],2023,feature_set='enriched').X)

    def test_current_effective_and_unmatured_pairs_do_not_feed_history(self):
        raw=self.history_fixture()
        a=build_model_a_features(raw,2023,feature_set='enriched')
        # Le bail commencé avant D mais se terminant après D est non mûr.
        mask=raw.sPropCode.eq('P1') & raw.sLeaseFrom.eq('2022-01-01')
        raw.loc[mask,'sRentEffective']=999999
        b=build_model_a_features(raw,2023,feature_set='enriched')
        pd.testing.assert_frame_equal(a.X,b.X)
        self.assertAlmostEqual(a.X.previous_same_unit_growth.iloc[0],1.1**(365.25/366)-1)

    def test_enriched_missing_history_and_small_recent_support(self):
        from src.features.model_a_features import EXPERIMENTAL_FEATURES
        raw=fixture().loc[lambda x:x.sLeaseFrom.lt('2023')]
        a=build_model_a_features(raw,2023,feature_set='enriched')
        self.assertEqual(len(a.X),2)
        self.assertTrue(a.X.loc[:,list(EXPERIMENTAL_FEATURES)].isna().all().all())
        raw=self.history_fixture()
        raw=raw.loc[raw.sUnitCode.eq('000')]
        a=build_model_a_features(raw,2023,feature_set='enriched')
        self.assertTrue(a.X.previous_same_unit_growth.notna().all())
        self.assertTrue(a.X.recent_building_growth.isna().all())
        self.assertTrue(a.X.recent_portfolio_growth.isna().all())

    def test_not_identifiable_economic_features_remain_rejected(self):
        from src.features.model_a_features import EXPERIMENT_SAFETY
        raw=self.history_fixture();raw['sRentEffective']=np.nan
        a=build_model_a_features(raw,2023,feature_set='enriched')
        for name in ['prior_effective_rent','concession_gap','rent_position']:
            self.assertEqual(EXPERIMENT_SAFETY[name],'NOT_IDENTIFIABLE')
            self.assertNotIn(name,a.X)
            with self.assertRaises(ValueError):select_model_a_features(a,[name])
        # Aucun segment/fallback rent_position n'est autorisé à contourner
        # l'absence de provenance temporelle du prix effectif courant.
        self.assertTrue(a.X.previous_same_unit_growth.isna().all())


if __name__=='__main__':unittest.main()
