"""Tests ciblés de provenance, de schéma et de disponibilité temporelle."""
from pathlib import Path
import unittest
import subprocess
import pandas as pd
from src.external_data import COLUMNS, get_available_external_data, validate_external_data


class ExternalDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pd.read_csv(Path(__file__).resolve().parents[1] / 'data/external/external_signals.csv')

    def test_public_dataset_valid(self):
        self.assertEqual(tuple(self.data.columns), COLUMNS)
        self.assertEqual(validate_external_data(self.data), [])
        self.assertEqual(len(self.data), 113)
        self.assertEqual(self.data.value.isna().sum(), 22)

    def test_reject_invalid_provenance_and_extra_columns(self):
        bad = self.data.iloc[:1].copy()
        bad.loc[:, 'source_url'] = 'https://ontario.ca.example.org/page'
        self.assertIn('Source ou URL non officielle.', validate_external_data(bad))
        bad['sUnitCode'] = 'colonne interdite'
        self.assertTrue(validate_external_data(bad))

    def test_reject_duplicate_and_invalid_date(self):
        bad = pd.concat([self.data.iloc[:1], self.data.iloc[:1]], ignore_index=True)
        self.assertIn('Clé logique dupliquée.', validate_external_data(bad))
        bad = self.data.iloc[:1].copy()
        bad.loc[:, 'available_date'] = 'date invalide'
        self.assertIn('Date invalide : available_date.', validate_external_data(bad))
        with self.assertRaises(ValueError):
            get_available_external_data(bad, '2026-10-03')

    def test_filter_inclusive_missing_future_and_copy(self):
        sample = self.data.iloc[:5].copy().reset_index(drop=True)
        sample['availability_verified'] = [True, True, True, True, False]
        sample['available_date'] = ['2025-12-31', '2026-01-01', None, '2025-12-30', '2025-01-01']
        sample.loc[3, 'value'] = float('nan')
        result = get_available_external_data(sample, '2025-12-31')
        self.assertEqual(result.index.tolist(), [0])
        result.loc[0, 'value'] = -999
        self.assertNotEqual(sample.loc[0, 'value'], -999)

    def test_only_qualified_values_enter_historical_cutoffs(self):
        self.assertEqual(len(get_available_external_data(self.data, '2025-12-31')), 9)
        result = get_available_external_data(self.data, '2026-10-03')
        self.assertEqual(len(result), 12)
        self.assertTrue(result.availability_verified.all())
        self.assertTrue(pd.to_datetime(result.available_date).le(pd.Timestamp('2026-10-03')).all())
        publication = pd.to_datetime(result.publication_date)
        self.assertTrue((publication.isna() | publication.le(pd.to_datetime(result.available_date))).all())

    def test_retrieval_is_not_availability(self):
        unverified = self.data.loc[~self.data.availability_verified]
        self.assertTrue(unverified.available_date.isna().all())
        self.assertTrue(unverified.retrieved_date.notna().all())
        self.assertTrue(get_available_external_data(unverified, '2099-12-31').empty)
        sample = self.data.iloc[:1].copy()
        sample['availability_verified'] = 'True'
        sample['available_date'] = '2025-01-01'
        self.assertTrue(get_available_external_data(sample, '2026-10-03').empty)

    def test_gitignore_public_exception_only(self):
        root = Path(__file__).resolve().parents[1]
        paths = ['data/external/external_signals.csv',
                 'data/raw/equinoxe_listings.csv',
                 'data/raw/equinoxe_lease_history.csv',
                 'data/raw/equinoxe_concessions.csv',
                 'data/raw/equinoxe_asking_history.csv',
                 'data/processed/test.csv', 'data/external/autre.csv',
                 'autre.csv']
        result = subprocess.run(
            ['git', 'check-ignore', '--no-index', '--stdin'],
            input='\n'.join(paths)+'\n', text=True, capture_output=True, cwd=root, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(set(result.stdout.splitlines()), set(paths[1:]))
        tracked = subprocess.run(['git', 'ls-files', 'data/raw', 'data/processed'],
                                 text=True, capture_output=True, cwd=root, check=True)
        self.assertFalse(any(p.endswith('.csv') for p in tracked.stdout.splitlines()))

    def test_invalid_percentage_reported(self):
        bad = self.data.iloc[:1].copy()
        bad.loc[:, 'value'] = 101
        self.assertTrue(validate_external_data(bad))


if __name__ == '__main__':
    unittest.main()
