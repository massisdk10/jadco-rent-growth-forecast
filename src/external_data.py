"""Contrôles des signaux publics et filtrage à date, sans jointure CRM."""
from urllib.parse import urlparse
import numpy as np
import pandas as pd

COLUMNS = ('year','geography','province','signal','value','unit','reference_period',
           'publication_date','available_date','source_name','source_url','notes',
           'retrieved_date','availability_verified','availability_notes')
KEY = ('geography','province','signal','reference_period')
OFFICIAL = {'SCHL':'cmhc-schl.gc.ca','Statistique Canada':'statcan.gc.ca',
            'TAL':'tal.gouv.qc.ca','Gouvernement de l’Ontario':'ontario.ca'}


def validate_external_data(data):
    """Retourner les problèmes sans supprimer ni corriger aucune observation.

    Les valeurs/dates absentes sont permises si la raison figure dans notes.
    Une date présente mais invalide est une erreur. Les plages sont des
    contrôles de plausibilité exploratoires, pas des filtres statistiques.
    """
    issues=[]
    if set(data.columns)!=set(COLUMNS):
        return ['Schéma canonique non conforme ou colonnes supplémentaires.']
    for col in ['geography','province','signal','unit','reference_period','source_name','source_url','notes','availability_notes']:
        if data[col].fillna('').astype(str).str.strip().eq('').any():issues.append(f'Champ obligatoire vide : {col}.')
    if data.duplicated(list(KEY)).any():issues.append('Clé logique dupliquée.')
    years=pd.to_numeric(data.year,errors='coerce')
    if (years.isna() | years.mod(1).ne(0) | ~years.between(1900,2100)).any():issues.append('Année invalide.')
    for name,url in zip(data.source_name,data.source_url):
        host=urlparse(str(url)).hostname or ''
        domain=OFFICIAL.get(name)
        if not domain or urlparse(str(url)).scheme!='https' or not (host==domain or host.endswith('.'+domain)):
            issues.append('Source ou URL non officielle.');break
    for col in ['publication_date','available_date','retrieved_date']:
        supplied=data[col].notna() & data[col].astype(str).str.strip().ne('')
        dates=pd.to_datetime(data[col],errors='coerce',format='mixed')
        if (supplied & dates.isna()).any():issues.append(f'Date invalide : {col}.')
    pub=pd.to_datetime(data.publication_date,errors='coerce',format='mixed')
    avail=pd.to_datetime(data.available_date,errors='coerce',format='mixed')
    if (avail<pub).any():issues.append('Disponibilité antérieure à la publication.')
    verified=data.availability_verified.map(lambda v: isinstance(v, (bool, np.bool_)) and bool(v))
    if not data.availability_verified.map(lambda v: isinstance(v, (bool, np.bool_))).all():
        issues.append('availability_verified doit être un booléen strict.')
    if (verified & (avail.isna() | pub.isna())).any():
        issues.append('Disponibilité vérifiée sans dates de publication/disponibilité.')
    if (~verified & avail.notna()).any():
        issues.append('Disponibilité renseignée sans vérification.')
    retrieved=pd.to_datetime(data.retrieved_date,errors='coerce',format='mixed')
    if retrieved.isna().any() or (avail>retrieved).any():
        issues.append('Date de récupération absente ou antérieure à la disponibilité.')
    values=pd.to_numeric(data.value,errors='coerce')
    if (data.value.notna() & values.isna()).any() or np.isinf(values).any():issues.append('Valeur numérique invalide.')
    pct=data.unit.eq('percent')
    if (pct & ~values.between(-100,100) & values.notna()).any():issues.append('Pourcentage hors plage exploratoire [-100,100].')
    bounded=data.signal.isin(['vacancy_rate','turnover_rate','ontario_guideline'])
    if (bounded & values.notna() & ~values.between(0,100)).any():issues.append('Taux non négatif hors plage [0,100].')
    if (data.unit.isin(['index_2002_100','CAD_per_month']) & values.le(0)).any():issues.append('Niveau non positif.')
    return issues


def get_available_external_data(data, as_of_date):
    """Copie des seules valeurs à disponibilité historique vérifiée, à la coupure inclusive.

    Une disponibilité absente exclut la ligne ; une date mal formée échoue.
    Les lignes à valeur manquante restent dans le CSV de traçabilité, mais ne
    sont pas retournées comme signaux utilisables. Aucune année n'est imputée ; aucun repli vers retrieved_date.
    """
    cutoff=pd.to_datetime(as_of_date,errors='raise')
    if pd.isna(cutoff):raise ValueError('Date de coupure manquante.')
    dates=pd.to_datetime(data.available_date,errors='raise',format='mixed')
    verified=data.availability_verified.map(lambda v: isinstance(v, (bool, np.bool_)) and bool(v))
    return data.loc[verified & dates.notna() & dates.le(cutoff) & data.value.notna()].copy(deep=True)
