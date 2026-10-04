"""Explicabilité française et export JSON limité aux résultats agrégés."""
from copy import deepcopy
import json
from pathlib import Path

FIELDS=('method_name','forecast_year','prediction_origin','target_definition','target_name','predicted_growth',
        'forecast','cohort','architecture','features_used','components','backtest_by_year',
        'aggregate_metrics','runner_up_sensitivity','uncertainty')
FORBIDDEN=('sPropCode','sUnitCode','UNIT_KEY','unit_key','sLeaseFrom','sRent','sRentEffective',
           'records','rows','contributions','probabilities','conditional_growth')


def validate_aggregate_payload(value):
    """Refuser identifiants/records, objets pandas/NumPy et nombres non finis."""
    def walk(node):
        if isinstance(node,dict):
            if any(k in FORBIDDEN for k in node):raise ValueError('Champ individuel interdit dans le rapport.')
            for v in node.values():walk(v)
        elif isinstance(node,list):
            for v in node:walk(v)
        elif node is not None and not isinstance(node,(str,int,float,bool)):
            raise ValueError('Objet individuel ou non JSON interdit.')
    walk(value)
    json.dumps(value,ensure_ascii=False,allow_nan=False)


def build_explanation(payload):
    """Construire seulement les champs agrégés déclarés, sans LLM."""
    if set(payload)!=set(FIELDS):raise ValueError('Schéma de packaging final invalide.')
    result=deepcopy(payload)
    result['selection_reason']='MAE pratiquement équivalent à la moyenne (écart <0,05 pp), même simplicité, pire erreur annuelle et RMSE plus faibles ; choix figé avant 2026.'
    result['drivers']=[
        {'name':'Probabilité historique qualifiée','value':payload['components']['historical_transition_probability'],'unit':'fraction','reason':'Taux global des labels occurrence disponibles à la coupure.'},
        {'name':'Croissance conditionnelle médiane','value':payload['components']['historical_conditional_growth'],'unit':'fraction','reason':'Médiane des croissances annualisées admissibles, agrégées par unité/année historique.'},
        {'name':'Éligibilité de cohorte','value':payload['cohort']['n_units'],'unit':'unités','reason':'Dernier bail connu avec échéance contractuelle en 2026 ; cohorte figée.'},
        {'name':'Comportement de backtest','value':payload['aggregate_metrics']['worst_year_pp'],'unit':'points de pourcentage','reason':'Erreur historique maximale déterminant le rayon des scénarios.'}]
    result['risk_factors']=['Dérive du taux de transition observée en 2025','Seulement trois années de backtest et sélection non imbriquée',
        'Fidélité historique CRM non entièrement prouvée','Maturité des labels et censure possibles']
    result['limitations']=['P1 ne mesure pas la croissance complète du rent roll, occupancy, vacancy, NOI ou tous les loyers.',
        'Les scénarios ne sont pas des intervalles de confiance ; aucune couverture garantie.',
        'Le champion au niveau portefeuille ne crée pas de différences prédictives entre unités.',
        "La médiane conditionnelle est un estimateur robuste, pas une espérance conditionnelle exacte.",
        'Asking et signaux externes non utilisés dans le champion faute de qualification prédictive.']
    validate_aggregate_payload(result)
    return result


def export_model_a_reports(report, output_dir):
    """Écrire trois JSON agrégés déclarés ; jamais de données individuelles."""
    allowed=set(FIELDS)|{'selection_reason','drivers','risk_factors','limitations'}
    if set(report)!=allowed:raise ValueError('Champs de rapport inconnus ; export refusé.')
    validate_aggregate_payload(report)
    directory=Path(output_dir);directory.mkdir(parents=True,exist_ok=True)
    summary={k:deepcopy(report[k]) for k in ['method_name','forecast_year','prediction_origin','target_definition','target_name',
        'predicted_growth','forecast','cohort','architecture','components','runner_up_sensitivity','uncertainty']}
    backtest={k:deepcopy(report[k]) for k in ['method_name','target_definition','architecture','backtest_by_year','aggregate_metrics']}
    outputs={'model_a_2026_summary.json':summary,'model_a_backtest_summary.json':backtest,'model_a_explanation.json':report}
    for name,payload in outputs.items():
        (directory/name).write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return [str(directory/name) for name in outputs]
