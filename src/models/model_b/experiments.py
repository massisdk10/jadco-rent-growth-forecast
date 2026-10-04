"""Configuration placeholders for the Model B experiment ladder."""

from dataclasses import dataclass
from enum import Enum

from src.features.model_dataset import ADMISSIBLE_FEATURES, validate_features


class ExperimentVariant(str, Enum):
    """Planned experiment families; this module does not run them."""

    B0_BASELINE = "B0"
    B1_HIERARCHICAL_INTERNAL = "B1"
    B2_ELIGIBLE_EXTERNAL = "B2"
    B3_EXTERNAL_SENSITIVITY = "B3"


# Signal identifiers present in the current public external-data schema.
# These remain separate context identifiers, never model feature names.
KNOWN_EXTERNAL_SIGNALS = frozenset(
    {
        "average_rent",
        "average_rent_growth",
        "non_turnover_rent_growth",
        "ontario_guideline",
        "rent_cpi_annual_growth",
        "rent_cpi_annual_mean",
        "rent_cpi_ytd_growth",
        "rent_cpi_ytd_mean",
        "tal_legacy_building_services",
        "tal_legacy_capital_expenditure",
        "tal_legacy_electricity",
        "tal_legacy_fuel_oil_other_energy",
        "tal_legacy_gas",
        "tal_legacy_maintenance",
        "tal_legacy_management",
        "tal_legacy_net_income",
        "tal_legacy_personal_services_rpa",
        "tal_new_base_rent",
        "tal_new_capital_expenditure",
        "tal_new_personal_services",
        "turnover_average_rent",
        "turnover_rent_growth",
        "vacancy_rate",
    }
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Declare a planned comparison without executing or joining any data.

    ``feature_names`` always refers to foundation-approved internal features.
    ``external_signals`` is an optional selection from the current public
    signal schema and is not an expansion of that feature allowlist.
    """

    variant: ExperimentVariant
    feature_names: tuple[str, ...] = ADMISSIBLE_FEATURES
    external_signals: tuple[str, ...] = ()
    scenario_name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.variant, ExperimentVariant):
            raise ValueError("variant must be an ExperimentVariant.")
        if len(self.external_signals) != len(set(self.external_signals)):
            raise ValueError("external_signals cannot contain duplicates.")
        unknown_signals = set(self.external_signals) - KNOWN_EXTERNAL_SIGNALS
        if unknown_signals:
            raise ValueError(f"Unknown external signals: {sorted(unknown_signals)}")

        if self.variant is ExperimentVariant.B0_BASELINE:
            if self.feature_names or self.external_signals or self.scenario_name:
                raise ValueError("B0 baseline cannot declare features or external context.")
            return

        try:
            validate_features(self.feature_names)
        except (TypeError, ValueError) as exc:
            raise ValueError("Experiment features must follow the foundation allowlist.") from exc
        if not self.feature_names:
            raise ValueError("Model experiments must declare at least one internal feature.")

        if self.variant is ExperimentVariant.B1_HIERARCHICAL_INTERNAL:
            if self.external_signals or self.scenario_name:
                raise ValueError("B1 is internal-only.")
        elif self.variant is ExperimentVariant.B2_ELIGIBLE_EXTERNAL:
            if not self.external_signals:
                raise ValueError("B2 must explicitly select eligible external signals.")
            if self.scenario_name:
                raise ValueError("B2 uses only cutoff-eligible external context, not scenarios.")
        elif self.variant is ExperimentVariant.B3_EXTERNAL_SENSITIVITY:
            if not self.external_signals:
                raise ValueError("B3 must explicitly select external signals for sensitivity.")
            if self.scenario_name is None or not self.scenario_name.strip():
                raise ValueError("B3 must declare a non-empty scenario_name.")


def run_experiment(config: ExperimentConfig) -> None:
    """Reserved entry point for later fold-by-fold Model B comparisons."""
    raise NotImplementedError(
        f"Experiment {config.variant.value} is configured only; execution is deferred."
    )