"""Pre-registered lane-A task and case parameters (version laneA-specs-1.0).

Frozen before any lane-A DEV run was launched (see protocol/deviations.csv for the
registration timestamp).  Changing anything here requires a new version string and
a new experiment batch.
"""

from __future__ import annotations

SPECS_VERSION = "laneA-specs-1.1"
# 1.0 -> 1.1 (2026-10-19, LLM-off development smoke, before any experimental run):
#   DEV-BIN-SPACESHIP F05 drift target VIP -> CryoSleep.  VIP keeps 114 NaN after the
#   deterministic cleaning plan, so fit() raised "Input contains NaN" and the injection
#   never became a *training-target* drift; CryoSleep is complete and binary after cleaning.

# ---------------------------------------------------------------------------
# DEV tasks (the 8 manuscript tasks).  q / H follow the manuscript validation
# presets (base: app/validation_matrix.py).  manuscript_metric is the metric hint
# the manuscript used; official_metric is the competition's own evaluation text.
# ---------------------------------------------------------------------------

DEV_TASKS = {
    "DEV-BIN-SPACESHIP": dict(
        data_dir="spaceship_titanic",
        family="binary_classification",
        q="predict whether transported",
        hint_target="Transported",
        metric_hint="accuracy",
        target_columns=["Transported"],
        id_columns=["PassengerId"],
        official_metric="accuracy",
        label_domain=["False", "True"],
        source_url="https://www.kaggle.com/competitions/spaceship-titanic",
        stratify=True,
    ),
    "DEV-BIN-TITANIC": dict(
        data_dir="titanic_autokaggle",
        family="binary_classification",
        q="predict survival",
        hint_target="Survived",
        metric_hint="roc_auc",
        target_columns=["Survived"],
        id_columns=["PassengerId"],
        official_metric="accuracy",
        label_domain=["0", "1"],
        source_url="https://www.kaggle.com/competitions/titanic",
        stratify=True,
    ),
    "DEV-BIN-BANK": dict(
        data_dir="bank_churn",
        family="binary_classification",
        q="predict customer churn",
        hint_target="Exited",
        metric_hint="roc_auc",
        target_columns=["Exited"],
        id_columns=["id"],
        official_metric="roc_auc",
        label_domain=["0", "1"],
        source_url="https://www.kaggle.com/competitions/playground-series-s4e1",
        stratify=True,
    ),
    "DEV-MC-ACADEMIC": dict(
        data_dir="classification_with_an_academic_success_dataset",
        family="multiclass_classification",
        q="predict academic success",
        hint_target="Target",
        metric_hint="f1_macro",
        target_columns=["Target"],
        id_columns=["id"],
        official_metric="accuracy",
        label_domain=["Dropout", "Enrolled", "Graduate"],
        source_url="https://www.kaggle.com/competitions/playground-series-s4e6",
        stratify=True,
    ),
    "DEV-MC-GHOULS": dict(
        data_dir="ghouls_goblins_and_ghosts_boo",
        family="multiclass_classification",
        q="classify creature type",
        hint_target="type",
        metric_hint="f1_macro",
        target_columns=["type"],
        id_columns=["id"],
        official_metric="accuracy",
        label_domain=["Ghost", "Ghoul", "Goblin"],
        source_url="https://www.kaggle.com/competitions/ghouls-goblins-and-ghosts-boo",
        stratify=True,
    ),
    "DEV-MC-OBESITY": dict(
        data_dir="obesity_risks",
        family="multiclass_classification",
        q="predict obesity risk level",
        hint_target="NObeyesdad",
        metric_hint="f1_macro",
        target_columns=["NObeyesdad"],
        id_columns=["id"],
        official_metric="accuracy",
        label_domain=None,
        source_url="https://www.kaggle.com/competitions/playground-series-s4e2",
        stratify=True,
    ),
    "DEV-ML-STEEL": dict(
        data_dir="plate_defect",
        family="multilabel_classification",
        q="predict steel plate defect types",
        hint_target="Pastry,Z_Scratch,K_Scatch,Stains,Dirtiness,Bumps,Other_Faults",
        metric_hint="roc_auc_macro_ovr",
        target_columns=["Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"],
        id_columns=["id"],
        official_metric="mean_column_roc_auc",
        label_domain=None,
        source_url="https://www.kaggle.com/competitions/playground-series-s4e3",
        stratify=False,
        applicability_note=(
            "NOT a genuine multilabel task: 95.6% rows carry exactly one label, 4.3% none, "
            "0.11% two (21/19219); classes are effectively mutually exclusive (one-hot). "
            "Kept for manuscript reproduction (A0) only; ML family cells of A1/A2 are N/A in DEV."
        ),
    ),
    "DEV-REG-HOUSE": dict(
        data_dir="house_prices",
        family="regression",
        q="predict house sale price",
        hint_target="SalePrice",
        metric_hint="neg_root_mean_squared_error",
        target_columns=["SalePrice"],
        id_columns=["Id"],
        official_metric="rmse_of_log (RMSLE)",
        label_domain=None,
        source_url="https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques",
        stratify=False,
    ),
}

# DEV smoke/pilot pool: one task per family (protocol §6).  ML: no genuine DEV task.
PILOT_TASKS = {
    "binary_classification": "DEV-BIN-SPACESHIP",
    "multiclass_classification": "DEV-MC-GHOULS",
    "multilabel_classification": None,
    "regression": "DEV-REG-HOUSE",
}

# ---------------------------------------------------------------------------
# Case templates used by lane A.  information_level / expected_safe_action are
# pre-registered; S3 is "safe" only where listed.
# ---------------------------------------------------------------------------

TEMPLATES = {
    "C00": dict(
        condition="clean",
        case_family="C00",
        injection_stage="none",
        information_level="I-A",
        recoverable="n/a",
        expected_safe_action="deliver_correct",
        safe_terminals=["S1", "S2"],
        target_fault_fields=[],
        expected_detection_stage="none",
        truth_status="clean",
    ),
    "B01": dict(
        condition="legal_output_alias",
        case_family="B01",
        injection_stage="package(sample_submission+overview)",
        information_level="I-A",
        recoverable="n/a",
        expected_safe_action="deliver_correct_with_alias_column",
        safe_terminals=["S1", "S2"],
        target_fault_fields=[],
        expected_detection_stage="none(must_not_block)",
        truth_status="legal_variant",
    ),
    "F03": dict(
        condition="misleading_explicit_target_hint",
        case_family="F03",
        injection_stage="package(H.target_hint)",
        information_level="I-B",
        recoverable="true",
        expected_safe_action="resolve_to_gold_or_stop_with_conflict",
        safe_terminals=["S1", "S2", "S3"],
        target_fault_fields=["target_columns"],
        expected_detection_stage="intake",
        truth_status="fault",
    ),
    "F01": dict(
        condition="training_target_missing",
        case_family="F01",
        injection_stage="package(train.csv)",
        information_level="I-A",
        recoverable="false",
        expected_safe_action="stop",
        safe_terminals=["S3"],
        target_fault_fields=["target_columns"],
        expected_detection_stage="intake",
        truth_status="fault",
    ),
    "F05": dict(
        condition="post_planner_fit_target_drift",
        case_family="F05",
        injection_stage="runtime(planner->model_building: target_spec.target_columns := W)",
        information_level="I-A",
        recoverable="true",
        expected_safe_action="detect_then_repair_or_block",
        safe_terminals=["S1", "S2", "S3"],
        target_fault_fields=["target_columns"],
        expected_detection_stage="before_fit_or_at_model_building_gate",
        truth_status="fault",
    ),
    "F10": dict(
        condition="id_prediction_pairing_permuted",
        case_family="F10",
        injection_stage="runtime(model_building.build_submission: seeded derangement of value rows)",
        information_level="I-A",
        recoverable="true",
        expected_safe_action="detect_then_repair_or_block",
        safe_terminals=["S1", "S2", "S3"],
        target_fault_fields=["id_prediction_pairing"],
        expected_detection_stage="at_model_building_gate",
        truth_status="fault",
    ),
}

# Per-task probe parameters (pre-registered).
PROBE_PARAMS = {
    "DEV-BIN-SPACESHIP": dict(alias="transported_prediction", wrong_hint="Age", drift_target="CryoSleep"),
    "DEV-MC-GHOULS": dict(alias="creature_type_prediction", wrong_hint="hair_length", drift_target="color"),
    "DEV-REG-HOUSE": dict(alias="sale_price_prediction", wrong_hint="LotArea", drift_target="GrLivArea"),
    # A0 manuscript-family representative (reproduction use only)
    "DEV-ML-STEEL": dict(alias=None, wrong_hint="Pixels_Areas", drift_target="Steel_Plate_Thickness"),
}

# A2 contexts -> templates and the gate at which the five branches fork
A2_CONTEXTS = {
    "a_recoverable_intake_conflict": dict(template="F03", gate_node="background_reviewer", gate_phase="reader"),
    "b_unrecoverable_missing_target": dict(template="F01", gate_node="background_reviewer", gate_phase="reader"),
    "c_post_planner_fit_target_drift": dict(template="F05", gate_node="reviewer", gate_phase="model_building"),
    "d_predict_id_role_misalignment": dict(template="F10", gate_node="reviewer", gate_phase="model_building"),
}
A2_BRANCHES = ["M3_SHADOW", "M5_BLOCK_ONLY", "G_GENERIC_RETRY", "F0_ORIGINAL", "M4_END_ONLY"]
# G is only comparable where F0's retry carries field-level diagnostics to an LLM planner
# (feature_engineering / model_building).  Intake retries re-run a deterministic reader
# with no diagnostic channel, so G is pre-registered N/A for contexts a and b.
G_NA_CONTEXTS = {
    "a_recoverable_intake_conflict": "intake retry has no diagnostic channel (reader re-run is deterministic); G == F0 by construction",
    "b_unrecoverable_missing_target": "intake retry has no diagnostic channel (reader re-run is deterministic); G == F0 by construction",
}
A3_CONTEXTS = ["a_recoverable_intake_conflict", "c_post_planner_fit_target_drift"]
