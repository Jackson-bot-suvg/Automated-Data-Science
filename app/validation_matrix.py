"""Dataset validation matrix for representative acceptance and smoke coverage."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetValidationSpec:
    key: str
    task: str
    dataset: str
    target: str
    metric_hint: str
    description: str
    task_family: str
    target_mode: str
    primary_acceptance: bool


PRIMARY_ACCEPTANCE_DATASETS: tuple[DatasetValidationSpec, ...] = (
    DatasetValidationSpec(
        key="spaceship_titanic",
        task="predict whether transported",
        dataset="data/spaceship_titanic",
        target="Transported",
        metric_hint="accuracy",
        description="single-target binary classification",
        task_family="binary_classification",
        target_mode="single_target",
        primary_acceptance=True,
    ),
    DatasetValidationSpec(
        key="academic_success",
        task="predict academic success",
        dataset="data/classification_with_an_academic_success_dataset",
        target="Target",
        metric_hint="f1_macro",
        description="single-target multiclass classification",
        task_family="multiclass_classification",
        target_mode="single_target",
        primary_acceptance=True,
    ),
    DatasetValidationSpec(
        key="house_prices",
        task="predict house sale price",
        dataset="data/house_prices",
        target="SalePrice",
        metric_hint="neg_root_mean_squared_error",
        description="single-target regression",
        task_family="regression",
        target_mode="single_target",
        primary_acceptance=True,
    ),
    DatasetValidationSpec(
        key="plate_defect",
        task="predict steel plate defect types",
        dataset="data/plate_defect",
        target="Pastry,Z_Scratch,K_Scatch,Stains,Dirtiness,Bumps,Other_Faults",
        metric_hint="roc_auc_macro_ovr",
        description="multi-target multilabel classification",
        task_family="multilabel_classification",
        target_mode="multi_label",
        primary_acceptance=True,
    ),
)

SECONDARY_SMOKE_DATASETS: tuple[DatasetValidationSpec, ...] = (
    DatasetValidationSpec(
        key="titanic_autokaggle",
        task="predict survival",
        dataset="data/titanic_autokaggle",
        target="Survived",
        metric_hint="roc_auc",
        description="single-target binary classification smoke dataset",
        task_family="binary_classification",
        target_mode="single_target",
        primary_acceptance=False,
    ),
    DatasetValidationSpec(
        key="bank_churn",
        task="predict customer churn",
        dataset="data/bank_churn",
        target="Exited",
        metric_hint="roc_auc",
        description="single-target binary classification smoke dataset",
        task_family="binary_classification",
        target_mode="single_target",
        primary_acceptance=False,
    ),
    DatasetValidationSpec(
        key="ghouls",
        task="classify creature type",
        dataset="data/ghouls_goblins_and_ghosts_boo",
        target="type",
        metric_hint="f1_macro",
        description="single-target multiclass classification smoke dataset",
        task_family="multiclass_classification",
        target_mode="single_target",
        primary_acceptance=False,
    ),
    DatasetValidationSpec(
        key="obesity_risks",
        task="predict obesity risk level",
        dataset="data/obesity_risks",
        target="NObeyesdad",
        metric_hint="f1_macro",
        description="single-target multiclass classification smoke dataset",
        task_family="multiclass_classification",
        target_mode="single_target",
        primary_acceptance=False,
    ),
)

ALL_VALIDATION_DATASETS: tuple[DatasetValidationSpec, ...] = (
    *PRIMARY_ACCEPTANCE_DATASETS,
    *SECONDARY_SMOKE_DATASETS,
)
