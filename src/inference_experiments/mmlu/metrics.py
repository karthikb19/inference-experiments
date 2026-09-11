"""MMLU weighted, subject, and category metrics."""

from __future__ import annotations

from collections import defaultdict

from inference_experiments.mmlu.models import (
    AccuracyMetric,
    CategoryMetric,
    DataValidationError,
    EvaluationMetrics,
    Prediction,
)

CATEGORY_SUBJECTS: dict[str, frozenset[str]] = {
    "stem": frozenset(
        {
            "abstract_algebra",
            "anatomy",
            "astronomy",
            "college_biology",
            "college_chemistry",
            "college_computer_science",
            "college_mathematics",
            "college_physics",
            "computer_security",
            "conceptual_physics",
            "electrical_engineering",
            "elementary_mathematics",
            "high_school_biology",
            "high_school_chemistry",
            "high_school_computer_science",
            "high_school_mathematics",
            "high_school_physics",
            "high_school_statistics",
            "machine_learning",
        }
    ),
    "humanities": frozenset(
        {
            "formal_logic",
            "high_school_european_history",
            "high_school_us_history",
            "high_school_world_history",
            "international_law",
            "jurisprudence",
            "logical_fallacies",
            "moral_disputes",
            "moral_scenarios",
            "philosophy",
            "prehistory",
            "professional_law",
            "world_religions",
        }
    ),
    "social_sciences": frozenset(
        {
            "econometrics",
            "high_school_geography",
            "high_school_government_and_politics",
            "high_school_macroeconomics",
            "high_school_microeconomics",
            "high_school_psychology",
            "human_sexuality",
            "professional_psychology",
            "public_relations",
            "security_studies",
            "sociology",
            "us_foreign_policy",
        }
    ),
    "other": frozenset(
        {
            "business_ethics",
            "clinical_knowledge",
            "college_medicine",
            "global_facts",
            "human_aging",
            "management",
            "marketing",
            "medical_genetics",
            "miscellaneous",
            "nutrition",
            "professional_accounting",
            "professional_medicine",
            "virology",
        }
    ),
}
SUBJECT_CATEGORY = {
    subject: category
    for category, subjects in CATEGORY_SUBJECTS.items()
    for subject in subjects
}


def aggregate_metrics(predictions: tuple[Prediction, ...]) -> EvaluationMetrics:
    """Aggregate the original size-weighted score and diagnostic slices."""
    if not predictions:
        raise ValueError("cannot aggregate an empty prediction set")
    unknown = sorted({row.subject for row in predictions} - SUBJECT_CATEGORY.keys())
    if unknown:
        raise DataValidationError(f"subjects have no MMLU category: {unknown}")

    by_subject: dict[str, list[Prediction]] = defaultdict(list)
    for prediction in predictions:
        by_subject[prediction.subject].append(prediction)
    subject_metrics = tuple(
        _accuracy(subject, tuple(rows)) for subject, rows in sorted(by_subject.items())
    )
    categories: list[CategoryMetric] = []
    for category, category_subjects in CATEGORY_SUBJECTS.items():
        rows = tuple(
            prediction
            for prediction in predictions
            if prediction.subject in category_subjects
        )
        if not rows:
            continue
        included_subjects = tuple(
            metric for metric in subject_metrics if metric.name in category_subjects
        )
        correct = sum(row.correct for row in rows)
        categories.append(
            CategoryMetric(
                name=category,
                correct=correct,
                total=len(rows),
                weighted_accuracy=correct / len(rows),
                subject_macro_accuracy=sum(
                    metric.accuracy for metric in included_subjects
                )
                / len(included_subjects),
            )
        )
    correct = sum(row.correct for row in predictions)
    return EvaluationMetrics(
        correct=correct,
        total=len(predictions),
        weighted_accuracy=correct / len(predictions),
        subject_macro_accuracy=sum(metric.accuracy for metric in subject_metrics)
        / len(subject_metrics),
        invalid_predictions=0,
        subjects=subject_metrics,
        categories=tuple(categories),
    )


def _accuracy(name: str, rows: tuple[Prediction, ...]) -> AccuracyMetric:
    correct = sum(row.correct for row in rows)
    return AccuracyMetric(
        name=name,
        correct=correct,
        total=len(rows),
        accuracy=correct / len(rows),
    )
