"""Artifact exporter: skops/MLflow serialization with dynamic sampler removal."""

from pathlib import Path
from typing import Any, Union
from imblearn.base import BaseSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline as SklearnPipeline


def strip_samplers_from_pipeline(pipeline: ImbPipeline) -> SklearnPipeline:
    """Drops all imblearn BaseSampler steps from pipeline prior to production artifact freezing."""
    clean_steps = [
        (name, step)
        for name, step in pipeline.steps
        if not isinstance(step, BaseSampler)
    ]
    return SklearnPipeline(steps=clean_steps)


def export_artifact(
    pipeline: Any,
    output_path: Union[str, Path],
    format_type: str = "skops",
):
    """Freezes deployment artifact using skops or MLflow fallback."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(pipeline, "steps"):
        pipeline = strip_samplers_from_pipeline(pipeline)

    if format_type == "skops":
        import skops.io as sio
        sio.dump(pipeline, output_path)
    elif format_type == "mlflow":
        import mlflow.sklearn
        mlflow.sklearn.save_model(pipeline, str(output_path))
    else:
        import joblib
        joblib.dump(pipeline, output_path)
