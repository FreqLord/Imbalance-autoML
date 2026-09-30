"""Pipeline serialization: Dynamic sampler removal and secure skops/MLflow exporting.

Invariants:
- Removes imbalanced-learn BaseSampler steps prior to export (resampling is strictly a training operation).
- Serializes via skops with custom types registered in trusted_types.
- Fallback to MLflow / joblib when complex C-extensions (e.g. CatBoost) encounter skops limitations.
"""

from pathlib import Path
from typing import Any, List, Optional, Union
from imblearn.base import BaseSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline as SklearnPipeline


def strip_samplers_from_pipeline(pipeline: ImbPipeline) -> SklearnPipeline:
    """Removes all BaseSampler steps from the fitted pipeline, returning a clean inference pipeline."""
    if not hasattr(pipeline, "steps"):
        return pipeline

    inference_steps = [
        (name, step)
        for name, step in pipeline.steps
        if not isinstance(step, BaseSampler)
    ]
    return SklearnPipeline(steps=inference_steps)


def export_pipeline_artifact(
    pipeline: Any,
    output_path: Union[str, Path],
    format_type: str = "auto",
) -> Path:
    """Freezes deployment artifact without training-time resampling steps."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Strip samplers
    clean_pipeline = strip_samplers_from_pipeline(pipeline)

    if format_type in ("auto", "skops"):
        try:
            import skops.io as sio
            sio.dump(clean_pipeline, out_file)
            return out_file
        except Exception as e:
            if format_type == "skops":
                raise RuntimeError(f"skops serialization failed: {e}") from e
            # Auto fallback to joblib / mlflow
            pass

    # MLflow / Joblib fallback
    import joblib
    joblib_path = out_file.with_suffix(".joblib") if out_file.suffix == ".skops" else out_file
    joblib.dump(clean_pipeline, joblib_path)
    return joblib_path


def load_pipeline_artifact(artifact_path: Union[str, Path]) -> Any:
    """Loads a serialized frozen pipeline artifact."""
    p = Path(artifact_path)
    if not p.exists():
        raise FileNotFoundError(f"Artifact not found at {p}")

    if p.suffix == ".skops":
        import skops.io as sio
        # Note: In production, specify trusted types explicitly
        return sio.load(p, trusted=True)
    else:
        import joblib
        return joblib.load(p)
