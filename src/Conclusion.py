import importlib
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
SAVED_MODELS_DIR = BASE_DIR / "saved_models"


def _extract_metrics_dict(value: Any) -> dict | None:
    """Return a normalized metrics dict if possible."""
    if isinstance(value, dict):
        if "accuracy" in value and "f1_score" in value:
            return value
    return None


def load_metrics(module_name: str) -> dict:
    """
    Load a model evaluation module and extract accuracy/f1_score.

    Supported formats:
      - a dict variable named 'results', 'metrics', 'evaluation', 'model_metrics',
        'report', or '<module>_evaluation'
      - module attributes named 'accuracy' and 'f1_score'
    """
    module = importlib.import_module(module_name)

    candidate_names = (
        "results",
        "metrics",
        "evaluation",
        "model_metrics",
        "report",
        f"{module_name.lower()}_evaluation",
        f"{module_name}_evaluation",
    )

    for attr_name in candidate_names:
        value = getattr(module, attr_name, None)
        metrics = _extract_metrics_dict(value)
        if metrics is not None:
            return metrics

    accuracy = getattr(module, "accuracy", None)
    f1_score = getattr(module, "f1_score", None)

    if accuracy is not None and f1_score is not None:
        return {"accuracy": accuracy, "f1_score": f1_score}

    # Try to detect a pandas Series/DataFrame-like export if present
    for attr_name in dir(module):
        if attr_name.startswith("_"):
            continue
        value = getattr(module, attr_name, None)
        metrics = _extract_metrics_dict(value)
        if metrics is not None:
            return metrics

    raise ValueError(
        f"Could not find evaluation metrics in module '{module_name}'. "
        f"Expected a dict containing 'accuracy' and 'f1_score', or module "
        f"attributes with those names."
    )


def file_size_kb(path: Path) -> float:
    """Return file size in KB."""
    return round(path.stat().st_size / 1024, 2)


# Load evaluation results
cnn_model_evaluation = load_metrics("CNN_model")
vgg_model_evaluation = load_metrics("VGG16")
resnet_model_evaluation = load_metrics("RESNET")
mobilenet_model_evaluation = load_metrics("MOBILE")
effnet_model_evaluation = load_metrics("EFFNET")

# Build comparison table
values = {
    "accuracy": [
        cnn_model_evaluation["accuracy"],
        vgg_model_evaluation["accuracy"],
        resnet_model_evaluation["accuracy"],
        mobilenet_model_evaluation["accuracy"],
        effnet_model_evaluation["accuracy"],
    ],
    "f1_score": [
        cnn_model_evaluation["f1_score"],
        vgg_model_evaluation["f1_score"],
        resnet_model_evaluation["f1_score"],
        mobilenet_model_evaluation["f1_score"],
        effnet_model_evaluation["f1_score"],
    ],
    "size_kb": [
        file_size_kb(SAVED_MODELS_DIR / "custom.keras"),
        file_size_kb(SAVED_MODELS_DIR / "vgg16.keras"),
        file_size_kb(SAVED_MODELS_DIR / "resnet.keras"),
        file_size_kb(SAVED_MODELS_DIR / "mobile.keras"),
        file_size_kb(SAVED_MODELS_DIR / "effnet.keras"),
    ],
}

df = pd.DataFrame(values, index=["cnn", "vgg", "resnet", "mobilenet", "efficientnet"])
df.to_pickle(BASE_DIR / "model_comparison.pkl")

print(df)