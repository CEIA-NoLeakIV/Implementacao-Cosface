from __future__ import annotations

import os
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------
# ``DATA_ROOT`` can be overridden via environment variable to adapt to the
# local filesystem without touching the repository.  By default we expect the
# datasets to live under ``data/raw`` inside the project.
DATA_ROOT = os.environ.get("DATA_ROOT", os.path.join(os.getcwd(), "data"))
VGGFACE2_TRAIN_PATH = os.path.join(DATA_ROOT, "raw", "vggface2_112x112")
LFW_PATH = os.path.join(DATA_ROOT, "raw", "lfw")
LFW_PAIRS_PATH = os.path.join(LFW_PATH, "lfw_ann.txt")

def _count_classes(dataset_path: str) -> int:
    '''Count the number of folders (classes) present in ''dataset_path''.
    The helper silently returns ``0`` if the directory does not exist.  This
    keeps the configuration importable on machines where the dataset is not
    available.'''

    if not os.path.isdir(dataset_path):
        return 0

    return len([
        entry for entry in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, entry))
    ])

@dataclass
class FaceRecognitionConfig:
    """Hyperparameters used during training and evaluation."""

    # Dataset parameters ---------------------------------------------------
    num_classes: int = _count_classes(VGGFACE2_TRAIN_PATH)
    image_size: tuple[int, int] = (112, 112)
    input_shape: tuple[int, int, int] = (112, 112, 3)
    embedding_size: int = 512

    # Fraction of the dataset to use during experiments.  A smaller fraction
    # speeds up iterations while still exposing the optimisation pipeline to a
    # meaningful number of identities.  This value is overwritten by the
    # training pipeline once the subset is materialised.
    dataset_fraction: float = 0.25

    # Training hyperparameters ---------------------------------------------
    batch_size: int = 128
    warmup_epochs: int = 5
    epochs: int = 30

    optimizer: str = "SGD"
    learning_rate: float = 5e-3
    min_learning_rate: float = 1e-5
    momentum: float = 0.9

    cosine_margin: float = 0.35
    cosine_scale: float = 30.0

    # Output artefacts -----------------------------------------------------
    checkpoint_path: str = "models/face_recognition_resnet50_cosface.keras"
    log_path: str = "deliverables/face_recognition_log.csv"
    figures_path: str = "reports/figures"

    # Backbone fine-tuning behaviour --------------------------------------
    # Number of layers (starting from the end of ResNet50) that will remain
    # trainable once we unfreeze the backbone.  This keeps the earlier blocks
    # frozen which is crucial for stability when using CosFace.
    trainable_backbone_layers: int = 46

    def update_num_classes(self, value: int) -> None:
        """Update the number of classes at runtime.

        ``CosFace`` needs to know the exact amount of classes because it
        changes the shape of the label input.  The training pipeline calls this
        method after sampling the dataset subset.
        """

        self.num_classes = int(value)


__all__ = [
    "DATA_ROOT",
    "VGGFACE2_TRAIN_PATH",
    "LFW_PATH",
    "LFW_PAIRS_PATH",
    "FaceRecognitionConfig",
]

CHECKPOINT_PATH = 'models/face_recognition_resnet50_cosface.keras'
LOG_PATH = 'deliverables/face_recognition_log.csv'
