from __future__ import annotations
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict

FILE_PATH = Path(__file__).resolve()

# Tenta detectar automaticamente a raiz do projeto buscando a pasta "app"
for parent in FILE_PATH.parents:
    if parent.name == "app":  # ajuste se o nome do diretório raiz for outro
        PROJECT_ROOT = parent
        break
else:
    # fallback: assume o diretório do arquivo
    PROJECT_ROOT = FILE_PATH.parent

print(f"[DEBUG] FILE_PATH = {FILE_PATH}")
print(f"[DEBUG] PROJECT_ROOT = {PROJECT_ROOT}")

DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DATA_ROOT = os.environ.get("DATA_ROOT", str(DEFAULT_DATA_ROOT))

def _resolved_vggface2_path(data_root: str) -> str:
    candidates = [
        "vggface2_112x112",
        "vggface2_HQ",
        "vggface2",
    ]
    for candidate in candidates:
        path = os.path.join(data_root, "raw", candidate)
        if os.path.isdir(path):
            return path
    return os.path.join(data_root, "raw", candidates[0])

VGGFACE2_TRAIN_PATH = _resolved_vggface2_path(DATA_ROOT)

VGGFACE2_CROPPED_PATH = "/home/ubuntu/noleak/EmbeddingFramework/data/vggface2_HQ_cropado/insightface_aligned_112x112"

LFW_PATH = os.path.join(DATA_ROOT, "raw", "lfw")
LFW_PAIRS_PATH = os.path.join(LFW_PATH, "lfw_ann.txt")

def _count_classes(dataset_path: str) -> int:
    '''Count the number of folders (classes) present in ''dataset_path''.
    The helper silently returns ``0`` if the directory does not exist.  This
    keeps the configuration importable on machines where the dataset is not
    available.'''

    if not os.path.isdir(dataset_path):
        return 0
    
    train_subfolder = os.path.join(dataset_path, 'train')
    if os.path.isdir(train_subfolder):
        dataset_path = train_subfolder

    return len([
        entry for entry in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, entry))
    ])

@dataclass
class FaceRecognitionConfig:
    """Hyperparameters used during training and evaluation."""
    num_classes: int = 0
    image_size: tuple[int, int] = (112, 112)
    input_shape: tuple[int, int, int] = (112, 112, 3)
    embedding_size: int = 512

    dataset_fraction: float = 1

    # Training hyperparameters ---------------------------------------------
    batch_size: int = 64
    warmup_epochs: int = 5
    epochs: int = 30

    optimizer: str = "SGD"
    learning_rate: float = 0.005
    min_learning_rate: float = 1e-5
    momentum: float = 0.9
    weight_decay: float = 0.0005

    lr_schedule: dict = field(default_factory=lambda: {
        "type": "MultiStepLR",
        "params": {
            "milestones": [10, 20, 25],
            "gamma": 0.1,
        },
    })

    cosine_margin: float = 0.35 
    cosine_scale: float = 30.0

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
