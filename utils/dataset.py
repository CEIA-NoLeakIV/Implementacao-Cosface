import os
from torch.utils.data import Dataset
import torch
import numpy as np
from typing import Dict, List, Optional
from PIL import Image


class ImageFolder(Dataset):
    def __init__(
        self,
        root: str,
        transform=None,
        use_landmarks: bool = False,
        landmarks_dict: Optional[Dict[str, List[List[float]]]] = None
    ) -> None:
        self.root = root
        self.transform = transform
        self.use_landmarks = use_landmarks
        self.landmarks_dict = landmarks_dict
        self.samples = self._make_dataset(root)
        
        # Filtra samples se usar landmarks (remove imagens onde não detectou face)
        if self.use_landmarks and self.landmarks_dict is not None:
             self.samples = [s for s in self.samples if os.path.relpath(s[0], root) in self.landmarks_dict]

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = self._load_image(path)

        if self.transform:
            image = self.transform(image)

        if self.use_landmarks:
            # Recupera landmarks, converte para tensor
            rel_path = os.path.relpath(path, self.root)
            landmarks = np.array(self.landmarks_dict.get(rel_path), dtype=np.float32)
            return image, torch.from_numpy(landmarks), label

        return image, label

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _load_image(path: str) -> Image.Image:
        """Loads an image from the given path."""
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')

    @staticmethod
    def _make_dataset(directory: str):
        """Creates a dataset of image paths and corresponding labels."""
        class_names = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}

        instances = []
        for class_name, class_index in class_to_idx.items():
            class_dir = os.path.join(directory, class_name)

            for root, _, file_names in os.walk(class_dir, followlinks=True):
                for file_name in sorted(file_names):
                    path = os.path.join(root, file_name)
                    if os.path.splitext(path)[1].lower() in {".jpg", ".jpeg", ".png"}:
                        instances.append((path, class_index))

        return instances
