"""
Dataset utilities: load image paths + labels from the iris/ folder.
Expected structure:
    iris_root/
        PersonName1/
            img_0001.jpg
            ...
        PersonName2/
            ...
"""

import os
from pathlib import Path


def load_dataset(iris_root: str) -> tuple[list[str], list[int], list[str]]:
    """
    Scan iris_root for person sub-folders containing images.

    Returns:
        paths  : list of absolute image file paths
        labels : list of integer class indices (0-based)
        classes: list of class names (folder names), sorted
    """
    root = Path(iris_root)
    classes = sorted(
        [d.name for d in root.iterdir() if d.is_dir()]
    )
    class_to_idx = {c: i for i, c in enumerate(classes)}

    paths, labels = [], []
    for cls in classes:
        cls_dir = root / cls
        for img_file in sorted(cls_dir.iterdir()):
            if img_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                paths.append(str(img_file))
                labels.append(class_to_idx[cls])

    return paths, labels, classes


def train_test_split_by_person(paths: list[str],
                                labels: list[int],
                                classes: list[str],
                                test_ratio: float = 0.2,
                                seed: int = 42):
    """
    Stratified split: keeps each person's images proportionally in train/test.
    Returns (train_paths, train_labels, test_paths, test_labels).
    """
    import random
    rng = random.Random(seed)

    train_paths, train_labels = [], []
    test_paths, test_labels = [], []

    for idx, cls in enumerate(classes):
        cls_paths = [p for p, l in zip(paths, labels) if l == idx]
        rng.shuffle(cls_paths)
        n_test = max(1, int(len(cls_paths) * test_ratio))
        test_paths.extend(cls_paths[:n_test])
        test_labels.extend([idx] * n_test)
        train_paths.extend(cls_paths[n_test:])
        train_labels.extend([idx] * (len(cls_paths) - n_test))

    return train_paths, train_labels, test_paths, test_labels
