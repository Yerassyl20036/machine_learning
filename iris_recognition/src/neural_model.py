"""
Part 2 & 3 – Neural Network Architecture for Iris Recognition

Architecture: Multi-Branch Fusion CNN
  Each branch mirrors one classical algorithm's intuition:
    Branch 1 (Gabor/Daugman): captures phase/frequency texture
    Branch 2 (LBP):           captures local micro-texture
    Branch 3 (HOG):           captures gradient orientation structure
    Branch 4 (ORB):           captures keypoint-level structure

  All branches share the same input (normalized iris image).
  Branch outputs are concatenated and passed to a fusion head → identity class.

Also defines:
  - IrisNet: full classification model (Part 2)
  - EndToEndIrisNet: wraps preprocessing + IrisNet in a single forward pass (Part 3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Branch definitions ────────────────────────────────────────────────────────

class GaborBranch(nn.Module):
    """
    Mimics Daugman IrisCode: uses elongated horizontal conv kernels
    to capture horizontal texture patterns (like 1-D Gabor filters per row).
    """
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 15), padding=(0, 7)),  # horizontal filters
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 7), padding=(1, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 8)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 8, out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class LBPBranch(nn.Module):
    """
    Mimics LBP: uses small local neighbourhood convolutions
    to capture micro-texture patterns.
    """
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class HOGBranch(nn.Module):
    """
    Mimics HOG: uses depthwise + pointwise convolutions to capture
    gradient orientations (similar to HOG cell histograms).
    """
    def __init__(self, out_dim: int = 128):
        super().__init__()
        # Sobel-like gradient filters
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1, groups=1),   # edge detection
            nn.ReLU(),
            nn.Conv2d(8, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class ORBBranch(nn.Module):
    """
    Mimics ORB keypoint descriptors: uses wider receptive field convolutions
    to capture structural keypoint-like features.
    """
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


# ── Fusion head ───────────────────────────────────────────────────────────────

class FusionHead(nn.Module):
    """
    Concatenates all branch outputs and maps to identity classes.
    Includes an attention-weighted fusion layer.
    """
    def __init__(self, branch_dim: int = 128, n_branches: int = 4,
                 n_classes: int = 16, dropout: float = 0.4):
        super().__init__()
        fused_dim = branch_dim * n_branches

        # Attention: learn per-branch importance weights
        self.attention = nn.Sequential(
            nn.Linear(fused_dim, n_branches),
            nn.Softmax(dim=1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, n_classes),
        )

    def forward(self, branch_outputs: list[torch.Tensor]) -> torch.Tensor:
        fused = torch.cat(branch_outputs, dim=1)      # (B, branch_dim * n_branches)

        # Attention-weighted fusion
        attn = self.attention(fused)                   # (B, n_branches)
        # Reshape and apply attention per branch
        B, D = fused.shape
        n_b = attn.shape[1]
        per_branch = fused.view(B, n_b, D // n_b)     # (B, n_branches, branch_dim)
        weighted = per_branch * attn.unsqueeze(-1)     # (B, n_branches, branch_dim)
        fused_weighted = weighted.view(B, D)           # (B, fused_dim)

        return self.classifier(fused_weighted)


# ── Full model (Part 2) ───────────────────────────────────────────────────────

class IrisNet(nn.Module):
    """
    Multi-Branch Fusion CNN for iris recognition.

    Input:  (B, 1, H, W) normalized iris image (float32, [0,1])
    Output: (B, n_classes) logits
    """

    def __init__(self, n_classes: int = 16, branch_dim: int = 128,
                 dropout: float = 0.4):
        super().__init__()
        self.gabor_branch = GaborBranch(branch_dim)
        self.lbp_branch   = LBPBranch(branch_dim)
        self.hog_branch   = HOGBranch(branch_dim)
        self.orb_branch   = ORBBranch(branch_dim)
        self.fusion       = FusionHead(branch_dim, n_branches=4,
                                       n_classes=n_classes, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.gabor_branch(x)
        b2 = self.lbp_branch(x)
        b3 = self.hog_branch(x)
        b4 = self.orb_branch(x)
        return self.fusion([b1, b2, b3, b4])

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Return fused embedding (before classifier) for metric learning."""
        b1 = self.gabor_branch(x)
        b2 = self.lbp_branch(x)
        b3 = self.hog_branch(x)
        b4 = self.orb_branch(x)
        return torch.cat([b1, b2, b3, b4], dim=1)


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class IrisDataset(torch.utils.data.Dataset):
    """
    Loads preprocessed iris images (numpy arrays) and returns tensors.
    """

    def __init__(self, images: list, labels: list,
                 augment: bool = False, target_size: int = 128):
        self.images = images
        self.labels = labels
        self.augment = augment
        self.target_size = target_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].astype('float32') / 255.0  # (H, W)
        label = self.labels[idx]

        if self.augment:
            img = self._augment(img)

        tensor = torch.from_numpy(img).unsqueeze(0)        # (1, H, W)
        return tensor, label

    @staticmethod
    def _augment(img):
        import numpy as np
        # Horizontal flip
        if np.random.rand() > 0.5:
            img = np.fliplr(img).copy()
        # Small brightness jitter
        img = np.clip(img + np.random.uniform(-0.05, 0.05), 0, 1)
        return img


# ── Training utilities ────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        total += len(labels)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        total += len(labels)
    return total_loss / total, correct / total


def train(model, train_loader, val_loader,
          n_epochs: int = 30,
          lr: float = 1e-3,
          device: str | None = None) -> dict:
    """
    Train IrisNet and return history dict with loss/acc per epoch.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, n_epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss, vl_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        print(f"Epoch {epoch:3d}/{n_epochs} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {vl_loss:.4f} acc {vl_acc:.4f}")

    return history
