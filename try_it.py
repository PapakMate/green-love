"""
try_it.py — hands-on demo for green-love.

Model: Deep ResNet-style CNN (~18M params) on 128×128 synthetic images.
       Heavy enough to saturate the GPU, still finishes benchmark epochs quickly.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from green_love import GreenLoveEstimator


# ─── Model ────────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """Residual block with optional channel expansion."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class HeavyResNet(nn.Module):
    """
    Deep ResNet: 4 stages × 3 residual blocks each = 24 conv layers + stem + head.
    Input: 3×128×128  →  Output: 100 classes
    ~18M parameters — keeps the GPU properly busy.
    """
    def __init__(self, num_classes: int = 100):
        super().__init__()
        # Stem: 3 → 64, 128×128 → 64×64
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # Stage 1: 64 → 64,  64×64
        self.stage1 = nn.Sequential(
            ResBlock(64, 64), ResBlock(64, 64), ResBlock(64, 64),
        )
        # Stage 2: 64 → 128, 64×64 → 32×32
        self.stage2 = nn.Sequential(
            ResBlock(64, 128, stride=2), ResBlock(128, 128), ResBlock(128, 128),
        )
        # Stage 3: 128 → 256, 32×32 → 16×16
        self.stage3 = nn.Sequential(
            ResBlock(128, 256, stride=2), ResBlock(256, 256), ResBlock(256, 256),
        )
        # Stage 4: 256 → 512, 16×16 → 8×8
        self.stage4 = nn.Sequential(
            ResBlock(256, 512, stride=2), ResBlock(512, 512), ResBlock(512, 512),
        )
        # Head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


# ─── Synthetic dataset ─────────────────────────────────────────────────────────

def make_dataloader(n_samples: int = 8_000, batch_size: int = 64) -> DataLoader:
    """Create a DataLoader with random 128×128 images."""
    images = torch.randn(n_samples, 3, 128, 128)
    labels = torch.randint(0, 100, (n_samples,))
    dataset = TensorDataset(images, labels)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


# ─── Training step ──────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, scaler, device):
    """Run one training epoch. Returns (avg_loss, accuracy %)."""
    model.train()
    total_loss = 0.0
    correct = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n * 100


# ─── Training loop ─────────────────────────────────────────────────────────────

def train(total_epochs: int = 500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    model = HeavyResNet(num_classes=100).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: HeavyResNet  |  Parameters: {n_params:,}")

    loader = make_dataloader(n_samples=8_000, batch_size=64)
    print(f"Batches per epoch:  {len(loader)}\n")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    # ── Estimator setup ──────────────────────────────────────────────────────
    estimator = GreenLoveEstimator(
        total_epochs=total_epochs,
        sample_epochs_pct=1.0,
        warmup_epochs=2,
        benchmark_task="resnet50",
        precision="fp16",
        auto_open_report=True,
    )

    # ── Phase 1: Benchmark ───────────────────────────────────────────────────
    # Estimator measures epoch timing, then generates a report and asks
    # whether to continue locally or move to Crusoe Cloud.
    for epoch in range(total_epochs):
        estimator.on_epoch_start(epoch)
        avg_loss, accuracy = train_epoch(model, loader, optimizer, criterion, scaler, device)
        print(f"Epoch {epoch+1:3d}/{total_epochs}  loss={avg_loss:.4f}  acc={accuracy:.1f}%")
        scheduler.step()
        if not estimator.on_epoch_end(epoch):
            break

    # ── Phase 2: Full training from scratch (if user chose to continue) ──────
    if estimator.user_chose_continue:
        print("\n🚀 Starting full training from epoch 1...\n")
        model = HeavyResNet(num_classes=100).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

        for epoch in range(total_epochs):
            avg_loss, accuracy = train_epoch(model, loader, optimizer, criterion, scaler, device)
            print(f"Epoch {epoch+1:3d}/{total_epochs}  loss={avg_loss:.4f}  acc={accuracy:.1f}%")
            scheduler.step()


if __name__ == "__main__":
    train(total_epochs=500)
