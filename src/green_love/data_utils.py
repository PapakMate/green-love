"""
DataLoader and Dataset subsampling utilities.

Used during the benchmark phase to run training on a subset of data
for faster estimation.
"""

import math
import logging
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader, Subset

logger = logging.getLogger(__name__)


def sample_dataset(
    dataset: Dataset,
    sample_pct: float = 10.0,
    seed: int = 42,
) -> Subset:
    """
    Create a random subset of a dataset.

    Args:
        dataset: The full PyTorch dataset.
        sample_pct: Percentage of data to keep (0-100).
        seed: Random seed for reproducibility.

    Returns:
        A torch.utils.data.Subset with the sampled indices.
    """
    total = len(dataset)
    n_samples = max(1, math.ceil(total * sample_pct / 100.0))
    n_samples = min(n_samples, total)

    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(total, generator=generator)[:n_samples].tolist()
    indices.sort()  # preserve ordering for determinism

    logger.info(
        f"Sampled {n_samples}/{total} data points "
        f"({sample_pct:.1f}%, seed={seed})"
    )
    return Subset(dataset, indices)


def sample_dataloader(
    dataloader: DataLoader,
    sample_pct: float = 10.0,
    seed: int = 42,
    batch_size: Optional[int] = None,
) -> DataLoader:
    """
    Create a new DataLoader from a subsampled version of the original dataset.

    Preserves the original DataLoader's configuration (batch_size, num_workers,
    pin_memory, etc.) except the dataset is replaced with a Subset.

    Args:
        dataloader: The original DataLoader.
        sample_pct: Percentage of data to keep (0-100).
        seed: Random seed for reproducibility.
        batch_size: Override batch size. If None, uses original.

    Returns:
        A new DataLoader wrapping the subsampled dataset.
    """
    subset = sample_dataset(dataloader.dataset, sample_pct, seed)

    bs = batch_size if batch_size is not None else dataloader.batch_size
    if bs is None:
        bs = 1

    # Preserve original DataLoader settings where possible
    kwargs = {
        "batch_size": bs,
        "num_workers": dataloader.num_workers,
        "pin_memory": dataloader.pin_memory,
        "drop_last": dataloader.drop_last,
    }

    # shuffle: use a new RandomSampler for the subset
    # (we don't carry over the original sampler since the dataset changed)
    if isinstance(dataloader.sampler, torch.utils.data.sampler.RandomSampler):
        kwargs["shuffle"] = True
    else:
        kwargs["shuffle"] = False

    new_loader = DataLoader(subset, **kwargs)
    logger.info(
        f"Created sampled DataLoader: {len(subset)} samples, "
        f"batch_size={bs}, ~{math.ceil(len(subset) / bs)} batches"
    )
    return new_loader
