from __future__ import annotations

from torch.utils.data import DataLoader, TensorDataset

from configs import PhysicalDatasetConfig
from filter.data_simulation import simulate_damped_oscillator_dataset


def make_physical_dataloader(
    cfg: PhysicalDatasetConfig,
    *,
    batch_size: int = 16,
    shuffle: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    _, train_obs = simulate_damped_oscillator_dataset(
        n_sequences=cfg.n_sequences,
        seq_len_obs=cfg.seq_len_obs,
        n_pred=cfg.n_pred,
        dt=cfg.dt,
        spring_k=cfg.spring_k,
        damping_c=cfg.damping_c,
        process_std=cfg.process_std,
        obs_std=cfg.obs_std,
        device=cfg.device,
    )

    ds = TensorDataset(train_obs)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )

