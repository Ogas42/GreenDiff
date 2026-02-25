from __future__ import annotations

from typing import Any, Dict, Tuple


def infer_loader_worker_settings(config: Dict[str, Any], dataset: Any) -> Dict[str, Any]:
    data_cfg = config.get("data", {})
    shuffle = data_cfg.get("shuffle")
    if shuffle is None:
        shuffle = not bool(getattr(dataset, "use_shards", False))
    num_workers = int(data_cfg.get("num_workers", 0))
    if bool(getattr(dataset, "use_shards", False)):
        shard_workers = data_cfg.get("shard_workers")
        num_workers = min(num_workers, 4) if shard_workers is None else int(shard_workers)
    return {
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": bool(data_cfg.get("pin_memory", False)),
        "persistent_workers": bool(data_cfg.get("persistent_workers", False)),
        "prefetch_factor": int(data_cfg.get("prefetch_factor", 2)),
    }


def build_train_dataloader(config: Dict[str, Any], stage_train_cfg: Dict[str, Any], dist_ctx: Any, split: str = "train") -> Tuple[Any, Any, Any]:
    from torch.utils.data import DataLoader, DistributedSampler

    from gd.data.dataset import GFDataset

    dataset = GFDataset(config, split=split)
    opts = infer_loader_worker_settings(config, dataset)
    sampler = (
        DistributedSampler(dataset, num_replicas=dist_ctx.world_size, rank=dist_ctx.rank, shuffle=opts["shuffle"])
        if dist_ctx.is_distributed
        else None
    )
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=stage_train_cfg["batch_size"],
        shuffle=(opts["shuffle"] if sampler is None else False),
        num_workers=opts["num_workers"],
        pin_memory=opts["pin_memory"],
        sampler=sampler,
    )
    if opts["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = opts["persistent_workers"]
        loader_kwargs["prefetch_factor"] = opts["prefetch_factor"]
    return dataset, sampler, DataLoader(**loader_kwargs)

