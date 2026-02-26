import os
import sys

import yaml


def _ensure_project_root():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if root not in sys.path:
        sys.path.insert(0, root)


_ensure_project_root()

from gd.data.dataset import GFDataset  # noqa: E402


def test_dataset_reads_potential_normalize_from_shard_index_metadata(tmp_path):
    ds = object.__new__(GFDataset)
    ds._cache_schema_meta_found = {}
    ds._cache_transform_signature_found = None
    ds.shard_files = []
    ds.shard_cum_sizes = []
    ds.num_samples = 0
    ds.use_cache = False
    ds.use_shards = False

    cache_dir = tmp_path
    index_path = cache_dir / "train_index.yaml"
    payload = {
        "ldos_schema_version": 3,
        "target_representation": "ldos_ab",
        "contains_physics_meta": True,
        "contains_defect_meta": False,
        "potential_normalize": False,
        "sublattice_resolved_ldos": True,
        "ldos_canonical_layout": "k_s_h_w",
        "ldos_model_layout": "flat_channels",
        "sublattice_count": 2,
        "shards": [],
    }
    index_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with open(index_path, "r", encoding="utf-8") as f:
        index = yaml.safe_load(f) or {}
    ds._cache_schema_meta_found = {
        "ldos_schema_version": index.get("ldos_schema_version"),
        "target_representation": index.get("target_representation"),
        "contains_physics_meta": index.get("contains_physics_meta"),
        "contains_defect_meta": index.get("contains_defect_meta"),
        "potential_normalize": index.get("potential_normalize"),
        "sublattice_resolved_ldos": index.get("sublattice_resolved_ldos"),
        "ldos_canonical_layout": index.get("ldos_canonical_layout"),
        "ldos_model_layout": index.get("ldos_model_layout"),
        "sublattice_count": index.get("sublattice_count"),
    }
    assert ds._cache_schema_meta_found["potential_normalize"] is False

