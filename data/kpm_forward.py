from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
import warnings

from gd.utils.obs_layout import is_sublattice_resolved


class KPMForward:
    """
    Forward operator for generating LDOS from a 2D potential using Kwant KPM
    (or direct inversion on small systems).
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.kpm_cfg = config["kpm"]
        self.ham_cfg = config["hamiltonian"]
        self.data_cfg = config.get("data", {})
        self.num_moments = int(self.kpm_cfg["moments"])
        self.sublattice_resolved_ldos = bool(is_sublattice_resolved(self.data_cfg))

        t_cfg = self.ham_cfg["t"]
        if isinstance(t_cfg, (list, tuple)):
            self.hopping_range = t_cfg
            self.hopping = float(np.mean(t_cfg))
        else:
            self.hopping_range = None
            self.hopping = float(t_cfg)
        self.current_hopping = self.hopping

        self.lattice_constant = self.ham_cfg["lattice_constant"]
        self.kernel = "jackson" if self.kpm_cfg["jackson_kernel"] else None
        self.num_vectors = self.kpm_cfg["num_random_vectors"]
        self.rng_seed = config["rng_seed"]
        self.mu = self.ham_cfg.get("mu", 0.0)
        self.add_mass = self.ham_cfg.get("add_mass", False)
        self.mass = self.ham_cfg.get("mass", 0.0)
        self.add_soc = self.ham_cfg.get("add_soc", False)
        self.soc_strength = self.ham_cfg.get("soc_strength", 0.0)
        self.add_nnn = self.ham_cfg.get("add_nnn", False)
        self.t2 = self.ham_cfg.get("t2", 0.0)
        self.add_mag_field = self.ham_cfg.get("add_mag_field", False)
        self.mag_field = self.ham_cfg.get("mag_field", 0.0)
        self.direct_inv_cfg = self.kpm_cfg.get("direct_inverse", {"enabled": False, "max_sites": 4096})

        self._graphene_feature_warned = False
        self.current_lattice_type = self._normalize_lattice_type(self.ham_cfg.get("type", "square_lattice"))
        self._last_sample_meta: Dict[str, Any] = {}
        self._current_defect_meta: Dict[str, Any] = {}
        self._mixed_lattice_warned = False
        self._random_lattice_types, self._random_lattice_probs = self._build_random_lattice_candidates()
        self._graphene_family_to_subidx = {}
        if self.sublattice_resolved_ldos:
            bad = [t for t in self._random_lattice_types if t != "graphene"]
            if bad:
                raise ValueError(
                    "data.sublattice_resolved_ldos=true requires graphene-only lattice sampling in Phase 1; "
                    f"got random_lattice_types={self._random_lattice_types!r}."
                )

    def compute_ldos(
        self,
        V: torch.Tensor,
        energies: Iterable[float],
        defect_meta: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            V: Potential tensor of shape (H, W).
            energies: Iterable of K energy points.
        Returns:
            LDOS tensor of shape (K, H, W) or (K, 2, H, W) when
            sublattice-resolved graphene LDOS is enabled.
        """
        V_np = V.detach().cpu().numpy()
        H, W = V_np.shape
        if self.direct_inv_cfg["enabled"] and (H * W) <= self.direct_inv_cfg["max_sites"]:
            return self._compute_ldos_direct(V_np, energies, defect_meta=defect_meta)

        import kwant
        import kwant.kpm

        syst = self._build_system(V_np, defect_meta=defect_meta)
        rng = self.rng_seed
        energies_np = np.array(list(energies), dtype=np.float64)

        t = max(abs(self.current_hopping), 1e-3)
        v_min, v_max = V_np.min(), V_np.max()
        bandwidth_margin = 12.0 * t
        sys_min = v_min - self.mu - bandwidth_margin
        sys_max = v_max - self.mu + bandwidth_margin
        bound_min = min(sys_min, energies_np.min()) - 0.5 * t
        bound_max = max(sys_max, energies_np.max()) + 0.5 * t

        op = kwant.operator.Density(syst, sum=False)
        kernel_func = kwant.kpm.jackson_kernel if self.kernel == "jackson" else None
        spectrum = kwant.kpm.SpectralDensity(
            syst,
            operator=op,
            num_moments=self.num_moments,
            kernel=kernel_func,
            rng=rng,
            bounds=(bound_min, bound_max),
            num_vectors=self.num_vectors,
        )
        ldos = spectrum(energies_np)
        if np.iscomplexobj(ldos):
            ldos = ldos.real

        out = self._allocate_ldos_output(len(energies_np), H, W)
        for i, site in enumerate(syst.sites):
            self._accumulate_site_ldos(out, site, ldos[:, i])
        return torch.from_numpy(out)

    def _compute_ldos_direct(
        self,
        V_np: np.ndarray,
        energies: Iterable[float],
        defect_meta: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Direct matrix inversion fallback for small systems."""
        import kwant

        energies_np = np.array(list(energies), dtype=np.float64)
        H, W = V_np.shape
        syst = self._build_system(V_np, defect_meta=defect_meta)
        Hmat = syst.hamiltonian_submatrix()

        eta = self.kpm_cfg.get("eta", 0.01)
        dim = Hmat.shape[0]
        ldos_flat = np.zeros((len(energies_np), dim), dtype=np.float32)
        I = np.eye(dim)
        for i, E in enumerate(energies_np):
            G = np.linalg.inv((E + 1j * eta) * I - Hmat)
            ldos_flat[i] = (-1.0 / np.pi) * np.imag(np.diag(G))

        norbs = self._num_orbs()
        out = self._allocate_ldos_output(len(energies_np), H, W)
        for i, site in enumerate(syst.sites):
            val = ldos_flat[:, i * norbs : (i + 1) * norbs].sum(axis=1)
            self._accumulate_site_ldos(out, site, val)
        return torch.from_numpy(out)

    def _build_system(self, V_np: np.ndarray, defect_meta: Optional[Dict[str, torch.Tensor]] = None):
        """Build a Kwant tight-binding system for square or graphene lattices."""
        import kwant

        H, W = V_np.shape
        lattice_type = self._normalize_lattice_type(self.ham_cfg.get("type", "square_lattice"))
        if lattice_type == "random":
            lattice_type = str(np.random.choice(self._random_lattice_types, p=self._random_lattice_probs))
        self.current_lattice_type = lattice_type

        if self.hopping_range is not None:
            self.current_hopping = float(np.random.uniform(self.hopping_range[0], self.hopping_range[1]))
        else:
            self.current_hopping = self.hopping

        norbs = self._num_orbs()
        self._last_sample_meta = {
            "lattice_type": self.current_lattice_type,
            "hopping": float(self.current_hopping),
            "eta": float(self.kpm_cfg.get("eta", 0.01)),
            "target_type": str(self.data_cfg.get("target_representation", "ldos_ab")),
            "num_orbs": int(norbs),
            "graphene_simplified_branch": bool(self.current_lattice_type == "graphene"),
        }
        self._current_defect_meta = self._normalize_defect_meta(defect_meta, H, W)
        self._last_sample_meta["defect_flags"] = {
            "has_vacancy": bool(self._current_defect_meta.get("vacancy_mask") is not None and np.any(self._current_defect_meta["vacancy_mask"])),
            "has_bond_disorder": bool(self._current_defect_meta.get("bond_mod") is not None and np.any(self._current_defect_meta["bond_mod"] != 0.0)),
            "has_onsite_ab_delta": bool(
                self._current_defect_meta.get("onsite_ab_delta") is not None
                and np.any(self._current_defect_meta["onsite_ab_delta"] != 0.0)
            ),
        }

        if lattice_type in ("graphene", "honeycomb"):
            if (self.add_nnn or self.add_mag_field or self.add_soc) and not self._graphene_feature_warned:
                warnings.warn(
                    (
                        "KPMForward graphene branch uses a simplified hopping construction and does not "
                        "fully support all direction-dependent effects (SOC / magnetic field / NNN). "
                        "Configured graphene physics may be only approximately represented."
                    ),
                    RuntimeWarning,
                )
                self._graphene_feature_warned = True

            lat = kwant.lattice.honeycomb(self.lattice_constant, norbs=norbs)
            a, b = lat.sublattices
            self._graphene_family_to_subidx = {a: 0, b: 1}
            syst = kwant.Builder()
            vac = self._current_defect_meta.get("vacancy_mask")
            onsite_ab = self._current_defect_meta.get("onsite_ab_delta")
            bond_mod = self._current_defect_meta.get("bond_mod")
            for i in range(H):
                for j in range(W):
                    if not self._graphene_site_vacant(vac, 0, i, j):
                        v_a = float(V_np[i, j] + self._graphene_onsite_ab_delta(onsite_ab, 0, i, j))
                        syst[a(i, j)] = self._onsite(v_a)
                    if not self._graphene_site_vacant(vac, 1, i, j):
                        v_b = float(V_np[i, j] + self._graphene_onsite_ab_delta(onsite_ab, 1, i, j))
                        syst[b(i, j)] = self._onsite(v_b)
            # Explicit A->B bonds so we can apply direction-dependent bond disorder.
            for i in range(H):
                for j in range(W):
                    if self._graphene_site_vacant(vac, 0, i, j):
                        continue
                    for bond_k, (bi, bj) in self._graphene_b_neighbors(i, j):
                        if not (0 <= bi < H and 0 <= bj < W):
                            continue
                        if self._graphene_site_vacant(vac, 1, bi, bj):
                            continue
                        if self._graphene_bond_missing(bond_mod, bond_k, i, j):
                            continue
                        hop = self._graphene_hop_matrix(bond_k, i, j, bond_mod)
                        syst[a(i, j), b(bi, bj)] = hop
            return syst.finalized()

        if self.sublattice_resolved_ldos:
            raise ValueError(
                "data.sublattice_resolved_ldos=true currently supports only graphene/honeycomb Hamiltonians."
            )

        self._graphene_family_to_subidx = {}
        lat = kwant.lattice.square(self.lattice_constant, norbs=norbs)
        syst = kwant.Builder()
        for x in range(H):
            for y in range(W):
                syst[lat(x, y)] = self._onsite(V_np[x, y])
        for x in range(H):
            for y in range(W):
                if x + 1 < H:
                    syst[lat(x, y), lat(x + 1, y)] = self._hop_matrix(1, 0, y)
                if y + 1 < W:
                    syst[lat(x, y), lat(x, y + 1)] = self._hop_matrix(0, 1, y)
                if self.add_nnn:
                    if x + 1 < H and y + 1 < W:
                        syst[lat(x, y), lat(x + 1, y + 1)] = self._nnn_matrix()
                    if x + 1 < H and y - 1 >= 0:
                        syst[lat(x, y), lat(x + 1, y - 1)] = self._nnn_matrix()
        return syst.finalized()

    def _allocate_ldos_output(self, num_energies: int, H: int, W: int) -> np.ndarray:
        if self.sublattice_resolved_ldos and self.current_lattice_type == "graphene":
            return np.zeros((num_energies, 2, H, W), dtype=np.float32)
        return np.zeros((num_energies, H, W), dtype=np.float32)

    def _site_sublattice_index(self, site) -> int:
        family = getattr(site, "family", None)
        if family in self._graphene_family_to_subidx:
            return int(self._graphene_family_to_subidx[family])
        fam_name = str(getattr(family, "name", ""))
        if fam_name.endswith("0"):
            return 0
        if fam_name.endswith("1"):
            return 1
        raise ValueError(f"Unable to map graphene site family to sublattice index: {family!r}")

    def _accumulate_site_ldos(self, out: np.ndarray, site, values: np.ndarray) -> None:
        x, y = site.tag
        if out.ndim == 3:
            out[:, x, y] += values
            return
        sub_idx = self._site_sublattice_index(site)
        out[:, sub_idx, x, y] += values

    def _normalize_defect_meta(self, defect_meta: Optional[Dict[str, Any]], H: int, W: int) -> Dict[str, Optional[np.ndarray]]:
        if defect_meta is None:
            return {"vacancy_mask": None, "onsite_ab_delta": None, "bond_mod": None}
        out: Dict[str, Optional[np.ndarray]] = {"vacancy_mask": None, "onsite_ab_delta": None, "bond_mod": None}
        if "vacancy_mask" in defect_meta and defect_meta["vacancy_mask"] is not None:
            vac = defect_meta["vacancy_mask"]
            vac_np = vac.detach().cpu().numpy() if isinstance(vac, torch.Tensor) else np.asarray(vac)
            if vac_np.shape != (2, H, W):
                raise ValueError(f"graphene vacancy_mask must have shape (2,{H},{W}), got {vac_np.shape}")
            out["vacancy_mask"] = vac_np.astype(bool, copy=False)
        if "onsite_ab_delta" in defect_meta and defect_meta["onsite_ab_delta"] is not None:
            onsite = defect_meta["onsite_ab_delta"]
            onsite_np = onsite.detach().cpu().numpy() if isinstance(onsite, torch.Tensor) else np.asarray(onsite)
            if onsite_np.shape != (2, H, W):
                raise ValueError(f"graphene onsite_ab_delta must have shape (2,{H},{W}), got {onsite_np.shape}")
            out["onsite_ab_delta"] = onsite_np.astype(np.float64, copy=False)
        if "bond_mod" in defect_meta and defect_meta["bond_mod"] is not None:
            bond = defect_meta["bond_mod"]
            bond_np = bond.detach().cpu().numpy() if isinstance(bond, torch.Tensor) else np.asarray(bond)
            if bond_np.shape != (3, H, W):
                raise ValueError(f"graphene bond_mod must have shape (3,{H},{W}), got {bond_np.shape}")
            out["bond_mod"] = bond_np.astype(np.float64, copy=False)
        return out

    def _graphene_site_vacant(self, vac: Optional[np.ndarray], sub_idx: int, i: int, j: int) -> bool:
        return bool(vac is not None and vac[sub_idx, i, j])

    def _graphene_onsite_ab_delta(self, onsite_ab: Optional[np.ndarray], sub_idx: int, i: int, j: int) -> float:
        if onsite_ab is None:
            return 0.0
        return float(onsite_ab[sub_idx, i, j])

    def _graphene_b_neighbors(self, i: int, j: int):
        # Direction convention shared with StructuralDefectSampler and LatentGreen residual.
        return ((0, (i, j)), (1, (i - 1, j)), (2, (i, j - 1)))

    def _graphene_bond_missing(self, bond_mod: Optional[np.ndarray], bond_k: int, i: int, j: int) -> bool:
        return bool(bond_mod is not None and bond_mod[bond_k, i, j] <= -1.0)

    def _graphene_hop_matrix(self, bond_k: int, i: int, j: int, bond_mod: Optional[np.ndarray]):
        scale = 1.0
        if bond_mod is not None:
            scale = 1.0 + float(bond_mod[bond_k, i, j])
        if self._num_orbs() == 1:
            return -self.current_hopping * scale
        return (-self.current_hopping * scale) * np.eye(2)

    def aggregate_sublattice_ldos(self, ldos: torch.Tensor) -> torch.Tensor:
        if ldos.dim() == 4 and ldos.shape[1] == 2:
            return ldos.sum(dim=1)
        return ldos

    def _num_orbs(self) -> int:
        return 2 if (self.add_soc or self.add_mass) else 1

    def _onsite(self, V: float) -> np.ndarray:
        if self.current_lattice_type in ("square_lattice", "square"):
            base = 4.0 * self.current_hopping + V - self.mu
        else:
            base = V - self.mu
        if self._num_orbs() == 1:
            return base
        I = np.eye(2)
        mat = base * I
        if self.add_mass:
            mat = mat + self.mass * np.array([[1.0, 0.0], [0.0, -1.0]])
        return mat

    def _hop_matrix(self, dx: int, dy: int, y: int) -> np.ndarray:
        phase = 1.0
        if self.add_mag_field and dx == 1 and dy == 0:
            phase = np.exp(1j * 2.0 * np.pi * self.mag_field * y)
        if self._num_orbs() == 1:
            return -self.current_hopping * phase
        I = np.eye(2)
        hop = -self.current_hopping * I
        if self.add_soc:
            if dx == 1 and dy == 0:
                hop = hop + (-1j * self.soc_strength) * np.array([[0.0, -1.0], [1.0, 0.0]])
            if dx == 0 and dy == 1:
                hop = hop + (1j * self.soc_strength) * np.array([[0.0, 1.0], [1.0, 0.0]])
        return hop * phase

    def _nnn_matrix(self) -> np.ndarray:
        if self._num_orbs() == 1:
            return -self.t2
        return -self.t2 * np.eye(2)

    def get_last_sample_meta(self) -> Dict[str, Any]:
        return dict(self._last_sample_meta)

    def _normalize_lattice_type(self, lattice_type: Any) -> str:
        s = str(lattice_type).strip().lower()
        if s in ("square", "square_lattice"):
            return "square_lattice"
        if s in ("graphene", "honeycomb"):
            return "graphene"
        if s == "random":
            return "random"
        return s

    def _build_random_lattice_candidates(self):
        raw_types = self.ham_cfg.get("random_lattice_types")
        if raw_types is None:
            types = ["square_lattice", "graphene"]
        else:
            if not isinstance(raw_types, (list, tuple)) or len(raw_types) == 0:
                raise ValueError("hamiltonian.random_lattice_types must be a non-empty list when provided.")
            types = [self._normalize_lattice_type(x) for x in raw_types]

        supported = {"square_lattice", "graphene"}
        bad = [t for t in types if t not in supported]
        if bad:
            raise ValueError(f"Unsupported random lattice types in random_lattice_types: {bad}")

        if len(set(types)) > 1 and not self._mixed_lattice_warned:
            warnings.warn(
                (
                    "KPMForward is mixing multiple lattice families in one dataset. "
                    "This is allowed, but training should record/condition on lattice type "
                    "to avoid entangling different Hamiltonian families."
                ),
                RuntimeWarning,
            )
            self._mixed_lattice_warned = True

        raw_weights = self.ham_cfg.get("random_lattice_weights")
        if raw_weights is None:
            return types, None
        if not isinstance(raw_weights, (list, tuple)) or len(raw_weights) != len(types):
            raise ValueError("hamiltonian.random_lattice_weights must match random_lattice_types in length.")

        probs = np.array(raw_weights, dtype=np.float64)
        if np.any(probs < 0):
            raise ValueError("hamiltonian.random_lattice_weights must be non-negative.")
        denom = float(probs.sum())
        if denom <= 0.0:
            raise ValueError("hamiltonian.random_lattice_weights must sum to > 0.")
        probs = probs / denom
        return types, probs
