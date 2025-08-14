#!/usr/bin/env python3
"""
CigKarst Dataset Preparation (Colab/Cloud-Friendly)

This script prepares a curated, lightweight dataset from the 7GB CigKarst data
and logs it to Weights & Biases as a Dataset Artifact, then links it to a
Dataset Registry collection. It is designed specifically for Colab:

- Always downloads raw data from Zenodo
- Uses Colab paths for data and outputs
- Logs to W&B using fixed configuration variables (no CLI)

Author: Claude Code Assistant (adapted for Colab-friendly execution)
"""

from __future__ import annotations
import os
import zipfile
import json
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

# Make tqdm optional (for minimal environments)
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, desc=None, total=None, unit=None, unit_scale=None):
        if desc:
            print(f"{desc}...")
        return iterable

try:
    import requests
except ImportError:
    requests = None  # We'll check at runtime if needed

try:
    import wandb
except ImportError:
    wandb = None  # We'll check at runtime


# ======================
# Configuration (Colab)
# ======================
# Set these directly in Colab by editing the values below.
WANDB_ENTITY = "geo-prior-shell"
WANDB_PROJECT = "wandb-example"
ARTIFACT_NAME = "cigkarst-geological-samples"

# Dataset Registry components (matches prepare_dataset.py behaviour)
# Final link target will be f"{REGISTRY_NAMESPACE}/{REGISTRY_DATASET_NAME}"
REGISTRY_NAMESPACE = "wandb-registry-dataset"
REGISTRY_DATASET_NAME = "CigKarst"  # Set to your dataset collection name

# Paths (auto-detect Colab vs local)
def _running_in_colab() -> bool:
    return os.path.exists("/content") or os.environ.get("COLAB_RELEASE_TAG") is not None

if _running_in_colab():
    DATA_DIR = Path("/content/data")
    OUTPUT_DIR = Path("/content/curated_cigkarst_samples")
else:
    DATA_DIR = Path("./data")
    OUTPUT_DIR = Path("./curated_cigkarst_samples")

# Curation parameters
PATCH_SIZE = 64
NUM_PATCHES = 8
MAX_PAIRS = 3

# =========================
# Data Discovery and Loading
# =========================
def discover_dat_files(data_dir: Path) -> Tuple[List[Path], List[Path]]:
    """Discover seismic and karst .dat files in the data directory."""
    print(f"Scanning {data_dir} for .dat files...")

    possible_seismic_dirs = [
        data_dir / "noise",
        data_dir / "seismic",
        data_dir,
    ]
    possible_karst_dirs = [
        data_dir / "karst",
        data_dir / "labels",
        data_dir,
    ]

    seismic_files: List[Path] = []
    karst_files: List[Path] = []

    for seismic_dir in possible_seismic_dirs:
        if seismic_dir.exists():
            seismic_files.extend(sorted(seismic_dir.glob("*.dat")))
            if seismic_files:
                print(f"  Found {len(seismic_files)} seismic .dat files in {seismic_dir}")
                break

    for karst_dir in possible_karst_dirs:
        if karst_dir.exists():
            karst_files.extend(sorted(karst_dir.glob("*.dat")))
            if karst_files:
                print(f"  Found {len(karst_files)} karst .dat files in {karst_dir}")
                break

    return seismic_files, karst_files


def load_dat_volume(file_path: Path) -> Optional[np.ndarray]:
    """Load a single CigKarst .dat volume file (expects 256^3 float32)."""
    try:
        data = np.fromfile(file_path, dtype=np.float32)
        expected_size = 256 ** 3
        if data.size != expected_size:
            print(f"  Warning: {file_path.name} has size {data.size}, expected {expected_size}")
            return None
        volume = data.reshape(256, 256, 256)
        if np.isnan(volume).any() or np.isinf(volume).any():
            print(f"  Warning: {file_path.name} contains NaN/Inf values")
            return None
        return volume
    except Exception as exc:  # pragma: no cover
        print(f"  Failed to load {file_path}: {exc}")
        return None


def find_matching_pairs(
    seismic_files: List[Path], karst_files: List[Path], max_pairs: int
) -> List[Tuple[Path, Path]]:
    """Match seismic/karst files by name; fallback to approximate matches."""
    print(f"Finding matching seismic/karst pairs (max {max_pairs})...")

    karst_by_stem = {p.stem: p for p in karst_files}
    pairs: List[Tuple[Path, Path]] = []

    # Exact matches first
    for s in seismic_files:
        if s.stem in karst_by_stem:
            pairs.append((s, karst_by_stem[s.stem]))
            print(f"  Matched: {s.stem}")
            if len(pairs) >= max_pairs:
                return pairs

    # Approximate matches if needed
    if len(pairs) < max_pairs:
        print(f"  Only {len(pairs)} exact matches; searching approximate names...")
        for s in seismic_files:
            if any(ps == s for ps, _ in pairs):
                continue
            best = None
            s_name = s.stem
            for k in karst_files:
                k_name = k.stem
                if (
                    s_name in k_name
                    or k_name in s_name
                    or len(set(s_name) & set(k_name)) > len(s_name) * 0.5
                ):
                    best = k
                    break
            if best is not None:
                pairs.append((s, best))
                print(f"  Approximate: {s_name} <-> {best.stem}")
                if len(pairs) >= max_pairs:
                    break

    print(f"  Found {len(pairs)} pairs total")
    return pairs


def extract_patches_from_volume(
    volume: np.ndarray, patch_size: int, num_patches: int, seed: int = 42
) -> List[Tuple[np.ndarray, Tuple[int, int, int]]]:
    """Extract random cubic patches from a 256^3 volume."""
    rng = np.random.default_rng(seed)
    patches: List[Tuple[np.ndarray, Tuple[int, int, int]]] = []
    max_origin = 256 - patch_size
    for _ in range(num_patches):
        x = int(rng.integers(0, max_origin + 1))
        y = int(rng.integers(0, max_origin + 1))
        z = int(rng.integers(0, max_origin + 1))
        patch = volume[x : x + patch_size, y : y + patch_size, z : z + patch_size]
        if patch.shape == (patch_size, patch_size, patch_size):
            patches.append((patch, (x, y, z)))
    return patches


def create_curated_dataset(
    data_dir: Path,
    output_dir: Path,
    patch_size: int,
    num_patches: int,
    max_pairs: int,
) -> bool:
    """Create curated dataset from raw .dat volumes and save as .npz pairs."""
    print("Creating curated dataset...")
    seismic_files, karst_files = discover_dat_files(data_dir)
    if not seismic_files:
        print("No seismic .dat files found.")
        return False
    if not karst_files:
        print("No karst .dat files found.")
        return False

    pairs = find_matching_pairs(seismic_files, karst_files, max_pairs=max_pairs)
    if not pairs:
        print("No matching file pairs found.")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    curated_samples: List[dict] = []

    for i, (seis_fp, karst_fp) in enumerate(pairs):
        print(f"Processing pair {i+1}/{len(pairs)}: {seis_fp.stem}")
        seismic_volume = load_dat_volume(seis_fp)
        karst_volume = load_dat_volume(karst_fp)
        if seismic_volume is None or karst_volume is None:
            print("  Skipping pair due to load errors")
            continue

        seismic_patches = extract_patches_from_volume(
            seismic_volume, patch_size=patch_size, num_patches=num_patches
        )
        karst_patches = extract_patches_from_volume(
            karst_volume, patch_size=patch_size, num_patches=num_patches
        )

        for j, ((seis_patch, seis_xyz), (karst_patch, karst_xyz)) in enumerate(
            zip(seismic_patches, karst_patches)
        ):
            sample_id = f"volume_{i}_patch_{j}"

            seis_path = output_dir / f"{sample_id}_seismic.npy"
            karst_path = output_dir / f"{sample_id}_karst.npy"

            # Do not rewrite existing patch files to keep artifact bytes identical across runs
            if not seis_path.exists():
                _write_deterministic_npy(
                    seis_path,
                    patch_array=seis_patch,
                    coordinates=seis_xyz,
                    volume_source=seis_fp.name,
                )
            if not karst_path.exists():
                _write_deterministic_npy(
                    karst_path,
                    patch_array=karst_patch,
                    coordinates=karst_xyz,
                    volume_source=karst_fp.name,
                )

            curated_samples.append(
                {
                    "sample_id": sample_id,
                    "seismic_file": seis_path.name,
                    "karst_file": karst_path.name,
                    "coordinates": [int(seis_xyz[0]), int(seis_xyz[1]), int(seis_xyz[2])],
                    "source_volume": seis_fp.name,
                    "patch_size": int(patch_size),
                    "data_range_seismic": [
                        float(np.min(seis_patch).item()),
                        float(np.max(seis_patch).item()),
                    ],
                    "data_range_karst": [
                        float(np.min(karst_patch).item()),
                        float(np.max(karst_patch).item()),
                    ],
                }
            )

        print(f"  Created {len(seismic_patches)} patch pairs")

    # Save deterministic metadata JSON (avoids zip timestamp diffs)
    metadata_path = output_dir / "dataset_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "samples": curated_samples,
                "total_samples": len(curated_samples),
                "patch_size": patch_size,
                # Use a canonical, machine-agnostic source identifier to avoid cross-machine diffs
                "source_directory": "zenodo://4285733",
            },
            f,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    total_size_mb = sum(f.stat().st_size for f in output_dir.glob("*")) / (1024 * 1024)
    print(f"Curated dataset created: {len(curated_samples)} samples (~{total_size_mb:.1f} MB)")
    return True


# =========================
# Zenodo Download Utilities
# =========================
ZENODO_URLS = {
    "seismic.zip": "https://zenodo.org/records/4285733/files/seismic.zip?download=1",
    "karst.zip": "https://zenodo.org/records/4285733/files/karst.zip?download=1",
}


def download_cigkarst_zenodo(data_dir: Path) -> bool:
    """Download CigKarst zips from Zenodo into data_dir.

    Idempotent behavior:
    - If extracted data already exists (noise/*.dat and karst/*.dat), skip downloads.
    - If zips already exist, skip downloading those files.
    """
    if requests is None:  # pragma: no cover
        print("The 'requests' package is required for Zenodo download.")
        return False

    data_dir.mkdir(parents=True, exist_ok=True)

    # If extracted data is already present, skip downloading entirely
    noise_dir = data_dir / "noise"
    karst_dir = data_dir / "karst"
    if noise_dir.exists() and any(noise_dir.glob("*.dat")) and \
       karst_dir.exists() and any(karst_dir.glob("*.dat")):
        print("Detected existing extracted data (noise/karst .dat). Skipping downloads.")
        return True

    print("Downloading CigKarst zips from Zenodo...")
    for filename, url in ZENODO_URLS.items():
        dst = data_dir / filename
        if dst.exists():
            print(f"  {filename} already exists, skipping.")
            continue
        try:
            print(f"  Downloading {filename}...")
            with requests.get(url, stream=True, timeout=300) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0))
                with open(dst, "wb") as f, tqdm(
                    desc=filename, total=total, unit="B", unit_scale=True
                ) as pbar:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
        except Exception as exc:  # pragma: no cover
            print(f"  Failed to download {filename}: {exc}")
            return False
    print("  Downloads complete.")
    return True


def extract_cigkarst_archives(data_dir: Path) -> bool:
    """Extract CigKarst zips to data_dir/noise and data_dir/karst, then remove zips."""
    print("Extracting dataset archives...")
    archives = {
        data_dir / "seismic.zip": data_dir / "noise",
        data_dir / "karst.zip": data_dir / "karst",
    }

    # Early exit if both outputs already have extracted data
    if all(extract_dir.exists() and any(extract_dir.glob("*.dat")) for extract_dir in archives.values()):
        print("  Detected existing extracted data. Skipping extraction.")
        return True
    for archive_path, extract_dir in archives.items():
        if not archive_path.exists():
            print(f"  Missing archive: {archive_path.name}")
            return False

        if extract_dir.exists() and any(extract_dir.glob("*.dat")):
            print(f"  {archive_path.name} appears already extracted.")
            continue

        try:
            print(f"  Extracting {archive_path.name}...")
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive_path, "r") as zf:
                for info in zf.infolist():
                    if info.filename.endswith(".dat"):
                        info.filename = Path(info.filename).name  # flatten structure
                        zf.extract(info, extract_dir)
        except Exception as exc:  # pragma: no cover
            print(f"  Failed to extract {archive_path.name}: {exc}")
            return False

    '''# Cleanup zips to save space 
    for archive_path in archives.keys():
        if archive_path.exists():
            archive_path.unlink()
            print(f"  Removed {archive_path.name}")
  '''
    print("  Extraction complete.")
    return True


# =========================
# W&B Logging and Registry
# =========================
def _convert_numpy_to_python(obj):
    """Recursively convert numpy types to Python primitives for metadata."""
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _convert_numpy_to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_numpy_to_python(v) for v in obj]
    return obj


def _write_deterministic_npy(
    dst_path: Path,
    *,
    patch_array: np.ndarray,
    coordinates: Tuple[int, int, int],
    volume_source: str,
):
    """Write a deterministic .npy and a deterministic JSON sidecar.

    - Enforces C-contiguous layout and float32 little-endian dtype for cross-machine determinism.
    - Keeps per-patch metadata in a tiny sidecar JSON, leaving the .npy bytes independent.
    """
    # Ensure C-contiguous float32 little-endian
    patch_c = np.ascontiguousarray(patch_array, dtype=np.float32)
    if patch_c.dtype.byteorder not in ("<", "|"):
        patch_c = patch_c.astype(np.dtype("<f4"), copy=False)
    # Save NPY (deterministic bytes for identical arrays)
    np.save(dst_path, patch_c)
    # Sidecar JSON (deterministic text file)
    sidecar = dst_path.with_suffix(".json")
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(
            {
                "coordinates": [int(coordinates[0]), int(coordinates[1]), int(coordinates[2])],
                "volume_source": str(volume_source),
                "shape": list(patch_c.shape),
                "dtype": str(patch_c.dtype),
            },
            f,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

def log_to_wandb_registry(
    curated_dir: Path,
    entity: str,
    project: str,
    registry_target_path: str,
    artifact_name: str,
) -> str:
    """Create and log a dataset artifact to W&B, then link it to a Registry path."""
    if wandb is None:  # pragma: no cover
        print("The 'wandb' package is required to log artifacts.")
        return ""

    print("Logging dataset to Weights & Biases...")
    run = wandb.init(entity=entity, project=project, job_type="dataset-curation")

    try:
        # JSON-only metadata for determinism (no NPZ anywhere)
        meta_json_path = curated_dir / "dataset_metadata.json"
        if meta_json_path.exists():
            with open(meta_json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            total_samples = int(meta.get("total_samples", 0))
        else:
            raise FileNotFoundError("dataset_metadata.json is missing in curated directory")

        # Human-readable dataset card lives in the artifact description (Markdown)
        DATASET_DESCRIPTION = f"""# Dataset Card: CigKarst (Workshop Demo)

## Overview
- **Domain**: Geological volumes (Y = seismic condition, X = karst structures)
- **Demo patch size**: 64×64×64 voxels (small for fast notebooks); larger contexts exist offline
- **Dtype**: float32

## Contents
- Pairs of patches per sample: Y (seismic) and X (karst)
- Fixed validation samples referenced in the workshop notebook

## Lineage
- **Registry entry**: {registry_target_path}
- **Source (Zenodo)**: https://zenodo.org/records/4285733
- **Intended use**: Conditional diffusion training/evaluation in workshop demos

## Processing and Subset Rationale
- Randomly extracted small cubic patches (64³) from original 256³ volumes to keep the demo lightweight and reproducible
- Each patch saved with coordinates and a reference to the source volume
- No normalization at save-time; per-patch value ranges are recorded in metadata
- A small, representative subset is used so Colab/notebook runs complete quickly while preserving signal characteristics

## Suggested Evaluation in W&B
- Per-epoch 2D slice grid: Y, X, X_pred, Residual
- Optional 3D at 0/mid/last epochs (PyVista Html), with files downloadable from the artifact
"""

        artifact = wandb.Artifact(
            name=artifact_name,
            type="dataset",
            description=DATASET_DESCRIPTION,
            metadata={
                "source": "CigKarst Project - Zenodo DOI: 10.5281/zenodo.4285733",
                "source_url": "https://zenodo.org/records/4285733",
                "total_samples": total_samples,
                "patch_size": "64x64x64 voxels",
                "data_format": "numpy array (.npy)",
                "domain": "geological_seismic",
                "subset_rationale": "Small representative subset for fast, reproducible workshop runs in notebooks/Colab.",
                "processing": [
                    "Randomly extract 64³ patches from 256³ volumes for both Y (seismic) and X (karst)",
                    "Save each patch with coordinates and source filename",
                    "Record per-patch value ranges in metadata",
                ],
                "intended_use": "Training/evaluation for conditional diffusion workshop demos on limited compute",
                "registry_entry": registry_target_path,
            },
        )

        # Ensure legacy metadata NPZ is not present (and won't be uploaded)
        legacy_meta_npz = curated_dir / "dataset_metadata.npz"
        if legacy_meta_npz.exists():
            try:
                legacy_meta_npz.unlink()
                print("  Removed legacy dataset_metadata.npz from curated directory")
            except Exception:
                pass

        # Add all .npy patch files
        for file_path in tqdm(sorted(curated_dir.glob("*.npy")), desc="Adding NPY files"):
            artifact.add_file(str(file_path))
        # Add dataset-level metadata JSON
        if meta_json_path.exists():
            artifact.add_file(str(meta_json_path))
        # Add sidecar JSONs for each patch (but avoid re-adding dataset_metadata.json)
        for sidecar_json in sorted(curated_dir.glob("*.json")):
            if sidecar_json.name == "dataset_metadata.json":
                continue
            artifact.add_file(str(sidecar_json))

        logged_artifact = wandb.log_artifact(artifact)
        print("  Artifact logged.")

        # Link to Registry path (e.g., 'wandb-registry-dataset/CigKarst')
        run.link_artifact(artifact=logged_artifact, target_path=registry_target_path)
        print(f"  Linked to Registry: {registry_target_path}")

        # Quick preview images
        sample_files = list(curated_dir.glob("*_seismic.npy"))[:3]
        images = []
        for f in sample_files:
            arr = np.load(f)
            mid = arr[:, :, arr.shape[2] // 2]
            vmin, vmax = float(np.min(mid)), float(np.max(mid))
            norm = (mid - vmin) / (vmax - vmin + 1e-12) if vmax > vmin else np.zeros_like(mid)
            images.append(wandb.Image(norm, caption=f"Seismic slice: {f.stem}"))
        if images:
            wandb.log({"dataset_samples": images})

        print("Dataset successfully logged to W&B.")
        return f"{artifact.name}:latest"

    except Exception as exc:  # pragma: no cover
        print(f"Failed to log to W&B: {exc}")
        import traceback

        traceback.print_exc()
        return ""
    finally:
        run.finish()


def main() -> bool:
    print("CigKarst Dataset Preparation (Colab)")
    print("=" * 60)
    print(f"W&B entity:           {WANDB_ENTITY}")
    print(f"W&B project:          {WANDB_PROJECT}")
    registry_target = f"{REGISTRY_NAMESPACE}/{REGISTRY_DATASET_NAME}"
    print(f"Registry target:      {registry_target}")
    print(f"Data dir:             {DATA_DIR}")
    print(f"Output dir:           {OUTPUT_DIR}")
    print(f"Patch size:           {PATCH_SIZE}")
    print(f"Num patches/volume:   {NUM_PATCHES}")
    print(f"Max pairs:            {MAX_PAIRS}")

    # Acquire data from Zenodo
    if not download_cigkarst_zenodo(DATA_DIR):
        print("\nFailed to download data from Zenodo.")
        return False
    if not extract_cigkarst_archives(DATA_DIR):
        print("\nFailed to extract Zenodo archives.")
        return False

    # Create curated dataset
    created = create_curated_dataset(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        patch_size=PATCH_SIZE,
        num_patches=NUM_PATCHES,
        max_pairs=MAX_PAIRS,
    )
    if not created:
        print("\nDataset curation failed.")
        return False

    # Log to W&B and link to Registry
    if wandb is None:
        print("\nwandb is not installed; please install it in Colab before running.")
        return False

    artifact_name = log_to_wandb_registry(
        curated_dir=OUTPUT_DIR,
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        registry_target_path=registry_target,
        artifact_name=ARTIFACT_NAME,
    )
    if not artifact_name:
        print("\nW&B Registry upload failed.")
        return False

    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETE")
    print(f"Artifact: {artifact_name}")
    print(f"Linked Registry: {registry_target}")
    return True


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)

