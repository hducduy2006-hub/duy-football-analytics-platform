"""Public cloud data synchronization for Streamlit Community Cloud deployments.

Locally, all datasets already live under ``sample_data/``, so every call resolves in
sub-millisecond fast-path checks with zero network I/O. In stateless cloud containers
(such as Streamlit Community Cloud), large Parquet/StatsBomb bundles excluded from
GitHub are fetched automatically from the public Hugging Face Dataset repository.
"""

from __future__ import annotations

import os
import sys
import tarfile
import threading
import urllib.parse
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = BASE_DIR / "sample_data"

HF_DATASET_REPO = os.environ.get("NCKH_HF_DATASET_REPO", "Duyyy2006/nckh-tud65-data")
HF_BASE_URL = f"https://huggingface.co/datasets/{HF_DATASET_REPO}/resolve/main"

MANAGED_PARQUET_FILES: tuple[str, ...] = (
    "pro_analytics/wc2018_events.parquet",
    "opta_modern/ucl_2627_events.parquet",
    "opta_modern/wc2026_events.parquet",
    "opta_modern/uel_events.parquet",
    "opta_modern/bundesliga_events.parquet",
    "opta_modern/ligue1_events.parquet",
    "opta_modern/seriea_events.parquet",
    "opta_modern/epl_events.parquet",
    "opta_modern/laliga_events.parquet",
    "opta_ucl/ucl_events.parquet",
)

_CHUNK_BYTES = 1024 * 1024
_USER_AGENT = "NCKH-TUD65-CloudSync/1.0"

_LOCKS_GUARD = threading.Lock()
_FILE_LOCKS: dict[str, threading.Lock] = {}
_BUNDLE_LOCK = threading.Lock()
_PREWARM_GUARD = threading.Lock()
_PREWARM_STARTED = False


def _normalize_rel_path(rel_path: str) -> str:
    norm = str(rel_path or "").replace("\\", "/").strip().lstrip("/")
    if norm.startswith("sample_data/"):
        norm = norm[len("sample_data/") :]
    return norm


def _get_file_lock(key: str) -> threading.Lock:
    with _LOCKS_GUARD:
        lock = _FILE_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _FILE_LOCKS[key] = lock
        return lock


def _count_files(dir_path: Path) -> int:
    if not dir_path.is_dir():
        return 0
    try:
        with os.scandir(dir_path) as it:
            return sum(1 for entry in it if entry.is_file())
    except OSError:
        return 0


def _atomic_download(url: str, dest_path: Path, timeout: int = 120) -> bool:
    """Download ``url`` atomically to ``dest_path`` using 1 MB chunks."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = Path(f"{dest_path}.tmp.{os.getpid()}.{threading.get_ident()}")
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp_path, "wb") as out_f:
            while True:
                chunk = resp.read(_CHUNK_BYTES)
                if not chunk:
                    break
                out_f.write(chunk)
        if tmp_path.stat().st_size <= 0:
            raise IOError(f"Empty download from {url}")
        os.replace(tmp_path, dest_path)
        return True
    except Exception as exc:
        print(f"[cloud_data_bootstrap] Failed downloading {url}: {exc}", file=sys.stderr)
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        return False


def _safe_extract_tar_gz(archive_path: Path, dest_dir: Path) -> None:
    """Safely extract tar.gz ensuring no path traversal or unsafe link targets."""
    dest_root = dest_dir.resolve()
    dest_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tf:
        for member in tf.getmembers():
            if not member.name or member.name.startswith("/") or member.issym() or member.islnk():
                continue
            member_dest = (dest_root / member.name).resolve()
            if member_dest != dest_root and dest_root not in member_dest.parents:
                raise ValueError(f"Unsafe tar member path blocked: {member.name}")
            if member.isdir():
                member_dest.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                member_dest.parent.mkdir(parents=True, exist_ok=True)
                extracted = tf.extractfile(member)
                if extracted is None:
                    continue
                with extracted, open(member_dest, "wb") as out_f:
                    while True:
                        buf = extracted.read(_CHUNK_BYTES)
                        if not buf:
                            break
                        out_f.write(buf)


def ensure_sample_file(rel_path: str, timeout: int = 120) -> str:
    """Ensure ``sample_data/<rel_path>`` exists locally, downloading from HF if missing."""
    rel_norm = _normalize_rel_path(rel_path)
    if not rel_norm:
        return str(SAMPLE_DIR)

    target = SAMPLE_DIR / rel_norm
    try:
        if target.is_file() and target.stat().st_size > 0:
            return str(target)
    except OSError:
        pass

    lock = _get_file_lock(rel_norm)
    with lock:
        try:
            if target.is_file() and target.stat().st_size > 0:
                return str(target)
        except OSError:
            pass

        encoded_rel = urllib.parse.quote(rel_norm, safe="/")
        url = f"{HF_BASE_URL}/{encoded_rel}"
        _atomic_download(url, target, timeout=timeout)

    return str(target)


def ensure_statsbomb_bundle(timeout: int = 120) -> bool:
    """Ensure StatsBomb events/ and lineups/ directories are populated."""
    events_dir = SAMPLE_DIR / "events"
    lineups_dir = SAMPLE_DIR / "lineups"
    if _count_files(events_dir) >= 100 and _count_files(lineups_dir) >= 100:
        return True

    with _BUNDLE_LOCK:
        if _count_files(events_dir) >= 100 and _count_files(lineups_dir) >= 100:
            return True

        SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
        archive_path = SAMPLE_DIR / f".statsbomb_bundle.tmp.{os.getpid()}.{threading.get_ident()}.tar.gz"
        url = f"{HF_BASE_URL}/statsbomb_events_lineups.tar.gz"
        try:
            ok = _atomic_download(url, archive_path, timeout=timeout)
            if not ok or not archive_path.is_file():
                return False
            _safe_extract_tar_gz(archive_path, SAMPLE_DIR)
            return _count_files(events_dir) >= 100 and _count_files(lineups_dir) >= 100
        except Exception as exc:
            print(f"[cloud_data_bootstrap] Failed extracting StatsBomb bundle: {exc}", file=sys.stderr)
            return False
        finally:
            try:
                if archive_path.exists():
                    archive_path.unlink()
            except OSError:
                pass


def _prewarm_parquets_worker(rel_paths: tuple[str, ...]) -> None:
    for rel in rel_paths:
        try:
            ensure_sample_file(rel, timeout=180)
        except Exception:
            pass


def bootstrap_cloud_data_once() -> dict:
    """Fast-path check locally; cold-start bundle sync + background pre-warmer in cloud."""
    global _PREWARM_STARTED

    bundle_ready = (
        _count_files(SAMPLE_DIR / "events") >= 100
        and _count_files(SAMPLE_DIR / "lineups") >= 100
    )
    missing_parquets = [
        rel
        for rel in MANAGED_PARQUET_FILES
        if not ((SAMPLE_DIR / rel).is_file() and (SAMPLE_DIR / rel).stat().st_size > 0)
    ]

    if bundle_ready and not missing_parquets:
        return {"mode": "local_complete", "missing_parquets": 0}

    if not bundle_ready:
        ensure_statsbomb_bundle(timeout=180)

    wc18_rel = "pro_analytics/wc2018_events.parquet"
    if wc18_rel in missing_parquets:
        ensure_sample_file(wc18_rel, timeout=120)

    remaining = tuple(rel for rel in missing_parquets if rel != wc18_rel)
    prewarmer_launched = False
    if remaining:
        with _PREWARM_GUARD:
            if not _PREWARM_STARTED:
                _PREWARM_STARTED = True
                t = threading.Thread(
                    target=_prewarm_parquets_worker,
                    args=(remaining,),
                    name="CloudParquetPrewarmer",
                    daemon=True,
                )
                t.start()
                prewarmer_launched = True

    return {
        "mode": "cloud_synced",
        "bundle_ready": _count_files(SAMPLE_DIR / "events") >= 100,
        "queued_parquets": len(remaining),
        "prewarmer_launched": prewarmer_launched,
    }
