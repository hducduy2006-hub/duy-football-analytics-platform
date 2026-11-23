"""Deploy NCKH TUD65 Football Analytics to Hugging Face Spaces (Docker)."""

from __future__ import annotations

import argparse
import fnmatch
import getpass
import os
import re
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

ALLOW_PATTERNS = [
    "README.md",
    "Dockerfile",
    ".dockerignore",
    ".gitattributes",
    "requirements.txt",
    "start_hf.sh",
    ".streamlit/**",
    "app.py",
    "api_server.py",
    "pro_features.py",
    "prematch_research.py",
    "daily_realtime_hub.py",
    "daily_match_center_bridge.py",
    "team_logos_store.py",
    "player_avatar_store.py",
    "build_pro_analytics_store.py",
    "fifa18_complete_dataset.csv",
    "worldcup_2018_shots.csv",
    "worldcup_match_stats.csv",
    "player_photos/**",
    "sample_data/**",
]

IGNORE_PATTERNS = [
    "**/__pycache__/**",
    "**/*.pyc",
    "**/*.bak*",
    "**/*.log",
    "**/.DS_Store",
    "**/Thumbs.db",
    ".cache/**",
    ".git/**",
    "sofascore_platform.db*",
    "server_logs/**",
    "sample_data/_tmp_opta_dl/**",
]

REQUIRED_FILES = [
    "README.md",
    "Dockerfile",
    ".dockerignore",
    ".gitattributes",
    "requirements.txt",
    "start_hf.sh",
    ".streamlit/config.toml",
    "app.py",
    "api_server.py",
    "pro_features.py",
    "prematch_research.py",
    "daily_realtime_hub.py",
    "daily_match_center_bridge.py",
    "team_logos_store.py",
    "player_avatar_store.py",
    "build_pro_analytics_store.py",
    "fifa18_complete_dataset.csv",
    "worldcup_2018_shots.csv",
    "worldcup_match_stats.csv",
]

REQUIRED_NONEMPTY_DIRS = [
    "sample_data/pro_analytics",
    "player_photos/avatars_webp",
]


def _configure_stdio_utf8() -> None:
    """Ensure Vietnamese console output renders cleanly on Windows terminals."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _matches_pattern(rel_posix: str, pattern: str) -> bool:
    """Match a POSIX relative path against Hub-style glob patterns."""
    if pattern.endswith("/**"):
        prefix = pattern[:-3]
        if prefix.startswith("**/"):
            needle = f"/{prefix[3:]}/"
            return rel_posix.startswith(prefix[3:] + "/") or needle in f"/{rel_posix}"
        return rel_posix == prefix or rel_posix.startswith(prefix + "/")

    if pattern.startswith("**/"):
        sub = pattern[3:]
        if fnmatch.fnmatch(rel_posix, sub):
            return True
    return fnmatch.fnmatch(rel_posix, pattern)


def _should_skip_dir(rel_dir_posix: str) -> bool:
    """Short-circuit heavy excluded top-level trees during local scanning."""
    top = rel_dir_posix.split("/", 1)[0]
    return top in {
        ".git",
        ".cache",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        "server_logs",
        ".vscode",
        ".devcontainer",
        "apps",
        "data",
    } or rel_dir_posix == "sample_data/_tmp_opta_dl" or rel_dir_posix.endswith("/__pycache__")


def collect_upload_manifest(base_dir: Path = BASE_DIR) -> list[tuple[str, int]]:
    """Return sorted (rel_posix_path, byte_size) pairs included in the deploy payload."""
    manifest: list[tuple[str, int]] = []
    for root, dirs, files in os.walk(base_dir):
        rel_root = Path(root).relative_to(base_dir)
        kept_dirs = []
        for d in sorted(dirs):
            rel_d = (rel_root / d).as_posix() if rel_root != Path(".") else d
            if not _should_skip_dir(rel_d):
                kept_dirs.append(d)
        dirs[:] = kept_dirs

        for fname in sorted(files):
            rel_p = (rel_root / fname).as_posix() if rel_root != Path(".") else fname
            if not any(_matches_pattern(rel_p, pat) for pat in ALLOW_PATTERNS):
                continue
            if any(_matches_pattern(rel_p, pat) for pat in IGNORE_PATTERNS):
                continue
            fpath = Path(root) / fname
            manifest.append((rel_p, fpath.stat().st_size))
    manifest.sort(key=lambda item: item[0])
    return manifest


def validate_runtime_bundle(base_dir: Path = BASE_DIR) -> None:
    """Fail fast if any required code, config, or dataset asset is missing."""
    missing = [rel for rel in REQUIRED_FILES if not (base_dir / rel).is_file()]
    if missing:
        raise FileNotFoundError(
            "Thiếu tệp runtime bắt buộc:\n  - " + "\n  - ".join(missing)
        )

    for rel_dir in REQUIRED_NONEMPTY_DIRS:
        target = base_dir / rel_dir
        if not target.is_dir() or not any(target.iterdir()):
            raise FileNotFoundError(f"Thư mục dữ liệu bắt buộc bị thiếu hoặc rỗng: {rel_dir}")

    start_script = (base_dir / "start_hf.sh").read_bytes()
    if b"\r" in start_script:
        raise ValueError("start_hf.sh đang chứa CRLF (\\r); cần chuẩn hóa về LF (\\n).")


def space_direct_url(repo_id: str) -> str:
    """Compute the canonical *.hf.space direct app URL from 'owner/space_name'."""
    parts = [p.strip() for p in repo_id.split("/") if p.strip()]
    if len(parts) != 2:
        raise ValueError("Tên Space phải có dạng '<username>/<space_name>' (VD: duy2006/nckh-tud65-football).")
    owner_slug = re.sub(r"[^a-z0-9-]+", "-", parts[0].lower()).strip("-")
    space_slug = re.sub(r"[^a-z0-9-]+", "-", parts[1].lower()).strip("-")
    return f"https://{owner_slug}-{space_slug}.hf.space"


def _ensure_huggingface_hub():
    """Import huggingface_hub or install it automatically when performing an upload."""
    try:
        import huggingface_hub  # noqa: F401
        return huggingface_hub
    except ImportError:
        print("⏳ Chưa có thư viện 'huggingface_hub'. Đang tự động cài đặt...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "huggingface_hub"])
        import huggingface_hub
        return huggingface_hub


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Triển khai NCKH TUD65 Football Analytics lên Hugging Face Spaces (Docker)."
    )
    parser.add_argument(
        "--repo",
        help="Tên Space đích dạng <username>/<space_name> (VD: hducduy2006/nckh-tud65-football)",
    )
    parser.add_argument(
        "--token",
        help="Hugging Face Write Token (ưu tiên đọc từ biến môi trường HF_TOKEN hoặc nhập ẩn)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Tạo Space ở chế độ Private (mặc định Public để mở thẳng trên điện thoại không cần đăng nhập)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chỉ kiểm tra tệp và thống kê dung lượng tải lên, không kết nối mạng.",
    )
    return parser.parse_args()


def main() -> int:
    _configure_stdio_utf8()
    args = parse_args()

    print("=== NCKH TUD65 Football Analytics · Hugging Face Spaces Deployer ===")
    validate_runtime_bundle(BASE_DIR)
    manifest = collect_upload_manifest(BASE_DIR)

    total_bytes = sum(size for _, size in manifest)
    total_mb = total_bytes / (1024 * 1024)
    largest = sorted(manifest, key=lambda x: x[1], reverse=True)[:5]

    print(f"✅ Kiểm tra runtime hợp lệ: {len(manifest):,} tệp · Tổng {total_mb:.2f} MB")
    print("📦 Top tệp lớn nhất:")
    for rel, size in largest:
        print(f"   - {rel}: {size / (1024 * 1024):.2f} MB")

    if args.dry_run:
        print("🔍 Chế độ --dry-run hoàn tất (không tải lên).")
        return 0

    repo_id = (args.repo or input("👉 Nhập tên Space (<username>/<space_name>): ")).strip()
    direct_url = space_direct_url(repo_id)

    token = (args.token or os.environ.get("HF_TOKEN") or "").strip()
    if not token:
        token = getpass.getpass("🔑 Dán Hugging Face Write Token (nhập ẩn, không lưu ra đĩa): ").strip()
    if not token:
        print("❌ Thiếu Hugging Face Write Token.")
        return 1

    hub = _ensure_huggingface_hub()
    api = hub.HfApi(token=token)

    print(f"🚀 Đang tạo / đồng bộ Space Docker '{repo_id}' (private={args.private})...")
    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="docker",
        private=args.private,
        exist_ok=True,
    )

    print(f"📤 Đang tải {len(manifest):,} tệp ({total_mb:.2f} MB) lên Hugging Face Spaces...")
    if hasattr(api, "upload_large_folder"):
        api.upload_large_folder(
            repo_id=repo_id,
            repo_type="space",
            folder_path=str(BASE_DIR),
            allow_patterns=ALLOW_PATTERNS,
            ignore_patterns=IGNORE_PATTERNS,
        )
    else:
        api.upload_folder(
            repo_id=repo_id,
            repo_type="space",
            folder_path=str(BASE_DIR),
            allow_patterns=ALLOW_PATTERNS,
            ignore_patterns=IGNORE_PATTERNS,
            commit_message="Deploy NCKH TUD65 Football Analytics (Docker)",
        )

    print("\n🎉 TẢI LÊN HOÀN TẤT!")
    print(f"🌐 Trang quản lý & theo dõi build Docker: https://huggingface.co/spaces/{repo_id}")
    print(f"📱 Link mở app cố định (máy tính & điện thoại, chạy cả khi tắt PC): {direct_url}")
    print("⏳ Lần đầu Hugging Face sẽ mất ~2-3 phút để build Docker container.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
