"""Fetch Fluent Speech Commands dataset from GitHub mirror.

Task: P0-08 / P1
Reference: docs/01 §1, docs/09 §1, docs/10 §4

Clones the repository with --depth 1, records exact commit hash and metadata to SOURCE.md,
and removes the nested .git directory.
"""

import argparse
import datetime
import os
import shutil
import subprocess
import sys

DEFAULT_REPO_URL = "https://github.com/GrindstoneLZX/FluentSpeechCommandsDataset.git"
DEFAULT_DEST = "data/raw/fluent_speech_commands_dataset"


def fetch_fsc(repo_url: str = DEFAULT_REPO_URL, dest: str = DEFAULT_DEST) -> None:
    dest_path = os.path.abspath(dest)
    parent_dir = os.path.dirname(dest_path)
    os.makedirs(parent_dir, exist_ok=True)

    if os.path.exists(dest_path) and os.path.exists(os.path.join(dest_path, "data")):
        print(f"[fetch_fsc] Direktori tujuan sudah ada dan berisi data: {dest_path}")
        return

    if os.path.exists(dest_path):
        print(f"[fetch_fsc] Membersihkan folder parsial: {dest_path}")
        if os.name == "nt":
            subprocess.run(["cmd", "/c", f'rmdir /s /q "{dest_path}"'], check=True)
        else:
            shutil.rmtree(dest_path)

    print(f'[fetch_fsc] Memulai shallow clone: git clone --depth 1 {repo_url} "{dest_path}"')
    # Jalankan git clone dengan output langsung terlihat
    process = subprocess.Popen(
        ["git", "clone", "--depth", "1", "--progress", repo_url, dest_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in iter(process.stdout.readline, ""):
        sys.stdout.write(line)
        sys.stdout.flush()

    process.stdout.close()
    return_code = process.wait()

    if return_code != 0:
        print(f"[fetch_fsc] Gagal melakukan git clone (exit code {return_code})", file=sys.stderr)
        sys.exit(return_code)

    # Ambil commit hash dan tanggal commit
    commit_res = subprocess.run(
        ["git", "-C", dest_path, "rev-parse", "HEAD"], capture_output=True, text=True
    )
    commit_hash = commit_res.stdout.strip()

    date_res = subprocess.run(
        ["git", "-C", dest_path, "log", "-1", "--format=%cd", "--date=iso"],
        capture_output=True,
        text=True,
    )
    commit_date = date_res.stdout.strip()

    # Catat provenance ke <dest>/../SOURCE.md
    source_md_path = os.path.join(parent_dir, "SOURCE.md")
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    source_content = (
        f"# Dataset Source Metadata\n\n"
        f"- **Source URL**: {repo_url}\n"
        f"- **Commit Hash**: {commit_hash}\n"
        f"- **Commit Date**: {commit_date}\n"
        f"- **Fetched At**: {timestamp}\n"
        f"- **Local Path**: {dest_path}\n"
    )
    with open(source_md_path, "w", encoding="utf-8") as f:
        f.write(source_content)
    print(f"[fetch_fsc] Berhasil mencatat provenance ke: {source_md_path}")

    # Hapus folder .git internal agar tidak mengotori repositori utama
    nested_git = os.path.join(dest_path, ".git")
    if os.path.exists(nested_git):
        print(f"[fetch_fsc] Menghapus folder .git internal: {nested_git}")
        if os.name == "nt":
            subprocess.run(["cmd", "/c", f'rmdir /s /q "{nested_git}"'], check=True)
        else:
            shutil.rmtree(nested_git)

    print(f"[fetch_fsc] Selesai! Dataset siap di: {dest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Fluent Speech Commands dataset mirror.")
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL, help="Git repository URL")
    parser.add_argument("--dest", default=DEFAULT_DEST, help="Destination directory")
    args = parser.parse_args()
    fetch_fsc(repo_url=args.repo_url, dest=args.dest)
