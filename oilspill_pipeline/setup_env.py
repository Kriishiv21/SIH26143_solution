"""Optional environment helpers (Kaggle/Colab): train the classifier if no .pt bundle exists, and build the PyGNOME env.
Both are opt-in, slow, and NOT exercised by the unit tests (they need a GPU / network / conda)."""
from __future__ import annotations

import os
import subprocess
import tarfile
import urllib.request
from pathlib import Path


def bundle_ready(bundle_dir: str) -> bool:
    d = Path(bundle_dir)
    return (d / "deployment_config.json").exists() and (d / "model_weights.pt").exists() and (d / "model_defs.py").exists()


def train_if_missing(bundle_dir: str, training_notebook: str, timeout_s: int = 12 * 3600) -> bool:
    """Execute the hardened training notebook headlessly (produces the Phase-34 export). Returns True if a bundle exists after."""
    if bundle_ready(bundle_dir):
        print(f"[train] bundle already present at {bundle_dir} -- skipping training")
        return True
    import nbformat
    from nbclient import NotebookClient
    print(f"[train] no bundle at {bundle_dir}; executing {training_notebook} (this takes hours on a GPU)")
    nb = nbformat.read(training_notebook, as_version=4)
    NotebookClient(nb, timeout=timeout_s, kernel_name="python3", allow_errors=False).execute()
    return bundle_ready(bundle_dir)


def ensure_pygnome_env(root: str = "/kaggle/working/micromamba_root", env_name: str = "pygnome") -> str:
    """Create an isolated conda env with PyGNOME via micromamba (same recipe as the original notebook). Returns its python path."""
    root_p = Path(root); root_p.mkdir(parents=True, exist_ok=True)
    mm = root_p / "micromamba"
    py = root_p / "envs" / env_name / "bin" / "python"
    if py.exists():
        return str(py)
    if not mm.exists():
        tar = root_p / "mm.tar.bz2"
        urllib.request.urlretrieve("https://micro.mamba.pm/api/micromamba/linux-64/latest", tar)
        with tarfile.open(tar) as t:
            t.extractall(root_p / "mm_extract")
        (root_p / "mm_extract" / "bin" / "micromamba").replace(mm); mm.chmod(0o755)
    env = root_p / "envs" / env_name
    subprocess.run([str(mm), "create", "-y", "-r", str(root_p), "-p", str(env), "-c", "conda-forge", "python=3.12", "pygnome"], check=True)
    subprocess.run([str(py), "-c", "import gnome; print('PyGNOME OK')"], check=True)
    return str(py)
