#!/usr/bin/env python3
import argparse, hashlib, os, platform, shutil, sys, tarfile, tempfile, urllib.request, zipfile
from pathlib import Path
try:
    import yaml
except ImportError:
    sys.exit("Missing PyYAML. Install it with: python3 -m pip install pyyaml")

def platform_key():
    s = platform.system().lower()
    osname = "windows" if s == "windows" else "linux" if s == "linux" else s
    m = platform.machine().lower()
    arch = "x86_64" if m in ("x86_64", "amd64") else "arm64" if m in ("aarch64", "arm64") else m
    return f"{osname}-{arch}"

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def safe_extract(archive, dest):
    dest = dest.resolve()
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            for n in z.namelist():
                if not (dest / n).resolve().is_relative_to(dest): raise RuntimeError("unsafe zip path")
            z.extractall(dest)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as t:
            for member in t.getmembers():
                if not (dest / member.name).resolve().is_relative_to(dest): raise RuntimeError("unsafe tar path")
            t.extractall(dest, filter="data")
    else:
        raise RuntimeError(f"unsupported archive: {archive}")

def install(name, spec, root, pkey, force=False):
    version = str(spec["version"]); artifact = spec.get("platforms", {}).get(pkey)
    if not artifact: raise RuntimeError(f"{name}: no artifact for {pkey}")
    dest = root / name / version; ready = dest / ".ready"
    if ready.exists() and not force:
        print(f"[OK] {name} {version}"); return
    expected = artifact["sha256"].lower()
    if expected.startswith("replace_"): raise RuntimeError(f"{name}: replace demo SHA256 in deps.yaml")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        archive = Path(td) / "package"
        print(f"[GET] {name} {version} ({pkey})")
        urllib.request.urlretrieve(artifact["url"], archive)
        actual = sha256(archive)
        if actual != expected: raise RuntimeError(f"{name}: SHA256 mismatch\nexpected {expected}\nactual   {actual}")
        staging = Path(td) / "extract"; staging.mkdir()
        safe_extract(archive, staging)
        if dest.exists(): shutil.rmtree(dest)
        shutil.copytree(staging, dest)
        ready.write_text(f"{name} {version} {pkey}\n")
        print(f"[INSTALLED] {dest}")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--manifest", default="deps.yaml"); ap.add_argument("--root", default="3rdparty"); ap.add_argument("--platform"); ap.add_argument("--force", action="store_true"); args=ap.parse_args()
    data=yaml.safe_load(Path(args.manifest).read_text()); pkey=args.platform or platform_key(); root=Path(args.root)
    print(f"Platform: {pkey}")
    for name,spec in data.get("dependencies", {}).items(): install(name,spec,root,pkey,args.force)
if __name__ == "__main__": main()
