# GNU Make + YAML dependency demo

Suggested repository layout:

    Makefile
    deps.yaml
    scripts/fetch_deps.py
    src/
    include/
    3rdparty/       # generated, ignored

Setup:

    make setup
    make doctor
    make deps
    make

Requires Python 3.12+ and GNU Make. `make setup` creates a local `.venv` and
installs PyYAML there. The other targets use that environment directly; no
activation or Conda installation is needed. On Windows, setup uses `py`;
on Linux, it uses `python3`. Override `BOOTSTRAP_PYTHON` to select another
interpreter, for example `make setup BOOTSTRAP_PYTHON="C:/Python312/python.exe"`.

The fetcher supports ZIP and TAR archives, not `.7z`. Manifest checksums must
be bare SHA-256 hex strings, without a `sha256:` prefix. The current
`bed-reader-cpp` entry needs a supported release archive and its checksum
before `make deps` can succeed.

Replace each demo GitHub Release URL and `REPLACE_WITH_SHA256` in `deps.yaml` with a real release asset and checksum.

Design: `deps.yaml` is the lock-style manifest; `fetch_deps.py` selects the OS/architecture artifact, downloads it, verifies SHA-256, safely extracts it under `3rdparty/<name>/<version>/`, and writes `.ready`. Re-running is idempotent. `make` is the developer-facing orchestrator.
