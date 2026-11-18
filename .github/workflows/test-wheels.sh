#!/usr/bin/env bash
# Install the wheel from dist/ that pip picks for each interpreter in
# $PYTHONS (plus GIL 3.15, which takes the abi3t wheel) and run the tests
# against it. Free-threaded interpreters must keep the GIL disabled.
set -euo pipefail
for py in $PYTHONS 3.15; do
  venv="$RUNNER_TEMP/venv-$py"
  uv venv -q --python "$py" "$venv"
  bin="$venv/bin"; [ -d "$bin" ] || bin="$venv/Scripts"
  uv pip install -q --python "$bin/python" numpy pytest
  uv pip install -q --python "$bin/python" --no-index --find-links dist paged-collections
  "$bin/python" - <<'EOF'
import sys, sysconfig
import paged_collections
ft = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
print(sys.version, "free-threaded" if ft else "GIL", paged_collections._core.__file__)
assert not ft or not sys._is_gil_enabled(), "importing paged_collections enabled the GIL"
EOF
  # From the repo root, `python -m pytest` imports the installed package:
  # the sources live under python/, which is not on sys.path.
  "$bin/python" -m pytest -q -p no:cacheprovider tests
done
