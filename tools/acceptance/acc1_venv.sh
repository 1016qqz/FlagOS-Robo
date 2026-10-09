#!/usr/bin/env bash
# Acceptance group 1+2+3: a FRESH venv, requirements/cuda/base.txt, pip install -e .
# Consumer-equivalent env check -- must NOT inherit the conda env's torch, so no
# --system-site-packages.
#
# IMPORTANT: every runtime artifact (venv, log) lives OUTSIDE the rsync'd repo
# tree. sync.sh uses `rsync --delete`, which deletes remote-only files; a previous
# run put the venv+log in _scratch/ and the next sync wiped them mid-install.
set -uo pipefail
cd /workspace/FlagOS-Robo

LOG=/tmp/acc1.log
VENV=/tmp/acc1_venv
: >"$LOG"
exec >>"$LOG" 2>&1

echo "=== acc1 start $(date -u +%FT%TZ)"
rm -rf "$VENV"
python -m venv "$VENV" || { echo "FATAL: venv create failed"; exit 1; }
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -V
pip -V
pip install --no-cache-dir -q -U pip setuptools wheel
echo "PIP_UPGRADED=$?"

echo "=== acceptance 1: pip install -r requirements/cuda/base.txt"
pip install --no-cache-dir -r requirements/cuda/base.txt
echo "RC_REQUIREMENTS=$?"

echo "=== acceptance 1b: pip install -e ."
pip install --no-cache-dir -e .
echo "RC_EDITABLE=$?"

echo "=== acceptance 2: import + version"
python -c "import flagos_robo; print('VERSION=' + flagos_robo.__version__)"
echo "RC_IMPORT=$?"

echo "=== acceptance 2b: platform in a venv-only env"
python -c "
from flagos_robo.platforms import get_platform
p = get_platform()
print('PLATFORM=%s %s %s %s' % (p.name(), p.dist_backend(), p.amp_device_type(), p.current_device()))
"
echo "RC_PLATFORM=$?"

echo "=== acceptance 3: purity gate inside the venv"
python tools/check_no_megatron.py --check-env
echo "RC_PURITY=$?"
pip list 2>/dev/null | grep -i megatron || echo "OK: no megatron in venv pip list"

echo "=== torch actually installed"
python -c "import torch; print('TORCH=' + torch.__version__, 'cuda=' + str(torch.cuda.is_available()), 'n=' + str(torch.cuda.device_count()))"
echo "RC_TORCH=$?"

echo "=== acc1 done $(date -u +%FT%TZ)"
