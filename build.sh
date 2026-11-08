#!/bin/bash
# Build paper.tex -> paper.pdf using tectonic.
set -e
cd "$(dirname "$0")"

if ! command -v tectonic >/dev/null 2>&1; then
  echo "==> tectonic not found. Install:"
  echo "    macOS:  brew install tectonic"
  echo "    Linux:  curl --proto '=https' --tlsv1.2 -fsSL https://drop-sh.fullyjustified.net | sh"
  echo "    or:     cargo install tectonic"
  exit 1
fi

# Ensure all figures referenced exist
MISSING=0
for f in plots/fig1_strikes.png plots/fig4_euler_scaling.png \
         plots/fig12_recovery_correct.png plots/fig13_stab_structure.png \
         plots/fig14_multiview_stab.png plots/fig7_mlp_gauge_gap.png; do
  if [ ! -f "$f" ]; then
    echo "MISSING: $f"
    MISSING=1
  fi
done
if [ "$MISSING" -eq 1 ]; then
  echo "Run the corresponding scripts in src/ first, then retry."
  exit 1
fi

echo "==> tectonic build (first run downloads packages; may take minutes)"
tectonic --keep-logs paper.tex

echo ""
echo "==> done:"
ls -la paper.pdf paper.log 2>/dev/null || true
