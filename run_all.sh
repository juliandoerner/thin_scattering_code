#!/usr/bin/env bash

set -euo pipefail

NP="${1:-8}"
RUN_HIGH_P="${RUN_HIGH_P:-0}"
cd "$(dirname "$0")"

if _cm=$(command -v dolfinx-complex-mode); then
    . "$_cm"
fi

mpirun --oversubscribe -n "${NP}" python run_resolution.py -p 1 -o output \
    -k 6 7 8 9 10 11 12 13 14   --thresh 1.2 --kh_min 0.25 --kh_max 3.0 --n_kh 20 \
    -L 2.0 --max_cells 500000 --order 1
mpirun --oversubscribe -n "${NP}" python run_resolution.py -p 2 -o output \
    -k 10 12 14 16 18 20 22 24 26     --thresh 1.2 --kh_min 0.5 --kh_max 3.0 --n_kh 20 \
    -L 2.0 --max_cells 5000000 --order 2


# Large-memory reguired
if [ "${RUN_HIGH_P}" = "1" ]; then
    mpirun --oversubscribe -n "${NP}" python run_resolution.py -p 3 -o output \
    -k 60 65 70 75 80 85 90 95 100 105 110    --thresh 1.2 --kh_min 1.0 --kh_max 3.5 --n_kh 20 \
    -L 2.5 --max_cells 8000000 --order 3
    
fi

python make_figures.py -o output

echo
echo "Done. Per-degree artifacts in ./output :"
echo "  critical_p{p}.csv  sweep_p{p}.csv  resolution_p{p}.png"
