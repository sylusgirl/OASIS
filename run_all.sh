#!/usr/bin/env bash
# Runs every experiment of the paper. Usage: bash run_all.sh   (Linux, GCC, Python 3; datasets in raw/, see README.md)
set -euo pipefail
P="$(cd "$(dirname "$0")" && pwd)"
cd "$P/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OASIS_WORK=work8 OASIS_PROCS="${OASIS_PROCS:-8}"
mkdir -p "$P/logs"
DATASETS="Uber Dragon GeoNames Porto"

# 1. build
g++ -O3 -march=native -std=c++17 -o bench7 bench.cpp
g++ -O3 -march=native -std=c++17 -o selbench selbench.cpp
g++ -O3 -march=native -std=c++17 -o selbench2 selbench2.cpp

# 2. datasets and streams
python prep.py "$P"
for DS in $DATASETS; do
  python gen.py "$P" "$DS" 1000000
  for n in 250000 500000 2000000; do python gen.py "$P" "$DS" "$n" --scal; done
done

# 3. search time of every plan on every query (minimum of three runs), and the expected numbers of answers
for DS in $DATASETS; do
  W=$P/work8/${DS}_1000000
  for s in train_full train_half valid mixed insert drift dist_uniform dist_data dist_hotspot dist_zipf dist_trajectory; do
    ./bench7 "$W/points.bin" "$W/$s.ev" "$W/$s.csv" 3 2>> "$P/logs/bench_${DS}.log"
  done
  for n in 250000 500000 2000000; do
    Wn=$P/work8/${DS}_$n
    ./bench7 "$Wn/points.bin" "$Wn/mixed.ev" "$Wn/mixed.csv" 3 2>> "$P/logs/bench_${DS}.log"
  done
  for n in 1000000 250000 500000 2000000; do python export_est.py "$P" "$DS" "$n"; done
done

# 4. compared selectors -> results/
for DS in $DATASETS; do python replay.py "$P" "$DS" 1000000 --tune; done
python choose.py "$P"
for DS in $DATASETS; do
  python replay.py "$P" "$DS" 1000000
  python replay.py "$P" "$DS" 1000000 --scal
done

# 5. OASIS -> results2/
for DS in $DATASETS; do python replay2.py "$P" "$DS" 1000000 --tune; done
python choose2.py "$P"
for DS in $DATASETS; do python replay2.py "$P" "$DS" 1000000 --cost; done   # selection time, one dataset at a time
for DS in $DATASETS; do
  python replay2.py "$P" "$DS" 1000000
  python replay2.py "$P" "$DS" 1000000 --scal
done

# 6. merge -> results_merged/, figures -> figures/
python merge2.py "$P"
FIGMODE=mdpi python plot_paper3.py "$P/results_merged" "$P/figures"
python plot_intro.py "$P/results_merged" "$P/figures"
echo ALL_DONE
