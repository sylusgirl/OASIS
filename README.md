# OASIS · Online search-plan selection for spatial queries

Research code and measured results for *OASIS: An Efficient and Training-Free Search-Plan Selector for Spatial Queries on Resource-Constrained Devices*.

OASIS chooses a search plan for each exact k-nearest-neighbor, radius or axis-aligned range query as it arrives. The choice matters even within a single index. On one million GeoNames points, a depth-first KD-tree search averaged 0.4 µs for the nearest neighbor and a best-first search averaged 0.6 µs. Asking for 1,000 neighbors reversed the ranking. Depth-first took 84.3 µs and best-first took 40.4 µs. OASIS learns which plans work from searches it has already seen.

中文简介：OASIS 根据已经执行过的查询，为下一次近邻、半径或矩形范围查询选择搜索方案。仓库包含七种精确搜索方案、在线选择器、对比实验和可直接重绘论文图表的实测数据。

## How OASIS makes a decision

1. **Group similar queries.** For each query type, the selector groups queries by expected answer size, query shape and local point density. It keeps timing statistics for the class and observation counts for the surrounding grid cell.
2. **Choose a plan.** Each class maintains a short candidate list. The selector usually chooses the best-known plan and occasionally explores another candidate. The experiment compares seven exact plans across KD-trees, a uniform grid and an R-tree.
3. **Limit the cost of exploration.** A trial that exceeds its time budget gives way to the best-known plan. Future choices use the accumulated experience. OASIS needs no offline training labels. The reported experiments choose a shared set of hyperparameters using separate validation streams.

The benchmark times each plan three times per query, checks agreement between plans, and keeps the minimum time. The replay evaluates each selector on the same traces and accounts for stopped trials and fallback searches. Separate C++ benchmarks measure the selector overhead. The workloads include mixed queries, five spatial distributions, insertions and distribution drift.

## What the recorded experiments show

Each row summarizes 20,000 mixed queries on one million points. Times are means in microseconds per query. The selection column is already included in the OASIS total. Oracle is a hindsight reference that takes the fastest measured plan for each query at zero selection cost.

| Dataset | Dimensions | OASIS total (µs) | Selection (µs) | Oracle (µs) |
|---|---:|---:|---:|---:|
| Uber | 2 | 7.637 | 0.238 | 6.731 |
| GeoNames | 2 | 7.028 | 0.237 | 6.118 |
| Porto | 2 | 6.750 | 0.243 | 5.620 |
| Dragon | 3 | 8.949 | 0.245 | 8.133 |

Two workloads illustrate the effect of changing query locations. On the GeoNames Zipf stream, OASIS averaged 6.222 µs against 12.711 µs for Frozen, the fastest competing selector on that stream (2.04×). On the Porto drift stream, OASIS averaged 11.969 µs against 131.818 µs for the offline AI+R-tree selector (11.01×). These are ratios of whole-stream means measured on a Xeon server. [GeoNames results](results_paper/GeoNames/summary.csv) · [Porto results](results_paper/Porto/summary.csv).

## Redraw the paper figures

From the repository root, run the following commands in Bash. This uses the recorded results in [results_paper/](results_paper/) and needs only Python and Matplotlib.

```bash
python -m pip install matplotlib
FIGMODE=mdpi python src/plot_paper3.py results_paper figures
python src/plot_intro.py results_paper figures
```

The scripts write nine figures, each as a PDF and a PNG, into `figures/`. In PowerShell, set `$env:FIGMODE = "mdpi"` on its own line, then run the two Python commands without the `FIGMODE=mdpi` prefix.

## Reproduce the measurements

Use Linux, Bash, a GCC toolchain with C++17 support, and CPython 3.11 or 3.12. The parallel replay uses the Linux `fork` start method to share preloaded traces. Set up a virtual environment from the repository root.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Download the following raw inputs and place them in `raw/`. The repository includes the processed results. A full measurement run needs the raw points for all four datasets, including at least two million usable points per dataset for the largest scalability run.

| Dataset | Source | Files in `raw/` |
|---|---|---|
| Uber | [FiveThirtyEight NYC pickups](https://github.com/fivethirtyeight/uber-tlc-foil-response) | `uber-apr.csv`, `uber-may.csv`, `uber-jun.csv`, `uber-jul.csv`, `uber-aug.csv`, `uber-sep.csv` (the six 2014 files, renamed) |
| GeoNames | [GeoNames](https://www.geonames.org) | `US.zip` (contains `US.txt`) |
| Porto | [Porto taxi trajectories](https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i) | `porto.zip` (contains `train.csv` or `train.csv.zip`) |
| Dragon | [Stanford 3D Scanning Repository](https://graphics.stanford.edu/data/3Dscanrep/) | `dragon.ply.gz` (XYZ RGB Dragon, binary PLY) |

Run the pipeline from the repository root.

```bash
bash run_all.sh
```

The runner defaults to eight worker processes for replay. Set `OASIS_PROCS` to change the worker count. Intermediate traces are written to `work8/`, benchmark logs to `logs/`, combined tables to `results_merged/`, and plots to `figures/`. If preprocessing reports `FAILED`, check the listed raw filenames and confirm that `data/` contains a binary for each dataset.

1. **Build.** `bench.cpp`, `selbench.cpp`, and `selbench2.cpp` are compiled with `g++ -O3 -march=native -std=c++17`.
2. **Data.** `prep.py` converts the raw files; `gen.py` samples 1M points per dataset (and 0.25M, 0.5M, and 2M points
   for the scalability study) and writes the streams: `mixed`, the five distribution streams, `insert`, `drift`, a
   validation stream, and the training streams of the offline selectors.
3. **Search times.** `bench7` (built from `bench.cpp`) runs every query with all seven plans, records the minimum of
   three runs per plan, and checks that all plans return the same answer.
4. **Compared selectors.** `replay.py` trains the offline selectors, replays every selector on the measured search
   times, and measures their selection times with `selbench`. It also replays an earlier variant of OASIS, whose
   parameters `choose.py` picks; step 6 replaces these rows by the results of `oasis2.py`, which is the OASIS of
   the paper.
5. **OASIS.** `replay2.py --tune` and `choose2.py` choose the parameters on the validation streams;
   `replay2.py --cost` measures the selection time with `selbench2`; `replay2.py` replays OASIS on all streams,
   including the ablation and the parameter study; `replay2.py --scal` runs the scalability study.
6. **Merge and figures.** `merge2.py` writes `results_merged/`; `plot_paper3.py` and `plot_intro.py` draw the figures.

Step 3 is by far the longest: it runs every query of every stream 21 times. We ran it as one batch job per dataset.
All measurements in the paper were taken on one thread of a server with two Intel Xeon Gold 6438Y+ CPUs; absolute times
on other machines differ.

## Code map

```
README.md
requirements.txt
run_all.sh            runs the whole pipeline
raw/                  put the downloaded datasets here
results_paper/        measured results used in the paper, one folder per dataset
src/
  prep.py             raw files -> data/<dataset>.bin (insertion order, scaled into the unit box)
  gen.py              samples the points and generates all query streams
  bench.cpp           the seven search plans; measures the search time of every plan on every query
  export_est.py       expected number of answers of every query (from cell counts and prefix sums)
  replay_core.py      shared code: loading, features, the compared selectors
  replay.py           replays the compared selectors on the measured times
  choose.py           picks the parameters of the earlier OASIS variant that replay.py still replays (see step 4)
  selbench.cpp        measures the selection time of the compared selectors
  oasis2.py           OASIS: two-level feature extraction and time-limited trials
  replay2.py          replays OASIS; ablation, parameter study, scalability
  choose2.py          picks the parameters of OASIS on the validation streams
  selbench2.cpp       measures the selection time of OASIS
  merge2.py           merges the results of OASIS and of the compared selectors
  plot_paper.py, plot_paper2.py, plot_paper3.py   figures of Section 5 (plot_paper3.py is the entry point)
  plot_intro.py       Figure 1
  why.py              mean search time of each plan per query type and size (the numbers of Figure 1a)
```

## Results and figures

| Paper | Source in `results_paper/` or `src/` | Figure file |
|---|---|---|
| Figure 1 | Panel (b) uses `GeoNames/stream.csv`. Panel (a) embeds the original `why.py` measurements in `plot_intro.py` | `intro_problem.pdf` |
| Query time on the mixed stream | `total_time.csv` | `exp_total.pdf` |
| Selector models | `selectors.csv` | `exp_selectors.pdf` |
| Query distributions | `distributions.csv` | `exp_distributions.pdf` |
| Insertions and query drift | `stream.csv` | `exp_stream.pdf` |
| Breakdown of the query time | `overhead.csv` | `exp_overhead.pdf` |
| Labeling time, training time, and memory (table) | `memory.csv` | - |
| Scalability | `scalability.csv` | `exp_scalability.pdf` |
| Ablation | `ablation2.csv` | `exp_ablation.pdf` |
| Parameter study | `params2.csv` | `exp_params.pdf` |

In `summary.csv`, `total_us` gives the mean time per query for each method and stream. `accuracy_pct` is the percentage of queries for which the initially selected plan runs within 5% of the fastest measured plan. The benchmark checks agreement on query answers separately. `BestFixed` chooses a single plan for each query type in hindsight across the stream.

`tune2.csv` records the validation sweep, `cost2.json` and `cost_us.json` hold measured selection times, and `info2.csv` counts OASIS classes and trials. The archive also includes `UnIS-R`, a periodically retrained variant that the paper's plots omit.

## Paper notation and implementation settings

| Paper | Code |
|---|---|
| Plans KD-DFS (A), KD-BF (B), KD-BFS (C), Grid (D) | `KD-DFS-MBR`, `KD-BF`, `KD-BFS-MBR`, `Grid` |
| Plans KD-DFS-Ball, KD-BFS-Ball, R-BF | `KD-DFS-Ball`, `KD-BFS-Ball`, `R-BF` |
| Class threshold tau = 1 | `tau = 1.0` |
| Candidates per class h = 2 | `r = 2` |
| Initial temperature theta_0 = 0.1 | `T0 = 0.1` |
| Time-limit factor b = 3 | `b = 3.0` |
| Warming factor beta | `gamma = 1.0` |
| Weight N_0 of the class experience | `N0` |
| Without two-level features / without time-limited trials / without both | `w/o Features` / `w/o Trials` / `Base` |

The parameter values used in the paper are in `results_paper/oasis2_config.json`; `choose2.py` selected them among
the settings listed in `tune2.csv`. Values that are not in this file keep the defaults in `oasis2.py`.

Three settings of the implementation are fixed in the code and are not varied in the paper:

* `N0 = 1e9` in the selected configuration: the combined mean of a plan then equals its class mean, and the cell
  experience enters only through the number of observations that sets the temperature. The validation results of
  the finite values `N0 = 8` and `N0 = 32` are in `tune2.csv`.
* `cf = 4`: the cell experience is kept on a grid that is four times coarser per dimension than the grid of the
  cell counts (16 x 16 cells in 2D, 6 x 6 x 6 in 3D).
* `K = 32`: at most 32 classes are kept per query type.
