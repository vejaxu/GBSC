# GBSC reproduction

This repository provides one entry point for the GBSC experiments on the
datasets in `../data/D-Spec`.

## Environment

All commands below use the `xwj_llm` conda environment:

```bash
conda run -n xwj_llm python main.py --help
```

## Experiment protocol

- MinMax-normalize every feature and project the data to two dimensions with
  PCA, following the released GBSC real-data code.
- Generate granular balls with the released weighted-density and radius rules.
- Search `delta = 2^p`, `p = -5, ..., 5`, once with seed 42.
- Select by NMI, using ARI and Hungarian-aligned macro-F1 as tie breakers.
- Re-run the selected delta with seeds `42, 3407, 4079, 2024, 0`.
- Report population mean and standard deviation for NMI, ARI, macro-F1, and
  end-to-end time (loading through predicted labels).

The affinity follows the released code exactly:

```text
exp((radius_i + radius_j - center_distance) / (2 * delta**2))
```

This differs from the squared/clipped Gaussian distance printed in the paper,
but it is the implementation that reproduces the published Pendigits values.

For at most 6000 granular balls, the graph is fully connected. Larger datasets
use a symmetric 30-nearest-neighbor graph because their dense affinity and
eigendecomposition are not computationally feasible. The selected backend is
recorded in every dataset summary and in `gbsc.csv`.

## Commands

Search parameters only:

```bash
conda run -n xwj_llm python main.py search --jobs 4
```

Reproduce five seeds from previously saved best parameters:

```bash
conda run -n xwj_llm python main.py reproduce --jobs 4
```

Search and reproduce in one command:

```bash
conda run -n xwj_llm python main.py all --jobs 4
```

Run a subset:

```bash
conda run -n xwj_llm python main.py all --datasets spiral pendigits --jobs 2
```

## Outputs

Each dataset is written under `results/<dataset>/`:

- `search.csv`: all 11 seed-42 search results;
- `best_params.json`: selected delta, graph backend, and granular-ball count;
- `best_labels_seed42.npy` / `.csv`: labels selected during seed-42 search;
- `plots/clustering_result.jpg` and `plots/true_labels.jpg`;
- `runs.csv`: the five fixed-seed runs;
- `summary.json`: mean and standard deviation.

The final aggregate, in the required dataset order, is `gbsc.csv`.
