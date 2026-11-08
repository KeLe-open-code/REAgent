# REAgent

Code accompanying **REAgent: A sampling agent with relational learning and ecological calibration for forest aboveground biomass estimation from remote sensing**.

Authors: Le Ke, Xixin Chen, Guoxiong Zhou, Xueyan Zhu, and Guoqing Liu.

REAgent combines a GraphSAGE biomass predictor with learned reference relations (RP), an agent that edits real reference nodes and recomputes predictions, and ecological class residual calibration (ECRC). Query locations remain in the evaluation after reference reduction. Compression refers to real reference plots, not neural-network parameter pruning.

## Installation

The included CPU source check is verified locally with Python 3.11, PyTorch 2.11.0 with CUDA 12.6, and PyTorch Geometric 2.8.1. These local check versions are listed in `requirements.txt`; use the paper's recorded benchmark environment for numerical reproduction. Install a compatible PyTorch build for your hardware before the remaining dependencies. The historical Full training entry uses CUDA; the synthetic smoke check runs on CPU.

```bash
python -m pip install -r requirements.txt
python tests/smoke.py
python train_full.py --help
```

## Train the standard Full model

Run from this repository root:

```bash
python train_full.py --data-root /path/to/frozen-folds --dataset hunan --fold 1 --seed 0 --ecology /path/to/ecology.csv --output outputs/full-hunan-fold1-seed0
```

`--dataset` is `hunan`, `hubei`, or `guizhou`; folds are 1–3 and the recorded seeds are 0, 1000, and 2000. The output directory must not already exist. Input schemas are described in [docs/data_format.md](docs/data_format.md). Data are supplied by the user and are not bundled here.

## Implementations

| Component | Main implementation |
|---|---|
| Frozen Full/A7 assembly | `experiment/agb_node_reduction/historical_case.py` |
| RP training and predictor construction | `physical_baseline_v2.py`, `relation_precision.py`, `relation_agop.py`, `relation_local_egop.py` |
| Sequential agent and real reference export | `physical_factorial_v2.py`, `rp_stop_agent.py`, `rp_conservative_agent.py`, `rp_agent_reference.py` |
| Training-side out-of-fold ecological calibration | `evaluation/eco_calibration.py` and the OOF construction in `historical_case.py` |
| Public-dataset feature dimensions and prediction/compression interfaces | `evaluation/dynamic_rp.py`, `evaluation/dynamic_agent.py`, `evaluation/run_case.py`, `evaluation/run_compression_case.py` |
| Fixed reference-count deployment used for budget experiments | `historical_exact_k_relaxed_case.py`, `historical_exact_k_curve_case.py` |

Paths in the last five rows are relative to `experiment/agb_node_reduction/`. Standard A7 natural stopping and fixed-budget deployment are distinct entry points. The latter must not be described as identical to the unmodified natural-stopping policy. Historical module names and inherited helpers are preserved for source fidelity; their other diagnostic arms are not additional components of the active Full model.

## Public-dataset interfaces

The generic case interfaces accept local configuration and data files:

```bash
python experiment/agb_node_reduction/evaluation/run_case.py --help
python experiment/agb_node_reduction/evaluation/run_compression_case.py --help
python experiment/agb_node_reduction/evaluation/run_transfer_case.py --help
```

Use the matching data access, split, source-training preprocessing, ecological metadata, and reference budget. Independent training, zero-shot evaluation, and few-shot adaptation are separate protocols. Original public data are obtained from their authors' repositories; the HHG dataset is not redistributed. The code does not download private data or include measured features, sample coordinates, AGB labels, predictions, or trained weights.

## Provenance and citation

`source_manifest.json` records the relative paths and SHA-256 hashes of the frozen source. Copied scientific implementations are unchanged. `train_full.py`, documentation, and the synthetic smoke check provide operational entry points and do not replace the model equations. The repository contains no new benchmark claims; the smoke check is an implementation check using generated inputs, not a replication of the paper's experiments.

Citation metadata are in `CITATION.cff`. A paper DOI will be added when available. See [LICENSE](LICENSE) for the code license. Third-party libraries retain their respective licenses.
