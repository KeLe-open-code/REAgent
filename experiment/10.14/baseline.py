"""First GraphSAGE AGB baseline. No final-holdout features/labels are extracted."""
from pathlib import Path
from datetime import datetime
from hashlib import sha256
import argparse
import csv
import importlib.metadata
import json
import os
import platform
import random
import sys
import time
import xml.etree.ElementTree as ET
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".cache/matplotlib"))
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import numpy as np
import openpyxl
from scipy.spatial.distance import cdist
import torch
from torch import nn
from torch_geometric.nn import SAGEConv

FEATURES = ([f"B{i}" for i in range(1, 8)]
            + ["RVI", "DVI", "NDVI", "mRVI/MSR", "SAVI", "MSAVI", "ARVI", "EVI", "TVI", "RGVI"]
            + [f"b{i}_{s}X{s}" for s in (5, 7, 9) for i in range(1, 57)])
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    rows = list(rows)
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(y, prediction):
    y, prediction = np.asarray(y, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    assert y.shape == prediction.shape and y.ndim == 1
    assert np.isfinite(y).all() and np.isfinite(prediction).all()
    residual = prediction - y
    mse = float(np.mean(residual ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    return {"n": len(y), "rmse": mse ** 0.5, "mae": float(np.mean(np.abs(residual))),
            "r2": 1 - float(np.sum(residual ** 2)) / sst if sst > 0 else None,
            "bias": float(residual.mean()), "rrmse_percent": 100 * mse ** 0.5 / float(y.mean())}


def numeric_rows(path, columns, allowed=None):
    """Read selected numeric columns and, after splitting, only allowed XML rows."""
    values = {}
    with ZipFile(path) as z, z.open("xl/worksheets/sheet1.xml") as stream:
        for _, element in ET.iterparse(stream, events=("end",)):
            if element.tag != NS + "row":
                continue
            row = int(element.attrib["r"])
            if row > 1 and (allowed is None or row in allowed):
                record = {}
                for cell in element:
                    col = "".join(c for c in cell.attrib["r"] if c.isalpha())
                    if col in columns:
                        value = cell.find(NS + "v")
                        record[columns[col]] = float(value.text) if value is not None else np.nan
                values[row] = record
            element.clear()
    return values


def spatial_split(coords, config):
    blocks = np.floor(coords / config["grid_width_native_coordinates"]).astype(np.int64)
    block_names = np.array([f"{a}_{b}" for a, b in blocks])
    groups = sorted(set(block_names.tolist()))
    ordered = np.random.default_rng(config["seed"]).permutation(groups)
    roles = np.full(len(coords), "train", dtype="U24")
    cursor = 0
    for role, fraction in [("holdout", config["holdout_fraction"]),
                           ("validation", config["validation_fraction"])]:
        target, count = int(np.ceil(len(coords) * fraction)), 0
        while count < target:
            group = ordered[cursor]
            selected = block_names == group
            roles[selected] = role
            count += int(selected.sum())
            cursor += 1
    assert (roles == "train").sum() > config["knn_k"]
    return roles, block_names


def fit_preprocessor(raw, y):
    medians = np.array([np.median(col[np.isfinite(col)]) if np.isfinite(col).any() else 0.0 for col in raw.T])
    filled = np.where(np.isfinite(raw), raw, medians)
    means, scales = filled.mean(axis=0), filled.std(axis=0)
    scales[scales < 1e-12] = 1.0
    assert y.std() > 0
    return {"medians": medians, "means": means, "scales": scales,
            "y_mean": np.array(y.mean()), "y_scale": np.array(y.std())}


def transform(raw, pre):
    filled = np.where(np.isfinite(raw), raw, pre["medians"])
    return ((filled - pre["means"]) / pre["scales"]).astype(np.float32)


def graph_edges(train_x, query_x, k):
    distance = cdist(train_x.astype(float), train_x.astype(float))
    np.fill_diagonal(distance, np.inf)
    neighbors = np.argsort(distance, axis=1, kind="stable")[:, :k]
    source = neighbors.ravel()
    target = np.repeat(np.arange(len(train_x)), k)
    training = np.stack([source, target]).astype(np.int64)
    if len(query_x):
        d_query = cdist(query_x.astype(float), train_x.astype(float))
        near = np.argsort(d_query, axis=1, kind="stable")[:, :k]
        query_edges = np.stack([near.ravel(), np.repeat(np.arange(len(query_x)) + len(train_x), k)])
        return training, np.concatenate([training, query_edges], axis=1)
    return training, training.copy()


def prepare(region, config, out):
    source, = list((ROOT / "dateset").glob(f"*{region}*.xlsx"))
    assert digest(source) == config["source_sha256"][region], "Unexpected raw-data version"
    wb = openpyxl.load_workbook(source, read_only=True, data_only=True)
    assert wb.sheetnames == ["Sheet1"]
    headers = list(next(wb.active.iter_rows(max_row=1, values_only=True)))
    wb.close()
    rename = {f"b{i}_BAND": f"B{i}" for i in range(1, 8)}
    rename.update({"x": "longitude", "y": "latitude", "heng": "longitude", "zong": "latitude"})
    column_map = {openpyxl.utils.get_column_letter(i + 1): rename.get(h, h) for i, h in enumerate(headers)}
    coord_columns = {c: name for c, name in column_map.items() if name in ("longitude", "latitude")}
    records = numeric_rows(source, coord_columns)
    excel_rows = sorted(records)
    coords = np.array([[records[r]["longitude"], records[r]["latitude"]] for r in excel_rows])
    assert np.isfinite(coords).all() and len(np.unique(coords, axis=0)) == len(coords)
    roles, blocks = spatial_split(coords, config)
    out.mkdir(parents=True, exist_ok=False)
    manifest = [{"sample_id": f"{region}:row{r}", "excel_row": r, "role": str(role), "block": str(block)}
                for r, role, block in zip(excel_rows, roles, blocks)]
    write_csv(out / "split.csv", manifest)
    order = np.r_[np.flatnonzero(roles == "train"), np.flatnonzero(roles == "validation")]
    n_train = int((roles == "train").sum())
    allowed = {excel_rows[i] for i in order}
    fields = {c: name for c, name in column_map.items() if name in FEATURES + ["AGB"]}
    rows = numeric_rows(source, fields, allowed)
    assert set(rows) == allowed
    raw = np.array([[rows[excel_rows[i]].get(f, np.nan) for f in FEATURES] for i in order])
    raw[raw == -9999] = np.nan
    y = np.array([rows[excel_rows[i]]["AGB"] for i in order])
    assert np.isfinite(y).all() and (y > 0).all()
    pre = fit_preprocessor(raw[:n_train], y[:n_train])
    x = transform(raw, pre)
    assert np.isfinite(x).all()
    train_edges, eval_edges = graph_edges(x[:n_train], x[n_train:], config["knn_k"])
    np.savez_compressed(out / "preprocessor.npz", **pre)
    np.savez_compressed(out / "development.npz", raw_features=raw, x=x, y=y,
                        coords=coords[order], blocks=blocks[order],
                        sample_ids=np.array([manifest[i]["sample_id"] for i in order]),
                        n_train=n_train, train_edges=train_edges, eval_edges=eval_edges,
                        missing_mask=~np.isfinite(raw), feature_names=np.array(FEATURES))
    report = {"source": str(source.relative_to(ROOT)), "source_sha256": digest(source),
              "counts": {r: int((roles == r).sum()) for r in ("train", "validation", "holdout")},
              "blocks": {r: len(set(blocks[roles == r])) for r in ("train", "validation", "holdout")},
              "label_source": "UNKNOWN", "label_unit": "original table units; unconfirmed",
              "crs": "UNKNOWN; grid width in native coordinate units", "holdout_values_extracted": False,
              "features": len(FEATURES), "missing_development_cells": int((~np.isfinite(raw)).sum()),
              "all_missing_training_features": [FEATURES[i] for i in np.flatnonzero(~np.isfinite(raw[:n_train]).any(axis=0))],
              "split_sha256": digest(out / "split.csv"), "preprocessor_fit_ids": [manifest[i]["sample_id"] for i in order[:n_train]],
              "edges_train": train_edges.shape[1], "edges_evaluation": eval_edges.shape[1],
              "feature_access": "train-only graph fit; one-way training-to-query evaluation edges"}
    write_json(out / "data_manifest.json", report)
    return report


class Regressor(nn.Module):
    def __init__(self, name, config, d=185):
        super().__init__()
        self.name = name
        h1, h2 = config["hidden_dims"]
        if name == "graphsage":
            self.first, self.second = SAGEConv(d, h1, aggr="mean"), SAGEConv(h1, h2, aggr="mean")
        else:
            assert name == "mlp"
            self.first, self.second = nn.Linear(d, h1), nn.Linear(h1, h2)
        self.dropout = nn.Dropout(config["dropout"])
        self.head = nn.Linear(h2, 1)

    def forward(self, x, edges):
        h = self.first(x, edges) if self.name == "graphsage" else self.first(x)
        h = self.dropout(torch.relu(h))
        h = self.second(h, edges) if self.name == "graphsage" else self.second(h)
        embedding = torch.relu(h)
        return self.head(embedding).squeeze(-1), embedding


def fix_seed(seed, threads):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def tensors(data):
    return torch.from_numpy(data["x"]), torch.from_numpy(data["train_edges"]), torch.from_numpy(data["eval_edges"])


def train_one(name, config, data_dir, output):
    fix_seed(config["seed"], config["threads"])
    data, pre = np.load(data_dir / "development.npz"), np.load(data_dir / "preprocessor.npz")
    x, train_edges, eval_edges = tensors(data)
    n = int(data["n_train"])
    y = data["y"]
    target = torch.tensor((y[:n] - pre["y_mean"]) / pre["y_scale"], dtype=torch.float32)
    model = Regressor(name, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    output.mkdir(parents=True, exist_ok=False)
    best, best_epoch, stale, rows = float("inf"), 0, 0, []
    start = time.perf_counter()
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction, _ = model(x[:n], train_edges)
        loss = nn.functional.mse_loss(prediction, target)
        assert torch.isfinite(loss), "Nonfinite loss"
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
        assert torch.isfinite(grad_norm)
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            full_prediction, _ = model(x, eval_edges)
        pred = full_prediction.numpy().astype(np.float64) * pre["y_scale"] + pre["y_mean"]
        mt, mv = metrics(y[:n], pred[:n]), metrics(y[n:], pred[n:])
        rows.append({"epoch": epoch, "train_loss_standardized": float(loss.detach()),
                     "train_rmse": mt["rmse"], "validation_rmse": mv["rmse"],
                     "validation_r2": mv["r2"], "gradient_norm": float(grad_norm)})
        if mv["rmse"] < best - config["min_delta"]:
            best, best_epoch, stale = mv["rmse"], epoch, 0
            torch.save({"model_name": name, "state_dict": model.state_dict(), "epoch": epoch,
                        "config": config, "feature_names": FEATURES,
                        "split_sha256": digest(data_dir / "split.csv")}, output / "best.pt")
        else:
            stale += 1
        if epoch == 1 or epoch % 50 == 0:
            print(f"{data_dir.name}/{name} epoch={epoch} train_RMSE={mt['rmse']:.4f} val_RMSE={mv['rmse']:.4f} val_R2={mv['r2']:.4f}", flush=True)
            write_csv(output / "history.csv", rows)
        if stale >= config["patience"]:
            break
    elapsed = time.perf_counter() - start
    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.inference_mode():
        scaled, embedding = model(x, eval_edges)
        train_alone, _ = model(x[:n], train_edges)
        no_edges, _ = model(x, torch.empty((2, 0), dtype=torch.long))
    pred = scaled.numpy().astype(np.float64) * pre["y_scale"] + pre["y_mean"]
    no_edge_pred = no_edges.numpy().astype(np.float64) * pre["y_scale"] + pre["y_mean"]
    assert np.allclose(scaled[:n].numpy(), train_alone.numpy(), atol=1e-6, rtol=1e-6)
    write_csv(output / "history.csv", rows)
    write_csv(output / "predictions.csv", [{"sample_id": str(sid), "role": "train" if i < n else "validation",
              "observed_agb": float(y[i]), "predicted_agb": float(pred[i]), "residual": float(pred[i] - y[i])}
              for i, sid in enumerate(data["sample_ids"])])
    np.savez_compressed(output / "responses.npz", prediction=pred, embedding=embedding.numpy(),
                        absolute_residual=np.abs(pred - y), sample_ids=data["sample_ids"],
                        no_edge_prediction=no_edge_pred)
    result = {"train": metrics(y[:n], pred[:n]), "validation": metrics(y[n:], pred[n:]),
              "no_edge_validation": metrics(y[n:], no_edge_pred[n:]),
              "best_epoch": best_epoch, "epochs_run": epoch, "elapsed_seconds": elapsed,
              "parameters": sum(p.numel() for p in model.parameters()),
              "max_scaled_prediction_change_without_edges": float(torch.max(torch.abs(scaled - no_edges))),
              "max_train_query_addition_difference": float(torch.max(torch.abs(scaled[:n] - train_alone))),
              "checkpoint_sha256": digest(output / "best.pt"), "predictions_sha256": digest(output / "predictions.csv")}
    write_json(output / "metrics.json", result)
    print(f"DONE {data_dir.name}/{name}: {json.dumps(result['validation'])}", flush=True)
    return result


def environment():
    packages = ("torch", "torch-geometric", "numpy", "scipy", "openpyxl", "matplotlib")
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "packages": {p: importlib.metadata.version(p) for p in packages},
            "torch_cuda_build": torch.version.cuda, "training_device": "cpu", "gpu_used": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "train"])
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    run = HERE / args.run
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    if args.action == "prepare":
        assert not (run / "config.json").exists(), "Run already prepared"
        run.mkdir(parents=True, exist_ok=True)
        write_json(run / "config.json", config)
        write_json(run / "environment.json", environment())
        for region in config["regions"]:
            result = prepare(region, config, run / "data" / region)
            print(json.dumps({"region": region, **result}, ensure_ascii=True), flush=True)
        write_json(run / "prepared.json", {"time": datetime.now().astimezone().isoformat(),
                   "config_sha256": digest(HERE / "config.json"), "code_sha256": digest(__file__)})
    else:
        assert digest(HERE / "config.json") == json.loads((run / "prepared.json").read_text())["config_sha256"]
        summary = {}
        for region in config["regions"]:
            directory = run / "data" / region
            data = np.load(directory / "development.npz")
            n = int(data["n_train"])
            summary[region] = {"mean_predictor": metrics(data["y"][n:], np.repeat(data["y"][:n].mean(), len(data["y"]) - n))}
            for name in config["models"]:
                summary[region][name] = train_one(name, config, directory, run / region / name)
                write_json(run / "summary.json", summary)
        write_json(run / "trained.json", {"finished_at": datetime.now().astimezone().isoformat(),
                   "code_sha256": digest(__file__), "config_sha256": digest(HERE / "config.json")})


if __name__ == "__main__":
    main()
