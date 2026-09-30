"""AIS trajectory prediction (CV / MLP / RNN / MLP+RNN) and behavioural anomaly scoring.

Ported from the attribution notebook with these changes:
  * training sequences are capped (stage1_max_sequences) so real AIS volumes stay tractable
  * the model set and RNN epochs are configurable
  * results are per (MMSI, segment); vessel-level aggregation happens in attribution.py
"""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd

from .ais import FEATURE_COLS
from .config import AISConfig

log = logging.getLogger(__name__)


class _Adam:
    def __init__(self, params, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = [np.zeros_like(p) for p in params]; self.v = [np.zeros_like(p) for p in params]; self.t = 0

    def step(self, params, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g ** 2
            p -= self.lr * (self.m[i] / (1 - self.b1 ** self.t)) / (np.sqrt(self.v[i] / (1 - self.b2 ** self.t)) + self.eps)


class SimpleRNNRegressor:
    """Vanilla tanh RNN. head='linear' -> RNN; head='mlp' -> MLP+RNN hybrid. (hand-rolled NumPy, no torch needed)"""

    def __init__(self, input_dim, output_dim=2, hidden_dim=16, head="linear", head_hidden=16, lr=0.01,
                 epochs=25, batch_size=128, l2=1e-4, seed=0):
        rng = np.random.default_rng(seed)
        self.hidden_dim, self.head, self.epochs, self.batch_size, self.l2 = hidden_dim, head, epochs, batch_size, l2
        s = 1.0 / np.sqrt(hidden_dim)
        self.Wxh = rng.uniform(-s, s, (input_dim, hidden_dim)); self.Whh = rng.uniform(-s, s, (hidden_dim, hidden_dim)); self.bh = np.zeros(hidden_dim)
        if head == "linear":
            self.Wy = rng.uniform(-s, s, (hidden_dim, output_dim)); self.by = np.zeros(output_dim)
            self._pn = ["Wxh", "Whh", "bh", "Wy", "by"]
        else:
            s2 = 1.0 / np.sqrt(head_hidden)
            self.W1 = rng.uniform(-s2, s2, (hidden_dim, head_hidden)); self.b1 = np.zeros(head_hidden)
            self.W2 = rng.uniform(-s2, s2, (head_hidden, output_dim)); self.b2 = np.zeros(output_dim)
            self._pn = ["Wxh", "Whh", "bh", "W1", "b1", "W2", "b2"]
        self.params = [getattr(self, p) for p in self._pn]
        self.opt = _Adam(self.params, lr=lr)

    def _fwd(self, X):
        n, T, _ = X.shape
        H = np.zeros((n, self.hidden_dim)); hs = [H]
        for t in range(T):
            H = np.tanh(X[:, t] @ self.Wxh + H @ self.Whh + self.bh); hs.append(H)
        if self.head == "linear":
            return H @ self.Wy + self.by, (X, hs, None)
        z1 = H @ self.W1 + self.b1; a1 = np.maximum(z1, 0)
        return a1 @ self.W2 + self.b2, (X, hs, (z1, a1))

    def _bwd(self, cache, dout):
        X, hs, hc = cache; n, T, _ = X.shape; H = hs[-1]
        g = {p: np.zeros_like(getattr(self, p)) for p in self._pn}
        if self.head == "linear":
            g["Wy"] = H.T @ dout / n; g["by"] = dout.mean(0); dH = dout @ self.Wy.T
        else:
            z1, a1 = hc
            g["W2"] = a1.T @ dout / n; g["b2"] = dout.mean(0)
            dz1 = (dout @ self.W2.T) * (z1 > 0)
            g["W1"] = H.T @ dz1 / n; g["b1"] = dz1.mean(0); dH = dz1 @ self.W1.T
        for t in reversed(range(T)):
            dtanh = dH * (1 - hs[t + 1] ** 2)
            g["Wxh"] += X[:, t].T @ dtanh / n; g["Whh"] += hs[t].T @ dtanh / n; g["bh"] += dtanh.mean(0)
            dH = dtanh @ self.Whh.T
        for p in self._pn:
            g[p] += self.l2 * getattr(self, p)
        return [g[p] for p in self._pn]

    def fit(self, X, Y, seed=0):
        rng = np.random.default_rng(seed); n = len(X)
        for _ in range(self.epochs):
            idx = rng.permutation(n)
            for i in range(0, n, self.batch_size):
                b = idx[i:i + self.batch_size]
                out, cache = self._fwd(X[b])
                self.opt.step(self.params, self._bwd(cache, 2.0 * (out - Y[b]) / len(b)))
        return self

    def predict(self, X):
        return self._fwd(X)[0]


def build_sequences(df: pd.DataFrame, cfg: AISConfig):
    H = cfg.history_steps
    X, Y, meta = [], [], []
    for (mmsi, sid), g in df.groupby(["MMSI", "segment_id"]):
        g = g.sort_values("BaseDateTime").reset_index(drop=True)
        F = g[FEATURE_COLS].to_numpy(np.float32); e, n_ = g.east_m.to_numpy(), g.north_m.to_numpy(); T = len(g)
        for t in range(H, T - 1):
            true_d = np.array([e[t + 1] - e[t], n_[t + 1] - n_[t]]); cv = np.array([e[t] - e[t - 1], n_[t] - n_[t - 1]])
            X.append(F[t - H:t]); Y.append((true_d - cv).astype(np.float32)); meta.append((mmsi, sid, g.loc[t, "BaseDateTime"]))
    if not X:
        raise ValueError("Not enough AIS history to build training sequences.")
    return np.stack(X), np.stack(Y), meta


def vessel_group_kfold(mmsi_array, n_splits, seed=0):
    u = np.unique(mmsi_array); n_splits = max(2, min(n_splits, len(u)))
    rng = np.random.default_rng(seed); rng.shuffle(u)
    for fold in np.array_split(u, n_splits):
        vm = np.isin(mmsi_array, fold)
        yield ~vm, vm


def fit_stage1_oof(X, Y, meta, cfg: AISConfig, seed=11) -> Tuple[pd.DataFrame, dict]:
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(seed)
    if len(X) > cfg.stage1_max_sequences:
        sel = np.sort(rng.choice(len(X), cfg.stage1_max_sequences, replace=False))
        Xs, Ys, ms = X[sel], Y[sel], [meta[i] for i in sel]
    else:
        Xs, Ys, ms = X, Y, meta
    n, _, nf = Xs.shape; Xf = Xs.reshape(n, -1); mm = np.array([m[0] for m in ms])
    oof = {k: np.zeros_like(Ys) for k in cfg.stage1_models}
    for tr, va in vessel_group_kfold(mm, cfg.n_folds, seed):
        if tr.sum() < 20 or va.sum() == 0:
            continue
        xs, ys = StandardScaler(), StandardScaler()
        Xtr = xs.fit_transform(Xf[tr]); Xva = xs.transform(Xf[va]); Ytr = ys.fit_transform(Ys[tr])
        seq = lambda a: a.reshape(-1, cfg.history_steps, nf)
        if "MLP" in oof:
            m = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=400, early_stopping=True, n_iter_no_change=15, alpha=1e-3, random_state=seed).fit(Xtr, Ytr)
            oof["MLP"][va] = ys.inverse_transform(m.predict(Xva))
        if "RNN" in oof:
            r = SimpleRNNRegressor(nf, 2, head="linear", epochs=cfg.stage1_rnn_epochs, seed=seed).fit(seq(Xtr), Ytr, seed)
            oof["RNN"][va] = ys.inverse_transform(r.predict(seq(Xva)))
        if "MLP+RNN" in oof:
            r = SimpleRNNRegressor(nf, 2, head="mlp", epochs=cfg.stage1_rnn_epochs, seed=seed).fit(seq(Xtr), Ytr, seed)
            oof["MLP+RNN"][va] = ys.inverse_transform(r.predict(seq(Xva)))
    errs = {"CV": np.linalg.norm(Ys, axis=1)}
    errs.update({k: np.linalg.norm(Ys - v, axis=1) for k, v in oof.items()})
    means = {k: float(v.mean()) for k, v in errs.items()}
    best = min(means, key=means.get)
    log.info("[Stage 1] OOF mean residual (m): %s -> selected %s", {k: round(v, 2) for k, v in means.items()}, best)
    df = pd.DataFrame(ms, columns=["MMSI", "segment_id", "BaseDateTime"]).assign(pred_error_m=errs[best], oof_model=best)
    return df, dict(mean_error_m=means, selected=best)


def _robust_z_oof(values, mmsi_array, n_folds, seed=5):
    z = np.full(len(values), np.nan)
    for tr, va in vessel_group_kfold(mmsi_array, n_folds, seed):
        tv = values[tr]
        if len(tv) == 0:
            continue
        med = np.median(tv); mad = np.median(np.abs(tv - med))
        scale = mad * 1.4826 if mad > 1e-9 else (tv.std() or 1.0)
        z[va] = (values[va] - med) / scale
    miss = np.isnan(z)
    if miss.any():
        z[miss] = (values[miss] - np.median(values)) / (np.std(values) or 1.0)
    return z


def compute_anomaly_scores(proc: pd.DataFrame, pred_err: pd.DataFrame, cfg: AISConfig, behavior_weight=0.7) -> pd.DataFrame:
    df = proc.merge(pred_err, on=["MMSI", "segment_id", "BaseDateTime"], how="left")
    df = df.sort_values(["MMSI", "segment_id", "BaseDateTime"]).reset_index(drop=True)
    g = df.groupby(["MMSI", "segment_id"])
    df["speed_jump"] = g["SOG"].diff().abs().fillna(0)
    cd = g["COG"].diff().abs().fillna(0); df["course_jump"] = np.minimum(cd, 360 - cd)
    df["pred_error_m"] = df["pred_error_m"].fillna(0)
    mm = df["MMSI"].to_numpy(); nf = cfg.n_folds
    zg = _robust_z_oof(df.internal_gap_min.to_numpy(float), mm, nf); zs = _robust_z_oof(df.speed_jump.to_numpy(float), mm, nf)
    zc = _robust_z_oof(df.course_jump.to_numpy(float), mm, nf); zp = _robust_z_oof(df.pred_error_m.to_numpy(float), mm, nf)
    df["ais_quality_z"] = np.clip(zg, 0, None)
    df["behavior_z"] = np.clip(0.4 * zs + 0.3 * zc + 0.3 * zp, 0, None)
    df["final_point_score"] = behavior_weight * df.behavior_z + (1 - behavior_weight) * df.ais_quality_z
    return df


def aggregate_vessel_scores(scored: pd.DataFrame) -> pd.DataFrame:
    """Per-vessel anomaly confidence in [0,1] = sigmoid of the worse of (median, 95th pct) point score."""
    def _agg(g):
        s = g["final_point_score"]
        raw = max(float(np.median(s)), float(np.percentile(s, 95)))
        return pd.Series(dict(anomaly_raw=raw, anomaly_confidence=float(1.0 / (1.0 + np.exp(-(raw - 3.0)))), n_points=len(g)))
    return scored.groupby("MMSI").apply(_agg).reset_index()
