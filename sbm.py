# -------------------------------------------------------------------------
# sbm_mininfo.py
# Minimal-information macro-structure via (weighted) Poisson SBM + MDL selection
# -------------------------------------------------------------------------

from __future__ import annotations
import math
import os
import pickle
import numpy as np
import networkx as nx
from typing import Dict, Tuple, List, Optional, Any
from tqdm import tqdm  # progreso visual

# ---- Numerics helpers ---------------------------------------------------------

try:
    from scipy.special import logsumexp as _logsumexp
except Exception:
    _logsumexp = None

def logsumexp(x: np.ndarray, axis: Optional[int]=None) -> np.ndarray:
    if _logsumexp is not None:
        return _logsumexp(x, axis=axis)
    m = np.max(x, axis=axis, keepdims=True)
    s = np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True)) + m
    return s if axis is None else np.squeeze(s, axis=axis)

# ---- Graph I/O ---------------------------------------------------------------

def graph_to_weight_matrix(G: nx.Graph, dtype=np.float64) -> np.ndarray:
    nodes = list(G.nodes())
    n = len(nodes)
    idx = {u: i for i, u in enumerate(nodes)}
    W = np.zeros((n, n), dtype=dtype)
    for u, v, d in G.edges(data=True):
        i, j = idx[u], idx[v]
        w = d.get("weight", 1.0)
        if i == j:
            continue
        W[i, j] += w
        W[j, i] += w
    return W

# ---- Poisson SBM -------------------------------------------------------------

class PoissonSBM:
    """Weighted Poisson SBM with mean w_ij ~ Poisson(Ω_{z_i z_j})."""

    def __init__(self, B: int, max_iter: int = 200, tol: float = 1e-6, seed: int = 0):
        self.B = int(B)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.rng = np.random.default_rng(seed)
        self.pi_ = None
        self.Omega_ = None
        self.r_ = None
        self.elbo_ = None

    @staticmethod
    def _init_r(W: np.ndarray, B: int, rng: np.random.Generator) -> np.ndarray:
        n = W.shape[0]
        deg = (W > 0).sum(axis=1).astype(float)
        bins = np.percentile(deg, np.linspace(0, 100, B+1))
        r = np.zeros((n, B))
        for i in range(n):
            b = np.searchsorted(bins, deg[i], side="right") - 1
            b = min(max(b, 0), B-1)
            r[i, b] = 1.0
        r += 1e-3 * rng.random((n, B))
        r /= r.sum(axis=1, keepdims=True)
        return r

    @staticmethod
    def _expected_block_counts(W: np.ndarray, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n, B = r.shape
        T = r.T @ W @ r
        m = np.triu(T, 0) / 2.0
        m = np.triu(m, 0) + np.triu(m, 1).T
        block_sizes = r.sum(axis=0)
        n_ab = np.empty((B, B))
        for a in range(B):
            for b in range(B):
                if a == b:
                    s = block_sizes[a]
                    n_ab[a, b] = s * (s - 1.0) / 2.0
                else:
                    n_ab[a, b] = block_sizes[a] * block_sizes[b] / 2.0
        n_ab = np.triu(n_ab, 0) + np.triu(n_ab, 1).T
        return m, n_ab

    @staticmethod
    def _update_Omega(m: np.ndarray, n_ab: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        Ω = (m + eps) / (n_ab + eps)
        Ω = (Ω + Ω.T) / 2.0
        return Ω

    @staticmethod
    def _update_pi(r: np.ndarray) -> np.ndarray:
        pi = r.mean(axis=0)
        pi /= pi.sum()
        return pi

    def _e_step(self, W: np.ndarray, pi: np.ndarray, Ω: np.ndarray, r: np.ndarray) -> np.ndarray:
        n, B = r.shape
        M = np.log(np.maximum(Ω, 1e-12))
        C = Ω
        RM = r @ M.T
        RC = r @ C.T
        log_pi = np.log(np.maximum(pi, 1e-15))
        term1 = W @ RM
        term2 = np.sum(RC, axis=0, keepdims=True)
        U = log_pi + term1 - term2
        U = U - logsumexp(U, axis=1)[:, None]
        r_new = np.exp(U)
        r_new = np.clip(r_new, 1e-12, 1.0)
        r_new /= r_new.sum(axis=1, keepdims=True)
        return r_new

    def _elbo(self, W: np.ndarray, pi: np.ndarray, Ω: np.ndarray, r: np.ndarray) -> float:
        n, B = r.shape
        log_pi = np.log(np.maximum(pi, 1e-15))
        term_mix = np.sum(r * (log_pi[None, :] - np.log(np.maximum(r, 1e-15))))
        M = np.log(np.maximum(Ω, 1e-15))
        T1 = r @ M @ r.T
        T2 = r @ Ω @ r.T
        iu = np.triu_indices(n, k=1)
        ll = np.sum(W[iu] * T1[iu] - T2[iu])
        return term_mix + ll

    def fit(self, W: np.ndarray) -> "PoissonSBM":
        W = np.asarray(W, dtype=float)
        r = self._init_r(W, self.B, self.rng)
        m, n_ab = self._expected_block_counts(W, r)
        Ω = self._update_Omega(m, n_ab)
        π = self._update_pi(r)
        prev_elbo = -np.inf
        for it in range(self.max_iter):
            r = self._e_step(W, π, Ω, r)
            π = self._update_pi(r)
            m, n_ab = self._expected_block_counts(W, r)
            Ω = self._update_Omega(m, n_ab)
            elbo = self._elbo(W, π, Ω, r)
            if abs(elbo - prev_elbo) < self.tol * (1.0 + abs(prev_elbo)):
                break
            prev_elbo = elbo
        self.pi_ = π
        self.Omega_ = Ω
        self.r_ = r
        self.elbo_ = prev_elbo
        return self

# ---- Model selection ---------------------------------------------------------

def mdl_score_bic_like(W: np.ndarray, model: PoissonSBM) -> float:
    n = W.shape[0]
    B = model.B
    elbo = model.elbo_ if model.elbo_ is not None else -np.inf
    k_params = (B * (B + 1)) // 2 + (B - 1)
    M = n * (n - 1) / 2.0
    L = -elbo + 0.5 * k_params * math.log(max(M, 2.0))
    return L

def fit_sbm_mdl(W: np.ndarray,
                B_min: int = 1,
                B_max: int = 12,
                max_iter: int = 200,
                tol: float = 1e-6,
                n_restarts: int = 3,
                seed: int = 0,
                show_progress: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    rng = np.random.default_rng(seed)
    best = {"score": np.inf, "B": None, "model": None}
    B_range = range(B_min, B_max + 1)
    iterator = tqdm(B_range, disable=not show_progress, desc="  Searching best B")
    for B in iterator:
        best_local = {"score": np.inf, "model": None}
        for rr in range(n_restarts):
            m = PoissonSBM(B=B, max_iter=max_iter, tol=tol, seed=int(rng.integers(1e9)))
            m.fit(W)
            score = mdl_score_bic_like(W, m)
            if score < best_local["score"]:
                best_local = {"score": score, "model": m}
        if best_local["score"] < best["score"]:
            best = {"score": best_local["score"], "B": B, "model": best_local["model"]}
        iterator.set_postfix({"best_B": best["B"], "score": best["score"]})
    model = best["model"]
    Ω, π, r = model.Omega_, model.pi_, model.r_
    Ωc, πc, perm = canonicalize_blocks(Ω, π)
    return Ωc, πc, r, best["B"], best["score"]

# ---- Canonicalization & vectorization ---------------------------------------

def canonicalize_blocks(Ω: np.ndarray, π: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    B = Ω.shape[0]
    intra = np.diag(Ω)
    order = np.lexsort((-π, -intra))
    Ωc = Ω[np.ix_(order, order)]
    πc = π[order] / np.sum(π)
    return Ωc, πc, order

def vectorize_mininfo(Ω: np.ndarray, π: np.ndarray, B_cap: int) -> np.ndarray:
    B = Ω.shape[0]
    iu = np.triu_indices(B)
    vΩ = Ω[iu].astype(float)
    vπ = π.astype(float)
    U_cap = B_cap * (B_cap + 1) // 2
    S = np.zeros(U_cap + B_cap, dtype=float)
    S[:len(vΩ)] = vΩ
    S[U_cap:U_cap+B] = vπ
    return S

# ---- Language pipeline -------------------------------------------------------

def minimal_info_by_language(lang_graphs: Dict[str, nx.Graph],
                             B_min: int = 1,
                             B_max: int = 10,
                             max_iter: int = 200,
                             tol: float = 1e-6,
                             n_restarts: int = 3,
                             seed: int = 0) -> Tuple[Dict[str, dict], np.ndarray, List[str]]:
    names = list(lang_graphs.keys())
    results = {}
    S_list = []
    print(f"📊 Processing {len(names)} languages...")
    for name in tqdm(names, desc="Languages"):
        W = graph_to_weight_matrix(lang_graphs[name])
        Ωc, πc, r, B_best, score = fit_sbm_mdl(
            W, B_min=B_min, B_max=B_max, max_iter=max_iter,
            tol=tol, n_restarts=n_restarts, seed=seed,
            show_progress=False
        )
        S = vectorize_mininfo(Ωc, πc, B_cap=B_max)
        results[name] = {"Omega": Ωc, "pi": πc, "S": S, "B": B_best, "mdl": score}
        S_list.append(S)
    S_mat = np.vstack(S_list)
    D = np.sqrt(((S_mat[:, None, :] - S_mat[None, :, :]) ** 2).sum(axis=2))
    return results, D, names

# ---- Summarization -----------------------------------------------------------

def summarize_language_result(name: str, res: dict, top_k: int = 4) -> str:
    Ω, π, B = res["Omega"], res["pi"], res["B"]
    msg = [f"Language: {name}",
           f"  B* (MDL): {B}",
           f"  Block sizes (π): {np.round(π, 4)}",
           f"  Ω (first {min(B, top_k)}x{min(B, top_k)} shown):"]
    k = min(B, top_k)
    for i in range(k):
        row = " ".join(f"{Ω[i,j]:.4f}" for j in range(k))
        msg.append(f"    {row}")
    return "\n".join(msg)

# ---- Pickle I/O --------------------------------------------------------------

def load_graphs_pickle(path: str) -> Dict[str, nx.Graph]:
    with open(path, "rb") as f:
        graphs = pickle.load(f)
    print(f"✅ Loaded {len(graphs)} graphs from '{path}'")
    return graphs

def save_mininfo_results(results: Dict[str, dict], D: np.ndarray,
                         names: List[str], out_dir: str = "mininfo_out") -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.pkl"), "wb") as f:
        pickle.dump({"results": results, "names": names}, f)
    np.save(os.path.join(out_dir, "distances.npy"), D)
    with open(os.path.join(out_dir, "distances.csv"), "w", encoding="utf-8") as f:
        f.write(",".join([""] + names) + "\n")
        for i, ni in enumerate(names):
            row = [ni] + [f"{D[i,j]:.6f}" for j in range(len(names))]
            f.write(",".join(row) + "\n")
    print(f"💾 Saved results to '{out_dir}/'")

# ---- Main -------------------------------------------------------------------

if __name__ == "__main__":
    PICKLES_DIR = "pickles"
    GRAPHS_PKL  = os.path.join(PICKLES_DIR, "graphs.pkl")
    B_MIN, B_MAX, N_RESTARTS, SEED = 1, 10, 5, 42

    print(f"📦 Loading graphs from: {GRAPHS_PKL}")
    lang_graphs = load_graphs_pickle(GRAPHS_PKL)
    print("🚀 Running SBM + MDL pipeline ...")

    results, D, names = minimal_info_by_language(
        lang_graphs, B_min=B_MIN, B_max=B_MAX,
        n_restarts=N_RESTARTS, seed=SEED
    )

    print("\n=== Summary of first 3 languages ===")
    for nm in names[:3]:
        print(summarize_language_result(nm, results[nm], top_k=5))
        print("-" * 50)

    save_mininfo_results(results, D, names)
    print("\n✅ Done.")
