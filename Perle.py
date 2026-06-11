from __future__ import annotations
import os
import csv
import json
import time
import torch
import random
import shutil
import subprocess
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import forgi
import structure


VALID_PAIRS = {('A', 'U'), ('U', 'A'), ('C', 'G'), ('G', 'C')}


@dataclass
class PerleConfig:
    name: str = "1hji"
    alphabet: str = "AUCG"
    seq_len: int = 15
    target_str: str = "GCCCUGAAAAAGGGC"
    struct_str: Optional[str] = None

    batch: int = 1024
    rounds: int = 200
    lr: float = 3e-3
    noise_d: int = 32
    hid_d: int = 128
    entropy_beta: float = 1e-2
    diversity_beta: float = 5e-2
    pair_weight: float = 1.0
    unpair_weight: float = 1.0

    pair_constraints: List[List[int]] = field(
        default_factory=lambda: [[1, 15], [2, 14], [3, 13], [4, 12], [5, 11]]
    )
    conserved_constraints: Dict[int, str] = field(default_factory=dict)
    unpair_constraints: List[List[int]] = field(default_factory=list)

    num_seq: int = 10
    eval_samples: int = 512
    gen_candidates: int = 8192
    gen_batch: int = 1024
    sample_temperature: float = 0.7

    input_dir: Optional[str] = None
    output_dir: Optional[str] = None
    target_cif: Optional[str] = None
    time_csv: Optional[str] = None
    detailed_summary_csv: Optional[str] = None
    train_summary_csv: Optional[str] = None

    seed: int = 777
    run_af3: bool = True
    af3_extra_args: str = ""
    af3_batch_size: int = 10
    af3_image: str = "/dssg/share/imgs/ai/alphafold/alphafold3-a800.sif"
    af3_model_dir: str = "/dssg/share/data/alphafold3/models"
    af3_db_dir: str = "/dssg/share/data/alphafold3/database"

    use_compile: bool = True
    compile_mode: str = "reduce-overhead"
    flush_every: int = 50

    pair_constraints0: List[List[int]] = field(init=False, repr=False)
    conserved_constraints0: Dict[int, str] = field(init=False, repr=False)
    unpair_constraints0: List[List[int]] = field(init=False, repr=False)

    def __post_init__(self):
        if self.input_dir is None:
            self.input_dir = f"./input_{self.name}"
        if self.output_dir is None:
            self.output_dir = f"./output_{self.name}"
        if self.target_cif is None:
            self.target_cif = f"{self.name}.cif"
        if self.time_csv is None:
            self.time_csv = f"time_{self.name}.csv"
        if self.detailed_summary_csv is None:
            self.detailed_summary_csv = f"detailed_summary_{self.name}.csv"
        if self.train_summary_csv is None:
            self.train_summary_csv = f"train_summary_{self.name}.csv"

        if len(self.target_str) != self.seq_len:
            raise ValueError("target_str length must equal seq_len")

        if self.struct_str is not None:
            if len(self.struct_str) != self.seq_len:
                raise ValueError("struct_str length must equal seq_len")
            self.pair_constraints = _parse_extended_dot_bracket_1based(self.struct_str)

        self.pair_constraints = _normalize_pairs_1based(self.pair_constraints, self.seq_len, "pair_constraints")
        self.unpair_constraints = _normalize_pairs_1based(self.unpair_constraints, self.seq_len, "unpair_constraints")
        self.conserved_constraints = _normalize_conserved_1based(self.conserved_constraints, self.seq_len, self.alphabet)

        self.pair_constraints0 = [[i - 1, j - 1] for i, j in self.pair_constraints]
        self.unpair_constraints0 = [[i - 1, j - 1] for i, j in self.unpair_constraints]
        self.conserved_constraints0 = {i - 1: nt for i, nt in self.conserved_constraints.items()}


def _normalize_pairs_1based(pairs: List[List[int]], seq_len: int, name: str) -> List[List[int]]:
    norm = []
    seen = set()
    for pair in pairs:
        if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
            raise ValueError(f"Each item in {name} must be [i, j].")
        i, j = int(pair[0]), int(pair[1])
        if not (1 <= i <= seq_len and 1 <= j <= seq_len):
            raise ValueError(f"{name} out of range: {pair}")
        if i == j:
            raise ValueError(f"{name} cannot contain identical positions: {pair}")
        key = tuple(sorted((i, j)))
        if key not in seen:
            seen.add(key)
            norm.append([key[0], key[1]])
    norm.sort(key=lambda x: x[0])
    return norm


def _normalize_conserved_1based(conserved: Dict[int, str], seq_len: int, alphabet: str) -> Dict[int, str]:
    norm = {}
    for pos, nt in conserved.items():
        pos = int(pos)
        if not (1 <= pos <= seq_len):
            raise ValueError(f"Conserved position out of range: {pos}")
        if nt not in alphabet:
            raise ValueError(f"Conserved nucleotide '{nt}' not in alphabet '{alphabet}'")
        norm[pos] = nt
    return dict(sorted(norm.items()))


def _parse_extended_dot_bracket_1based(db: str) -> List[List[int]]:
    open_to_close = {'(': ')', '[': ']', '{': '}', '<': '>'}
    close_to_open = {v: k for k, v in open_to_close.items()}
    stacks = {ch: [] for ch in open_to_close}
    pairs: List[List[int]] = []

    for idx, ch in enumerate(db, start=1):
        if ch in open_to_close:
            stacks[ch].append(idx)
        elif ch in close_to_open:
            op = close_to_open[ch]
            if not stacks[op]:
                raise ValueError(f"Unbalanced dot-bracket string: unexpected '{ch}' at position {idx}")
            pairs.append([stacks[op].pop(), idx])
        elif ch in ('.', '&'):
            continue
        else:
            raise ValueError(f"Unsupported character '{ch}' in struct_str at position {idx}")

    for op, st in stacks.items():
        if st:
            raise ValueError(f"Unbalanced dot-bracket string: missing closing bracket(s) for '{op}' at positions {st}")

    pairs.sort(key=lambda p: p[0])
    return pairs


def _pairs_to_dot_bracket_1based(pair_constraints: List[List[int]], seq_len: int) -> str:
    db = ['.'] * seq_len
    for i, j in pair_constraints:
        i0, j0 = i - 1, j - 1
        if i0 < j0:
            db[i0] = '('
            db[j0] = ')'
        else:
            db[j0] = '('
            db[i0] = ')'
    return "".join(db)


def _build_vocab(alphabet: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    stoi = {ch: i for i, ch in enumerate(alphabet)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def _ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


class GenMLP(nn.Module):
    def __init__(self, vocab: int, seq_len: int, zdim: int, hid: int):
        super().__init__()
        self.zdim = zdim
        self.seq_len = seq_len
        self.vocab = vocab
        self.net = nn.Sequential(
            nn.Linear(zdim, hid),
            nn.ReLU(),
            nn.Linear(hid, hid),
            nn.ReLU(),
            nn.Linear(hid, seq_len * vocab),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).view(z.size(0), self.seq_len, self.vocab)


def _build_conserved_mask(seq_len: int, vocab: int, conserved0: Dict[int, str], stoi: Dict[str, int]) -> torch.Tensor:
    mask = torch.ones(seq_len, vocab, dtype=torch.bool)
    for pos, nt in conserved0.items():
        mask[pos, :] = False
        mask[pos, stoi[nt]] = True
    return mask


def _apply_conserved_mask(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~mask.unsqueeze(0), float("-inf"))


def _enforce_conserved(seq: str, conserved0: Dict[int, str]) -> str:
    s = list(seq)
    for i, nt in conserved0.items():
        s[i] = nt
    return "".join(s)


def _to_str(sample: torch.Tensor, itos: Dict[int, str]) -> str:
    return "".join(itos[int(i)] for i in sample)


def _calc_similarity(seq: str, target: str) -> float:
    same = sum(1 for a, b in zip(seq, target) if a == b)
    return 100.0 * same / len(seq)


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_generator(seed: int, device: torch.device) -> torch.Generator:
    return torch.Generator(device=device.type).manual_seed(seed)


def _prepare_runtime(cfg: PerleConfig, stoi: Dict[str, int], itos: Dict[int, str], device: torch.device) -> Dict[str, torch.Tensor]:
    wc_map = {"A": "U", "U": "A", "C": "G", "G": "C"}
    runtime = {
        "target_idx": torch.tensor([stoi[ch] for ch in cfg.target_str], dtype=torch.long, device=device),
        "wc": torch.tensor([stoi[wc_map[itos[i]]] for i in range(len(cfg.alphabet))], dtype=torch.long, device=device),
        "pair_i": torch.tensor([i for i, _ in cfg.pair_constraints0], dtype=torch.long, device=device),
        "pair_j": torch.tensor([j for _, j in cfg.pair_constraints0], dtype=torch.long, device=device),
        "unpair_i": torch.tensor([i for i, _ in cfg.unpair_constraints0], dtype=torch.long, device=device),
        "unpair_j": torch.tensor([j for _, j in cfg.unpair_constraints0], dtype=torch.long, device=device),
    }
    return runtime


def _score_base_pairing(samples: torch.Tensor, runtime: Dict[str, torch.Tensor]) -> torch.Tensor:
    pair_i = runtime["pair_i"]
    if pair_i.numel() == 0:
        return torch.ones(samples.size(0), device=samples.device)
    pair_j = runtime["pair_j"]
    wc = runtime["wc"]
    return (samples[:, pair_j] == wc[samples[:, pair_i]]).float().mean(dim=1)


def _score_unpairing(samples: torch.Tensor, runtime: Dict[str, torch.Tensor]) -> torch.Tensor:
    unpair_i = runtime["unpair_i"]
    if unpair_i.numel() == 0:
        return torch.ones(samples.size(0), device=samples.device)
    unpair_j = runtime["unpair_j"]
    wc = runtime["wc"]
    return (samples[:, unpair_j] != wc[samples[:, unpair_i]]).float().mean(dim=1)


def _similarity_fraction(samples: torch.Tensor, runtime: Dict[str, torch.Tensor]) -> torch.Tensor:
    target_idx = runtime["target_idx"]
    return (samples == target_idx.unsqueeze(0)).float().mean(dim=1)


def _bounded_reward(pair_score: torch.Tensor, similarity_frac: torch.Tensor, beta: float) -> torch.Tensor:
    return pair_score * (1.0 - beta * similarity_frac)


def _seq_pair_score(seq: str, pair_constraints0: List[List[int]]) -> float:
    if not pair_constraints0:
        return 1.0
    good = sum((seq[i], seq[j]) in VALID_PAIRS for i, j in pair_constraints0)
    return good / len(pair_constraints0)


def _seq_unpair_score(seq: str, unpair_constraints0: List[List[int]]) -> float:
    if not unpair_constraints0:
        return 1.0
    good = sum((seq[i], seq[j]) not in VALID_PAIRS for i, j in unpair_constraints0)
    return good / len(unpair_constraints0)


def _satisfied_pair_tuples(seq: str, pair_constraints0: List[List[int]]) -> List[Tuple[int, int]]:
    return [(i + 1, j + 1) for i, j in pair_constraints0 if (seq[i], seq[j]) in VALID_PAIRS]


@torch.inference_mode()
def _evaluate_fixed_bank(model: GenMLP, eval_z: torch.Tensor, conserved_mask: torch.Tensor,
                         cfg: PerleConfig, runtime: Dict[str, torch.Tensor]) -> Dict[str, float]:
    logits = _apply_conserved_mask(model(eval_z), conserved_mask)
    toks = logits.argmax(-1)
    pair_score = _score_base_pairing(toks, runtime)
    similarity_frac = _similarity_fraction(toks, runtime)
    reward = _bounded_reward(pair_score, similarity_frac, cfg.diversity_beta)
    return {
        "mean_reward": reward.mean().item(),
        "mean_base_pairing": pair_score.mean().item(),
    }


@torch.inference_mode()
def _generate_from_model(model: GenMLP, cfg: PerleConfig, conserved_mask: torch.Tensor,
                         runtime: Dict[str, torch.Tensor], device: torch.device,
                         itos: Dict[int, str]) -> List[str]:
    all_toks = []
    remaining = cfg.gen_candidates
    gen = _make_generator(cfg.seed + 22222, device)

    while remaining > 0:
        n = min(cfg.gen_batch, remaining)
        z = torch.randn(n, model.zdim, generator=gen, device=device)
        logits = _apply_conserved_mask(model(z), conserved_mask)

        greedy = logits.argmax(-1)
        all_toks.append(greedy)

        if cfg.sample_temperature > 0:
            sampled = torch.distributions.Categorical(logits=logits / cfg.sample_temperature).sample()
            all_toks.append(sampled)
        remaining -= n

    toks = torch.cat(all_toks, dim=0)
    pair_score = _score_base_pairing(toks, runtime)
    unpair_score = _score_unpairing(toks, runtime)
    similarity = 100.0 * (toks == runtime["target_idx"].unsqueeze(0)).float().mean(dim=1)
    exact = ((pair_score == 1.0) & (unpair_score == 1.0)).float()

    order = torch.lexsort if hasattr(torch, "lexsort") else None
    if order is None:
        rank = torch.argsort(
            exact * 1e9 + pair_score * 1e6 + unpair_score * 1e3 - similarity,
            descending=True,
        )
    else:
        rank = torch.lexsort((-similarity, unpair_score, pair_score, exact), dim=0).flip(0)

    sequences = []
    seen = set()
    for idx in rank.tolist():
        seq = _enforce_conserved(_to_str(toks[idx].cpu(), itos), cfg.conserved_constraints0)
        if seq in seen:
            continue
        seen.add(seq)
        sequences.append(seq)
        if len(sequences) >= cfg.num_seq:
            break

    exact_count = sum(
        _seq_pair_score(seq, cfg.pair_constraints0) == 1.0 and _seq_unpair_score(seq, cfg.unpair_constraints0) == 1.0
        for seq in sequences
    )
    if exact_count < len(sequences):
        print(f"[Perle] Warning: only {exact_count}/{len(sequences)} returned sequences satisfy all pair/unpair constraints exactly.")
    return sequences


def _create_af3_json(sequence: str, json_path: str, name: str):
    data = {
        "name": name,
        "sequences": [{"rna": {"id": "A", "sequence": sequence, "unpairedMsa": ""}}],
        "modelSeeds": [1],
        "dialect": "alphafold3",
        "version": 1,
    }
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)


def _write_af3_batch_script(script_path: str, batch_input_dir: str, batch_output_dir: str, cfg: PerleConfig):
    abs_input = os.path.abspath(batch_input_dir)
    abs_output = os.path.abspath(batch_output_dir)
    extra_args = f" {cfg.af3_extra_args}" if cfg.af3_extra_args else ""
    script = f"""#!/bin/bash
set -euo pipefail
singularity exec \\
  --nv \\
  --bind {abs_input}:/root/af_input \\
  --bind {abs_output}:/root/af_output \\
  --bind {cfg.af3_model_dir}:/root/models \\
  --bind {cfg.af3_db_dir}:/root/public_databases \\
  {cfg.af3_image} \\
  /alphafold3_venv/bin/python /app/alphafold/run_alphafold.py \\
  --input_dir=/root/af_input \\
  --model_dir=/root/models \\
  --db_dir=/root/public_databases \\
  --output_dir=/root/af_output{extra_args}
"""
    with open(script_path, 'w') as f:
        f.write(script)
    os.chmod(script_path, 0o755)


def _run_af3_batch(batch_input_dir: str, batch_output_dir: str, batch_tag: str, cfg: PerleConfig) -> int:
    script_path = os.path.join(cfg.input_dir, f"{batch_tag}.sh")
    _write_af3_batch_script(script_path, batch_input_dir, batch_output_dir, cfg)
    return subprocess.run(["bash", script_path]).returncode


def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    if P.shape != Q.shape or P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("P and Q must both have shape (N, 3).")

    cP = P.mean(axis=0)
    cQ = Q.mean(axis=0)
    X = P - cP
    Y = Q - cQ
    H = X.T @ Y
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0.0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    t = cQ - R @ cP
    P_aligned = P @ R.T + t
    rmsd = np.sqrt(np.mean(np.sum((P_aligned - Q) ** 2, axis=1)))
    return float(rmsd), R, t


def _calc_rmsd_kabsch_cif(predicted_cif: str, target_cif: str) -> float:
    query_rna = structure.load_cif(predicted_cif)
    target_rna = structure.load_cif(target_cif)
    query_c1 = np.array([residue.get_atom_by_name("C1'").get_coord() for residue in query_rna[0]])
    target_c1 = np.array([residue.get_atom_by_name("C1'").get_coord() for residue in target_rna[0]])
    rmsd, _, _ = kabsch_rmsd(query_c1, target_c1)
    return float(rmsd)


def _get_dot_bracket_from_cif(cif_path: str) -> str:
    rna_components = forgi.load_rna(cif_path, pdb_remove_pk=False)
    if not isinstance(rna_components, (list, tuple)):
        rna_components = list(rna_components)
    cg = rna_components[0]
    fasta = cg.to_fasta_string().strip().splitlines()
    if len(fasta) >= 3:
        return fasta[2].strip().replace("&", "")
    return cg.to_dotbracket_string().replace("&", "")


def _pair_tuples_from_dot_bracket_1based(db: str) -> List[Tuple[int, int]]:
    return [tuple(p) for p in _parse_extended_dot_bracket_1based(db)]


def _pair_tuples_from_cif(cif_path: str) -> List[Tuple[int, int]]:
    return _pair_tuples_from_dot_bracket_1based(_get_dot_bracket_from_cif(cif_path))


def _dotbracket_diff_vector(pred_db: Optional[str], target_db: Optional[str], seq_len: int) -> str:
    if not pred_db or not target_db:
        return ""
    bits = []
    for i in range(seq_len):
        c_pred = pred_db[i] if i < len(pred_db) else ' '
        c_tgt = target_db[i] if i < len(target_db) else ' '
        bits.append('0' if c_pred == c_tgt and c_pred != ' ' else '1')
    return "(" + ",".join(bits) + ")"


def _find_pred_cif(job_dir: str, job_name: str) -> Optional[str]:
    top_cif = os.path.join(job_dir, f"{job_name}_model.cif")
    if os.path.exists(top_cif):
        return top_cif
    seed_cif = os.path.join(job_dir, "seed-1_sample-0", "model.cif")
    if os.path.exists(seed_cif):
        return seed_cif
    if os.path.isdir(job_dir):
        for root, _, files in os.walk(job_dir):
            for f in files:
                if f.endswith(".cif"):
                    return os.path.join(root, f)
    return None


def _run_af3_and_rmsd(sequences: List[str], iteration: str, cfg: PerleConfig) -> Tuple[List[float], List[Optional[str]], List[List[Tuple[int, int]]], float]:
    rmsd_values: List[float] = []
    dot_brackets: List[Optional[str]] = []
    pair_details_from_af3: List[List[Tuple[int, int]]] = []
    total_af3_time = 0.0

    for batch_start in range(0, len(sequences), cfg.af3_batch_size):
        batch_idx = batch_start // cfg.af3_batch_size + 1
        batch_tag = f"iter{iteration}_batch{batch_idx}"
        batch_input_dir = os.path.join(cfg.input_dir, batch_tag)
        batch_output_dir = os.path.join(cfg.output_dir, batch_tag)
        shutil.rmtree(batch_input_dir, ignore_errors=True)
        shutil.rmtree(batch_output_dir, ignore_errors=True)
        _ensure_dirs(batch_input_dir, batch_output_dir)

        job_names = []
        batch_sequences = sequences[batch_start: batch_start + cfg.af3_batch_size]
        for global_idx, seq in enumerate(batch_sequences, start=batch_start + 1):
            job_name = f"iter{iteration}_seq{global_idx}"
            job_names.append(job_name)
            _create_af3_json(seq, os.path.join(batch_input_dir, f"{job_name}.json"), name=job_name)

        if cfg.run_af3:
            print(f"[Perle] Running AlphaFold3 for batch {batch_idx} with {len(batch_sequences)} sequences")
            t0 = time.time()
            rc = _run_af3_batch(batch_input_dir, batch_output_dir, batch_tag, cfg)
            dt = time.time() - t0
            total_af3_time += dt
            print(f"[Perle] AF3 batch {batch_idx} done (rc={rc}) | total time = {dt:.2f}s")

        for job_name in job_names:
            pred_cif = _find_pred_cif(os.path.join(batch_output_dir, job_name), job_name)
            if pred_cif and os.path.exists(pred_cif):
                rmsd = _calc_rmsd_kabsch_cif(pred_cif, cfg.target_cif)
                db = _get_dot_bracket_from_cif(pred_cif)
                af3_pairs = _pair_tuples_from_cif(pred_cif)
            else:
                print(f"[Perle] Warning: No CIF found for {job_name}")
                rmsd = float('inf')
                db = None
                af3_pairs = []
            rmsd_values.append(rmsd)
            dot_brackets.append(db)
            pair_details_from_af3.append(af3_pairs)

    return rmsd_values, dot_brackets, pair_details_from_af3, total_af3_time


def _save_detailed_summary(sequences: List[str], rmsd_values: List[float], dot_brackets: List[Optional[str]],
                           pair_details_from_af3: List[List[Tuple[int, int]]], target_dot_bracket: Optional[str], cfg: PerleConfig):
    with open(cfg.detailed_summary_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            "seq_index", "sequence", "rmsd",
            "similarity_percent", "recovery_loss",
            "base_pairing_score", "unpair_satisfaction_score",
            "total_pair_constraints", "satisfied_pairs", "pair_details",
            "pair_details_from_AF3", "dot_bracket_from_AF3", "dot_bracket_diff"
        ])

        for idx, (seq, rmsd) in enumerate(zip(sequences, rmsd_values), start=1):
            sim = _calc_similarity(seq, cfg.target_str)
            rec = 100.0 - sim
            pair_score = _seq_pair_score(seq, cfg.pair_constraints0)
            unpair_score = _seq_unpair_score(seq, cfg.unpair_constraints0)
            pair_details = _satisfied_pair_tuples(seq, cfg.pair_constraints0)
            db_pred = dot_brackets[idx - 1] if dot_brackets else None
            w.writerow([
                idx,
                seq,
                rmsd,
                sim,
                rec,
                pair_score,
                unpair_score,
                len(cfg.pair_constraints),
                len(pair_details),
                pair_details,
                pair_details_from_af3[idx - 1],
                db_pred or "",
                _dotbracket_diff_vector(db_pred, target_dot_bracket, cfg.seq_len) if target_dot_bracket else "",
            ])


def train_and_generate(cfg: PerleConfig):
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    _ensure_dirs(cfg.input_dir, cfg.output_dir)

    device = _get_device()
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    stoi, itos = _build_vocab(cfg.alphabet)
    conserved_mask = _build_conserved_mask(cfg.seq_len, len(cfg.alphabet), cfg.conserved_constraints0, stoi).to(device)
    runtime = _prepare_runtime(cfg, stoi, itos, device)

    with open(cfg.time_csv, 'w', newline='') as f:
        csv.writer(f).writerow(["phase", "seq_index", "seconds"])

    with open(cfg.train_summary_csv, 'w', newline='') as f:
        csv.writer(f).writerow(["round", "mean_reward", "mean_base_pairing", "baseline"])

    model = GenMLP(vocab=len(cfg.alphabet), seq_len=cfg.seq_len, zdim=cfg.noise_d, hid=cfg.hid_d).to(device)
    if cfg.use_compile and hasattr(torch, "compile"):
        model = torch.compile(model, mode=cfg.compile_mode)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    baseline = torch.tensor(0.0, device=device)
    eval_z = torch.randn(cfg.eval_samples, model.zdim, generator=_make_generator(cfg.seed + 12345, device), device=device)

    print(f"[Perle] Training start | name={cfg.name} | rounds={cfg.rounds} | seq_len={cfg.seq_len} | device={device.type}")
    t0 = time.time()
    train_rows = []

    for r in range(1, cfg.rounds + 1):
        z = torch.randn(cfg.batch, model.zdim, device=device)
        logits = _apply_conserved_mask(model(z), conserved_mask)
        dist = torch.distributions.Categorical(logits=logits)
        toks = dist.sample()
        logp = dist.log_prob(toks)
        ent = dist.entropy()

        pair_reward = _score_base_pairing(toks, runtime)
        similarity_frac = _similarity_fraction(toks, runtime)
        reward = _bounded_reward(pair_reward, similarity_frac, cfg.diversity_beta)

        baseline = 0.95 * baseline + 0.05 * reward.mean().detach()
        advantage = (reward - baseline).unsqueeze(1)
        policy_loss = - (advantage * logp).sum(dim=1).mean()
        entropy_bonus = - cfg.entropy_beta * ent.mean()
        loss = policy_loss + entropy_bonus

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        metrics = _evaluate_fixed_bank(model, eval_z, conserved_mask, cfg, runtime)
        train_rows.append([r, metrics["mean_reward"], metrics["mean_base_pairing"], float(baseline.item())])

        if r == 1 or r % 50 == 0 or r == cfg.rounds:
            print(
                f"[Perle] round={r} "
                f"reward={metrics['mean_reward']:.4f} "
                f"pair={metrics['mean_base_pairing']:.4f} "
                f"baseline={float(baseline.item()):.4f}"
            )

        if r % cfg.flush_every == 0 or r == cfg.rounds:
            with open(cfg.train_summary_csv, 'a', newline='') as f:
                csv.writer(f).writerows(train_rows)
            train_rows.clear()

    model_time = time.time() - t0
    with open(cfg.time_csv, 'a', newline='') as f:
        csv.writer(f).writerow(["model_iteration", "", model_time])
    print(f"[Perle] Training done in {model_time:.2f}s")

    t1 = time.time()
    sequences = _generate_from_model(model, cfg, conserved_mask, runtime, device, itos)
    gen_time = time.time() - t1
    with open(cfg.time_csv, 'a', newline='') as f:
        csv.writer(f).writerow(["sequence_generation", "", gen_time])
    print(f"[Perle] Generated {len(sequences)} sequences in {gen_time:.4f}s")

    target_db_str = cfg.struct_str if cfg.struct_str is not None else _pairs_to_dot_bracket_1based(cfg.pair_constraints, cfg.seq_len)

    if cfg.run_af3:
        rmsd_values, dot_brackets, pair_details_from_af3, af3_time = _run_af3_and_rmsd(sequences, "final", cfg)
        with open(cfg.time_csv, 'a', newline='') as f:
            csv.writer(f).writerow(["af3_total", "", af3_time])
    else:
        rmsd_values = [float('inf')] * len(sequences)
        dot_brackets = [None] * len(sequences)
        pair_details_from_af3 = [[] for _ in sequences]

    _save_detailed_summary(sequences, rmsd_values, dot_brackets, pair_details_from_af3, target_db_str, cfg)
    print(f"[Perle] Saved:\n  {cfg.time_csv}\n  {cfg.train_summary_csv}\n  {cfg.detailed_summary_csv}")

    return {
        "sequences": sequences,
        "rmsd_values": rmsd_values,
        "time_csv": cfg.time_csv,
        "train_summary_csv": cfg.train_summary_csv,
        "detailed_summary_csv": cfg.detailed_summary_csv,
    }


if __name__ == "__main__":
    cfg = PerleConfig()
    train_and_generate(cfg)
