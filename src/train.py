"""Train the dense control and the sparse-node model under identical conditions."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import CharDataset  # noqa: E402
from model import CharLM  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ffn", choices=["dense", "sparse"], required=True)
    parser.add_argument("--data", default="hindi_corpus.txt", type=Path)
    parser.add_argument("--out", default="runs", type=Path)
    parser.add_argument("--dim", default=96, type=int)
    parser.add_argument("--layers", default=3, type=int)
    parser.add_argument("--heads", default=4, type=int)
    parser.add_argument("--seq-len", default=128, type=int)
    parser.add_argument("--batch", default=8, type=int)
    parser.add_argument("--steps", default=150, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--threads", default=1, type=int)
    parser.add_argument("--nodes", default=8, type=int)
    parser.add_argument("--active", default=2, type=int)
    parser.add_argument("--max-chars", default=1_500_000, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    dataset = CharDataset(args.data, args.seq_len, args.max_chars)
    model = CharLM(
        vocab=dataset.vocab_size, dim=args.dim, layers=args.layers, heads=args.heads,
        seq_len=args.seq_len, ffn=args.ffn, nodes=args.nodes, active=args.active,
    )
    parameters = sum(p.numel() for p in model.parameters())
    print(f"{args.ffn}: {parameters:,} parameters, "
          f"FFN compute ratio {model.compute_ratio():.2f}")

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    started = time.time()
    history = []

    for step in range(1, args.steps + 1):
        model.train()
        inputs, targets = dataset.batch(dataset.train, args.batch, args.seq_len, rng)
        optimiser.zero_grad()
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()

        if step % 25 == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                vi, vt = dataset.batch(dataset.valid, args.batch, args.seq_len,
                                       np.random.default_rng(1))
                vl = F.cross_entropy(model(vi).reshape(-1, dataset.vocab_size),
                                     vt.reshape(-1)).item()
            elapsed = time.time() - started
            history.append({"step": step, "valid_loss": vl, "seconds": elapsed})
            print(f"step {step:4d}  valid {vl:.4f}  ppl {math.exp(vl):7.2f}  {elapsed:6.1f}s",
                  flush=True)

    metrics = {
        "ffn": args.ffn,
        "parameters": parameters,
        "compute_ratio": model.compute_ratio(),
        "steps": args.steps,
        "seconds": round(time.time() - started, 2),
        "final_valid_loss": history[-1]["valid_loss"],
        "final_perplexity": math.exp(history[-1]["valid_loss"]),
        "history": history,
    }
    if args.ffn == "sparse":
        metrics["node_load"] = model.active_load()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.ffn}_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"final valid {metrics['final_valid_loss']:.4f}  ppl {metrics['final_perplexity']:.2f}")
    if "node_load" in metrics:
        print("node load (layer 0):", [round(v, 3) for v in metrics["node_load"][0]])


if __name__ == "__main__":
    main()
