# Anviksha Sparse Brain — same transformer, only the FFN replaced

A third experiment, in its own repository. The previous models are untouched.

The brief: **do not change the architecture.** Keep attention, norms, embeddings
and the head exactly as they are. Upgrade only what has to change, so the model
behaves more like a brain: only the nodes needed at that moment are active, the
active nodes interconnect dynamically, and they process in parallel — on a CPU,
with as little compute as possible.

## The one change

The feed-forward block is the only thing replaced. It is where most parameters
and most arithmetic already live, so it is the only place a sparse design can
pay off.

```
dense FFN     2 · d · 4d           = 8d² parameters, 8d² multiply-adds per token
node field    8 nodes of width d/2 = 8d² parameters, same total
              top-2 of 8 active     = 2d² multiply-adds  →  ¼ of the arithmetic
```

Same weights budget. A quarter of the FFN arithmetic. Everything else — the
attention, the residual stream, the layer norms — is byte-for-byte the standard
block.

## Dynamic interconnection and parallelism

- **Routing per token, per layer.** A linear router scores all eight nodes and
  keeps the top two. Different tokens, and different layers, pick different
  nodes; nothing is fixed at build time.
- **Interconnection.** After each active node runs, the co-active pair exchanges
  one message — the mean of their pre-projection outputs, passed through a
  learned linear map — before being combined with learned weights. Which nodes
  talk to each other is decided per token, so the graph is different for every
  token and every layer.
- **Parallelism.** Tokens that chose the same node are gathered and run as one
  batched matmul, so all active nodes across the whole batch run together rather
  than in a loop over tokens.

## Measured — and the honest verdict

Hindi character-level LM, 1.5 M characters, identical data order, identical
batch, single thread, CPU.

| Configuration | Dense | Sparse | Sparse / dense |
|---|---|---|---|
| dim 96, 3 layers, 150 steps | 3.1441 loss · 23.20 ppl · 29.2 s | 3.2204 · 25.04 ppl · 34.3 s | **1.17× slower** |
| dim 192, 2 layers, 60 steps | 3.2491 · 25.77 ppl · 15.9 s | 3.3410 · 28.25 ppl · 15.8 s | **1.00× — a tie** |
| Parameters | 418,848 | 448,224 | 1.07× |
| FFN arithmetic | 8d² per token | 2d² per token | **0.25×** |

**The arithmetic saving is real and the wall-clock saving is not.** At dim 96 the
sparse model is 17 % *slower*; at dim 192 it is exactly as fast. Quality is
slightly behind at both, because only a quarter of the FFN capacity is being
trained per step.

### Why the quarter-arithmetic does not become a quarter of the time

1. **The FFN is not the bottleneck at these sizes.** Attention, layer norms and
   the embedding dominate at d = 96 with a 128-token sequence. Cutting the FFN
   by 4× removes a quarter of the work, not three quarters of the runtime.
2. **Gathering costs more than the matmul it saves.** Sorting 1,024 tokens into
   eight groups and scattering the results is memory traffic. At d = 96 the
   matmuls are small enough that this overhead is comparable to the arithmetic
   itself. A gather-and-batched-matmul version was tried and measured **6.5×
   slower** than the grouped loop, so the loop is what shipped.
3. **Sparsity pays at scale, not at toy scale.** Mixture-of-experts wins when
   the per-node matmul is large enough to amortise dispatch — hundreds of
   millions of parameters, big batches, GPU tensor cores. At 0.4 M parameters on
   two CPU cores, it does not.

### What did work as designed

Routing stays balanced, which is the usual failure mode of sparse experts:

```
node load, layer 0: [0.097, 0.138, 0.067, 0.146, 0.211, 0.152, 0.088, 0.101]
```

No node is starved and none collapses to zero — the learned router distributes
work on its own, with no auxiliary load-balancing loss.

## What I would change to actually get the speedup

1. **Raise the dimension to 512+ and the batch to 64+.** The crossover exists;
   it is above the sizes that fit in this sandbox's 2 GB.
2. **Keep one shared node field across all layers** instead of one per layer, so
   the routing decision is amortised and the node set is genuinely global.
3. **Batch many sequences per node call** by flattening the whole batch into one
   token pool before routing — currently each forward pass routes separately.
4. **Only then measure again.** If sparse still does not win at d = 512 on CPU,
   the honest conclusion is that this mechanism belongs on a GPU, and CPU work
   should spend its budget elsewhere.

## What is brain-like here, and what is not

Real: conditional computation (only some units run), dynamic wiring (the graph
is per-token), parallel activation of the selected set, and no fixed pathway
through the network.

Not real: this is not a brain. There are no spikes, no local learning rules, no
recurrence over time, and backpropagation does the credit assignment. The
"nodes" are small MLPs. Calling the routing "attention" or the nodes "neurons"
would be a metaphor, and metaphors do not survive measurement.

## Run it

```bash
python src/train.py --ffn dense  --steps 150 --batch 8
python src/train.py --ffn sparse --steps 150 --batch 8
python src/train.py --ffn dense  --steps 60 --batch 8 --dim 192 --layers 2
python src/train.py --ffn sparse --steps 60 --batch 8 --dim 192 --layers 2
```

## Layout

```
src/nodes.py    SparseNodeField, DenseFFN, shared attention Block
src/model.py    CharLM with either FFN
src/data.py     character-level dataset
src/train.py    identical loop for both variants
configs/default.json
```

The three-tier memory from the previous experiment lives in
`anviksha-triune` and is deliberately **not** merged here: two mechanisms at
once cannot be attributed to either.
