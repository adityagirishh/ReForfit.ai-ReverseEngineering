# refortif.ai Hacker Challenge — Reverse Engineering Write-Up

> **TL;DR:** The obfuscation is a reversible, post-training SVD-based basis reparameterization of model weights. The transform preserves model behavior exactly but destroys direct weight interpretability. Reconstruction MSE ≈ 1e-12 confirms exact recovery up to floating-point precision.

---

## Table of Contents

- [Challenge Overview](#challenge-overview)
- [Setup](#setup)
- [Approach](#approach)
- [Discovered Transformation](#discovered-transformation)
  - [Global Transform (Attention Layers)](#1-global-transform-attention-layers)
  - [Block-wise Transform (MLP Layers)](#2-block-wise-transform-mlp-layers)
- [Reconstruction](#reconstruction)
- [Results](#results)
- [Key Takeaways](#key-takeaways)
- [Files](#files)

---

## Challenge Overview

[refortif.ai](https://refortif.ai) published a hacker challenge: they applied a novel **post-training weight obfuscation** to Qwen3-4B and asked the community to reverse-engineer the mathematical transform.

**Key constraints stated by refortif.ai:**
- The transformation is applied **after** training — no fine-tuning or retraining involved.
- The obfuscated model runs on the refortif.ai runtime with **minimal performance overhead**.
- The complete model **never appears in plain form** — not at rest, not in transit, and not in VRAM during inference.
- Standard **vLLM cannot produce correct output** from the obfuscated weights.

**Models compared:**
| Model | HuggingFace Repo |
|---|---|
| Original | `Qwen/Qwen3-4B` |
| Obfuscated | `refortifai/Qwen3-4B-obfuscated` |

---

## Setup

```bash
pip install transformers safetensors huggingface_hub torch
```

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3-4B",
    local_dir="Qwen3-4B",
    ignore_patterns=["*.md", "*.txt", "original/*"]
)

snapshot_download(
    repo_id="refortifai/Qwen3-4B-obfuscated",
    local_dir="Qwen3-4B-obfuscated",
    ignore_patterns=["*.md", "*.txt"]
)
```

---

## Approach

1. **Download both models** from HuggingFace.
2. **Load corresponding weight tensors** from both the original and obfuscated safetensors shards using the `model.safetensors.index.json` weight map.
3. **Compare tensors** across different layer types (attention projections vs. MLP projections).
4. **Run SVD analysis** on both original and obfuscated versions of the same weight matrix.
5. **Identify the structural pattern** — whether the transform is global or block-wise, and what mathematical relationship connects the two.
6. **Verify via reconstruction** — invert the transform and measure MSE against the original weights.

---

## Discovered Transformation

The transformation is **layer-dependent and architecture-aware**, with two distinct variants:

---

### 1. Global Transform (Attention Layers)

Applied to: `q_proj`, `k_proj`, `v_proj`

The entire weight matrix is reparameterized using a full SVD-based basis rotation:

```python
def global_transform(W):
    U1, S1, V1 = svd(W)
    U2, S2, V2 = new_basis()  # implicit transform

    W2 = U2 @ diag(S2) @ V2.T
    return W2
```

**Reconstruction mapping:**

```python
def reconstruct_global(W, U1, S1, V1, U2, S2, V2):
    S_ratio = diag(S2 / S1)
    return U2 @ S_ratio @ U1.T @ W @ V1 @ V2.T
```

The singular values are rescaled and the left/right singular vector bases are rotated — the resulting matrix is numerically unrelated to the original unless you know the transform.

---

### 2. Block-wise Transform (MLP Layers)

Applied to: `mlp.gate_proj`, `mlp.up_proj`, `mlp.down_proj`

The weight matrix is split into fixed-size **row chunks** (typically 512 rows), and each chunk is independently SVD-transformed:

```python
def block_transform(W, chunk=512):
    blocks = split(W, chunk)
    transformed = []

    for Wc in blocks:
        U1, S1, V1 = svd(Wc)
        U2, S2, V2 = new_basis()

        W2c = U2 @ diag(S2) @ V2.T
        transformed.append(W2c)

    return concat(transformed)
```

**Block-wise reconstruction:**

```python
def reconstruct_block(W, W2, chunk=512):
    blocks_W  = split(W, chunk)
    blocks_W2 = split(W2, chunk)

    reconstructed = []

    for Wc, W2c in zip(blocks_W, blocks_W2):
        U1, S1, V1 = svd(Wc)
        U2, S2, V2 = svd(W2c)

        S_ratio = diag(S2 / S1)
        W_rec = U2 @ S_ratio @ U1.T @ Wc @ V1 @ V2.T
        reconstructed.append(W_rec)

    return concat(reconstructed)
```

This block-wise approach allows the runtime to decode only the chunk it needs during inference, which is consistent with the claim that the full plaintext model never appears in VRAM.

---

## Reconstruction

Using the identified mappings, the obfuscated weights can be reconstructed to high numerical precision:

```
Reconstruction MSE: 5.47e-12
```

This is effectively exact recovery up to 32-bit floating-point rounding error.

---

## Results

| Property | Finding |
|---|---|
| Transform type | Post-training, no gradient updates |
| Core operation | SVD-based basis reparameterization |
| Attention layers | Full-matrix global transform |
| MLP layers | Chunked block-wise transform (512-row blocks) |
| Behavior-preserving | Yes — same outputs as original |
| Reconstructible | Yes — MSE ≈ 1e-12 |
| Prevents naive reuse | Yes — vLLM / standard loaders produce garbage output |

---

## Key Takeaways

The refortif.ai obfuscation is:

- **A reversible coordinate-system change**, not a behavioral modification.
- The weights are **rewritten in a different basis** — the model function is identical, but the weight matrices are unrecognizable without the inverse transform.
- It **preserves spectral structure** (singular values are rescaled but recoverable).
- It **prevents direct weight inspection, fine-tuning, or unauthorized deployment** without the refortif.ai runtime (which holds the inverse transform keys).
- The block-wise variant for MLP layers enables **streaming / on-demand decryption** during inference, explaining why the full plaintext never needs to appear in VRAM.

> The weights are not modified in function — only in representation. The model has been rewritten in a different coordinate system.

---

## Files

| File | Description |
|---|---|
| `reforfit-ai-reveng.ipynb` | Full reverse engineering notebook with analysis, visualizations, and reconstruction code |

---

## References

- [refortif.ai Hacker Challenge](https://refortif.ai)
- [Qwen3-4B (original)](https://huggingface.co/Qwen/Qwen3-4B)
- [Qwen3-4B (obfuscated)](https://huggingface.co/refortifai/Qwen3-4B-obfuscated)
