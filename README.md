# ReForfit.ai Reverse Engineering

This repository contains analysis and experiments to compare an original large model (Qwen3-4B) against an obfuscated version (refortifai/Qwen3-4B-obfuscated). The primary goal is to detect and characterize structural transformations applied to the obfuscated weights and to show how the original weights can be reconstructed.

Notebook: reforfit-ai-reveng.ipynb (source of this README)

## Summary

- We download both model checkpoints (original and obfuscated) and compare matching tensor weights.
- Element-wise comparisons (correlation, cosine similarity, distribution plots) show almost no direct linear relationship between corresponding weights.
- Spectral properties (singular values) are preserved: singular values of corresponding blocks are nearly identical.
- The obfuscation is a block-wise reparameterization: each column chunk (empirically chunk=512) undergoes an SVD-basis change plus singular-value scaling. By aligning SVDs per block we can reconstruct the obfuscated weights from the original with negligible MSE (~5.47e-12 in the notebook example).

## Key Findings

1. Correlation and cosine similarity between flattened original and obfuscated weights are near-zero → no simple element-wise mapping.
2. Distribution statistics (mean/std) and sorted-value plots differ substantially, so obfuscation alters per-element distribution.
3. Singular values are highly similar (corr ≈ 0.99999999), indicating the operator's spectral profile is preserved.
4. Block-wise SVD alignment (chunk size = 512) reconstructs obfuscated weights with near-zero MSE, proving the obfuscation is a block-wise change of singular bases (a structured, invertible transform rather than quantization or random noise).

## Reproduce (high-level)

Requirements
- Python 3.8+ (the notebook used Python 3.12)
- torch
- safetensors
- huggingface-hub
- transformers
- numpy
- scipy
- matplotlib

Example setup

pip install -U transformers huggingface_hub safetensors torch numpy scipy matplotlib

Main steps (from notebook)
1. Download checkpoints (example uses huggingface_hub.snapshot_download):
   - Qwen/Qwen3-4B → local_dir="Qwen3-4B"
   - refortifai/Qwen3-4B-obfuscated → local_dir="Qwen3-4B-obfuscated"

2. Helper to load a particular tensor from safetensors shards using model.safetensors.index.json and safetensors.torch.load_file.

3. Compare a representative tensor (example key used in the notebook: `model.layers.0.self_attn.q_proj.weight`) with:
   - correlation, norm ratio, mean/std
   - sorted-value plots and histograms
   - cosine similarity
   - unique values / histogram bins (to rule out coarse quantization)
   - block uniqueness to detect per-block repetition
   - Spearman correlation and power-law fit

4. Perform SVD on corresponding (sub)matrices and compare singular values.

5. Split matrices into column chunks (found best chunk ≈ 512) and compute per-chunk SVDs. Align U/V matrices per chunk and apply singular-value correction to reconstruct the obfuscated block. Concatenate reconstructed blocks and compute MSE.

6. Visualize results: sorted-value comparison, singular values overlay, histograms, and a reconstruction error heatmap.

## Notebook structure
- Cells include installation, snapshot downloads, safetensors loader, basic statistics, distribution plots, monotonicity tests (Spearman), linear-mapping tests, SVD-based analyses, per-chunk SVD alignment, and final reconstruction with numeric verification (MSE).

## Interpretation and implications
- The obfuscation used is invertible and structured (block-wise SVD reparameterization). Although per-element values and distributions change dramatically, the model's linear operators are functionally equivalent up to per-block basis changes and singular-value scaling.
- This means that obfuscation by reparameterizing blocks via orthonormal basis changes (and possibly singular-value adjustments) does not hide the operator-level structure and can be reversed by spectral alignment when original weights are available.

## Files
- reforfit-ai-reveng.ipynb — full interactive analysis and code used to generate the results summarized above.

## Notes & Cautions
- The notebook downloads potentially large model checkpoints from Hugging Face; ensure you have sufficient disk, bandwidth, and access permissions.
- Running full SVDs on large weight matrices can be memory- and compute-intensive. Use smaller sub-blocks, sampling, or a machine with adequate RAM/CPU/GPU.
- Respect model licensing and access policies when downloading third-party checkpoints.

## Contact
If you need clarifications or help reproducing the experiments, open an issue or contact me.

---

