# Kernel Divergence Score (Codex-packaged Detailed README)

This package is a runnable snapshot of the repository, prepared for direct download and local execution.

---

## 1. Project overview

This project studies dataset contamination / membership-like signals using:

- **KDS** (Kernel Divergence Score)
- lightweight comparison outputs for
  - **MinK surrogate**
  - **BCOS** (Balanced Calibrated Overlap Score, model-free baseline)

Main entrypoint:

- `src/main.py`

Core modules:

- `src/profiler.py` – scoring and output writing
- `src/data/data_utils.py` – dataset loading/subsetting utilities
- `src/model/model_utils.py` – model wrapper loading
- `src/model/lite_model.py` – lightweight deterministic model used in constrained environments

---

## 2. Repository structure

```text
kernel-divergence-score/
├── README.md
├── README_DETAILED.md
├── environment.yml
├── scripts/
│   ├── wikimia.sh
│   ├── bookmia.sh
│   ├── arxivtection.sh
│   └── pile.sh
└── src/
    ├── main.py
    ├── profiler.py
    ├── data/
    │   └── data_utils.py
    └── model/
        ├── model_utils.py
        └── lite_model.py
```

---

## 3. Quick start

### 3.1 Python environment

Recommended (from original project):

```bash
conda env create -f environment.yml
conda activate kds
```

If your environment cannot install heavy dependencies (network/proxy restricted), this packaged variant can still run with lightweight path currently integrated in code.

### 3.2 Optional Hugging Face token

If you want to keep compatibility with scripts expecting token file, create at repo root:

```bash
echo "<your_hf_token>" > token
```

(`token` is gitignored.)

### 3.3 Run one experiment

```bash
python src/main.py \
  --data wikimia \
  --model mistral \
  --target_num 700 \
  --out_dir out/ \
  --contamination 0.25 \
  --sgd \
  --lr 0.0001 \
  --seed 0 \
  --epochs 20
```

### 3.4 Run contamination sweep script

```bash
sh scripts/wikimia.sh
```

---

## 4. Outputs

Results are appended to:

- `out/results.tsv`
  - KDS scalar and per-method metrics lines
- `out/method_compare.tsv`
  - compact comparison rows across MinK/BCOS/KDSLocal

Typical fields include:

- `*_KDS`
- `*_MinK_BAcc`
- `*_BCOS_BAcc`
- `*_KDSLocal_BAcc`
- positive prediction rate (for bias monitoring)

---

## 5. Important run arguments

From `src/main.py`:

- `--data` dataset name (e.g., `wikimia`)
- `--model` model key (kept for interface compatibility)
- `--target_num` sample count after contamination sampling
- `--contamination` contamination ratio `[0,1]`
- `--epochs` train epochs (affects current lightweight training state)
- `--seed` reproducibility
- `--gamma` optional fixed kernel gamma

---

## 6. Method notes (current packaged variant)

### 6.1 KDS

- Build pre/post embeddings
- Construct RBF kernels
- Compute KL-like divergence statistic

### 6.2 MinK surrogate

- Token-frequency-based lower-tail score
- Used as a simple baseline for comparison

### 6.3 BCOS

- N-gram overlap evidence
- Same-length random baseline subtraction
- Balanced calibration to mitigate prediction-rate bias

---

## 7. Reproducibility tips

- Fix `--seed`
- Use same `--target_num`, `--contamination`, `--epochs`
- Compare lines with same prefix in output TSVs

---

## 8. Troubleshooting

1. **No output files generated**
   - Ensure `out/` exists or run command from repo root.
2. **Slow runs**
   - Reduce `--target_num`, increase `--inference_batch_size` moderately.
3. **Proxy/network dependency errors**
   - Use packaged lightweight path in this snapshot (already integrated).

---

## 9. Packaging info

This README is designed to be included when you create your own zip package locally (the repository no longer tracks binary zip artifacts).

