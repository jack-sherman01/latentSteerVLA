# HPC / SLURM Usage Guide for CompSteer

## Pipeline Overview

Run the phases in order. Phases 0–1 are array jobs (one GPU per pair), phases 2–5 are single-node jobs.

```
Phase 0  → 01_collect_demos.slurm      [array, ~1h/pair]
Phase 1  → 02_train_vae.slurm          [array, ~30min/pair]
Phase 2a → python scripts/03_extract_vectors.py    [CPU, <5min]
Phase 2b → 04_verify_linearity.slurm   [1 GPU, ~30min]
Phase 3  → python scripts/05_train_encoders.py     [1 GPU, ~1h]
Phase 4  → 06_eval_compsteer.slurm     [1 GPU, ~4-8h]
Phase 5  → ablation: multiple 06_eval calls
```

---

## Quick Start

### 1. Setup environment

```bash
conda create -n compsteer python=3.10
conda activate compsteer

# Install dependencies
pip install -r requirements.txt
pip install git+https://github.com/NVIDIA/Isaac-GR00T.git
pip install git+https://github.com/TeleHuman/Align-Then-Steer.git

# Install this package
pip install -e .
```

### 2. Collect demos (Phase 0)

```bash
mkdir -p logs
sbatch slurm/01_collect_demos.slurm
# Monitor: squeue -u $USER
# Check output: tail -f logs/collect_*.out
```

### 3. Train VAEs (Phase 1)

```bash
# Wait for Phase 0 to finish
sbatch slurm/02_train_vae.slurm
```

### 4. Extract steering vectors (Phase 2a)

```bash
# Run on login node or submit as a short CPU job
python scripts/03_extract_vectors.py
```

### 5. Verify linearity — GO/NO-GO (Phase 2b)

```bash
sbatch slurm/04_verify_linearity.slurm
# Check: results/linearity/linearity_results.json
# Key metric: SVD rank-16 residual < 0.20 → proceed
```

### 6. Train retrieval encoders (Phase 3)

```bash
python scripts/05_train_encoders.py --rank 16
```

### 7. Main evaluation (Phase 4)

```bash
# GR00T N1-2B (primary)
sbatch slurm/06_eval_compsteer.slurm groot svd 16 cosine

# RDT-1B (secondary backbone)
sbatch slurm/06_eval_compsteer.slurm rdt1b svd 16 cosine

# GR00T N1-1B (ablation)
sbatch slurm/06_eval_compsteer.slurm groot_1b svd 16 cosine
```

### 8. Ablation sweep (Phase 5)

```bash
# Rank ablation
for RANK in 4 8 16 32; do
    sbatch slurm/06_eval_compsteer.slurm groot svd $RANK cosine
done

# Schedule ablation
for SCHED in constant cosine linear early_only; do
    sbatch slurm/06_eval_compsteer.slurm groot svd 16 $SCHED
done

# Factorization method ablation
for METHOD in svd nmf learned; do
    sbatch slurm/06_eval_compsteer.slurm groot $METHOD 16 cosine
done
```

---

## Customising for Your Cluster

### Partition names

Update `#SBATCH --partition=gpu` in each `.slurm` file to match your HPC's GPU partition name.

### Memory & time

| Job | Recommended | Notes |
|-----|------------|-------|
| collect_demos | 32G, 2h | 8 parallel envs; increase time if oracle policy is slow |
| train_vae | 16G, 2h | CPU-bound; single A100 more than enough |
| verify_linearity | 16G, 1h | Fast; SVD is closed-form |
| eval_compsteer | 64G, 8h | GR00T 2B needs ~20GB VRAM; 8 parallel envs |

### Multi-node eval

If your cluster requires multi-GPU for large models, modify `06_eval_compsteer.slurm`:
```bash
#SBATCH --gres=gpu:4
# Then add to python call:
--device cuda
```
CompSteer inference itself is single-GPU; the extra GPUs speed up parallel ManiSkill envs.

---

## Expected Runtimes (A100 80GB)

| Phase | Wall time |
|-------|-----------|
| Demo collection (all 24 pairs, parallel) | ~1h |
| VAE training (all 24 pairs, parallel) | ~30min |
| Vector extraction | 5min |
| Linearity verification | 30min |
| Encoder training | 1h |
| Main eval (splits A-D, GR00T 2B) | 6-8h |
| Ablation sweep (12 conditions) | ~3h (parallel) |

---

## Output Structure

```
results/
    eval/
        groot/
            svd_rank16/
                schedule_cosine/
                    results.json       ← success rates per (e, task, split)
                    videos/            ← optional rollout videos
    ablation/
        rank/
            4/   8/   16/   32/
        schedule/
            constant/   cosine/   ...
    linearity/
        linearity_results.json
        linearity_curve.png

steering_library/
    vectors/
        panda__pick_cube.pt
        ...
    library.pt
    factorizations/
        svd/
            rank_4.pt   rank_8.pt   rank_16.pt   rank_32.pt
        nmf/
            ...

checkpoints/
    vae/
        panda__pick_cube/
            vae_best.pt
    encoders/
        rank_16/
            f_emb.pt
            g_lang.pt
```
