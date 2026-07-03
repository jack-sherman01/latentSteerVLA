"""
CompSteer: Compositional Inference-Time Latent Steering for Zero-Shot VLA Generalization.

Pipeline overview:
    Phase 0  – collect ManiSkill demos per (embodiment, task) pair
    Phase 1  – train asymmetric VAE per pair (ATE align stage, extended to GR00T)
    Phase 2  – extract + factorize steering vectors  →  Δe_i, Δt_j library
    Phase 3  – train lightweight retrieval encoders f_emb, g_lang
    Phase 4  – zero-shot eval: compose Δe + Δt, inject at inference
    Phase 5  – ablations + geometric analysis
"""

__version__ = "0.1.0"
