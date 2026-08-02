# Independent local reproduction — Preconditioned DeltaNet

OpenReview ID: `UC6YiTOeKb`  
Paper: *Preconditioned DeltaNet: Curvature-aware Sequence Modeling for Linear Recurrences*.

This repository is an independent clean-room reproduction. It uses local CPU/GPU only; it does **not** use Hugging Face cpu-upgrade, Jobs, paid, or remote compute. See `STATUS.md`, `contract/`, and `evidence/source/` for the pinned live contract and source inputs.

## Claim 1 checkpoint

A five-seed, finite clean-room float64 PDN/PLA recurrence check is retained at `outputs/claim1_exact_preconditioning/`. Exact residuals are below `1e-10`; a diagonal-preconditioner negative control fails as expected. This is explicitly a **toy** outcome, not a universal theorem proof; see `logbook/claim-1.md`.
