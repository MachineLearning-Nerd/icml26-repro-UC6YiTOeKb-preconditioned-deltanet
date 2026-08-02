# Claim 1 protocol — PDN / PLA recurrence equivalence

**Exact live claim:** Theorem 3.1 establishes that the Preconditioned DeltaNet (PDN) recursion satisfies `S_t = C_t P_t` for all `t`, formally deriving the equivalence between linear attention and the delta rule under exact preconditioning (Section 3.1, Theorem 3.1).

**Pinned source location:** `evidence/source/claim1_method_excerpt.tex`, extracted verbatim from `main_material/3_method.tex` in the pinned arXiv archive. The source states the exact inverse-key-Gram setting and cautions that the equivalence no longer holds for approximate key-Gram preconditioners.

**Pre-registered local route:** independently implement the exact finite recurrence on full-rank key matrices for several fixed seeds; compute both sides without shared state; require numerical residual at float64 tolerance. A singular/diagonal approximate-preconditioner control must fail equality where the source says the theorem does not apply. Retain seeds, commands, outputs, and hashes.

**Verdict rule:** only a direct controlled recurrence result may support a finite scoped outcome. It cannot by itself prove the theorem for arbitrary sequence length or establish throughput/benchmark claims.
