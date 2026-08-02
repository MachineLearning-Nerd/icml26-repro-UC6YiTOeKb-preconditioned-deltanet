# Claim 1 — exact PDN/PLA recurrence equivalence

**Exact live claim.** Theorem 3.1 establishes that the Preconditioned DeltaNet (PDN) recursion satisfies `S_t = C_t P_t` for all `t`, deriving the equivalence between linear attention and the delta rule under exact preconditioning.

**Verdict: toy.** This is a local finite, clean-room float64 recurrence check, not an independent proof for arbitrary sequences.

## Method and source pin

The exact update and theorem are retained verbatim in [`evidence/source/claim1_method_excerpt.tex`](../evidence/source/claim1_method_excerpt.tex), with source archive/PDF hashes in [`evidence/source/SHA256SUMS`](../evidence/source/SHA256SUMS). The clean-room implementation is [`src/claim1_exact_preconditioning.py`](../src/claim1_exact_preconditioning.py). It uses five fixed seeds (`17,29,43,71,101`), key dimension 6, value dimension 4, 30 tokens, `G_0=I`, and float64. The PDN update is evaluated online while `C_t P_t` is independently reconstructed from accumulated `C_t`, `G_t`, and a direct inverse.

## Commands and results

```bash
python src/claim1_exact_preconditioning.py --out outputs/claim1_exact_preconditioning
python -m pytest -q
(cd outputs/claim1_exact_preconditioning && sha256sum -c SHA256SUMS)
```

(For this host, the pinned local environment is declared in `requirements.txt`.) The maximum exact state residual is `7.22e-16` and maximum query-output residual is `2.66e-15`, below the predeclared `1e-10` threshold. Raw rows, compressed fixture residuals, configuration, run log, summary, and checksums are in [`outputs/claim1_exact_preconditioning/`](../outputs/claim1_exact_preconditioning/).

## Negative control and limitation

Replacing the full inverse-Gram write-key preconditioner with its diagonal produced a minimum state residual of `0.5055` across the same seeds. This expected failure is outside the theorem's exact-preconditioning premise. The finite synthetic fixture does not prove Theorem 3.1 universally or address claims 2–5.
