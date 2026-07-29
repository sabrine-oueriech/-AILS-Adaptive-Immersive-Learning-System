# AILS — Adaptive Immersive Learning System

Companion code for the paper:
> **AILS: An Adaptive Immersive Learning System with Real-Time Cognitive Load Estimation, LLM Exercise Generation, and SHAP-Based Explainability**

---

## Repository structure

```
AILS.ipynb          Main notebook (all pipeline cells)
README.md           This file
.gitignore          Excludes main.tex (article source, private)
```

---

## Cell execution order

| Cell | Role | Article section |
|------|------|----------------|
| 1–3  | Imports & API setup | — |
| 4–6  | Google Sheets authentication | §4.3 |
| 7–10 | CL model (Eq. 1–3), fusion weights (Eq. 3) | §3.1 |
| 11–12 | CF recommender (cosine similarity) | §3.2.1 |
| 13   | Constants, MODULES, POPULATION_CF/SYN, verify_exercise | §3.2.0.4 / §3.3 |
| 14–16 | RAG context builder | §3.2.0.2 |
| 17   | SHAP model (GBR + KernelExplainer, Table 4 mapping) | §3.2.2 |
| 18   | SAFE_TEMPLATES fallback library | §3.3 |
| 19   | *(empty — was duplicate of Cell 17, now removed)* | — |
| 20   | Form 1 / Form 2 parsers | §4.2 |
| 21   | Google Sheets live analysis (builds df_l) | §7 |
| 22–23 | SHAP explainability plots | §3.2.2 |
| 24   | Experiment configuration (EXP_STATE, constants) | §4.3 |
| 25   | Full pipeline: generate_exercise_rag, _check_mastery, Gradio UI | §3–§4 |
| 26   | Launch Gradio interface | §4.3 |
| 27   | Statistical analysis (Wilcoxon, CL t-test, Pearson) | §7.2–§7.3 |
| 28   | Wilcoxon signed-rank test — dynamic from real data | Table 17 / §7.2 |

> **Run Cell 21 before Cells 27–28** to build `df_l` from the live Google Sheet.

---

## Dependencies

```bash
pip install gradio mistralai google-auth google-auth-oauthlib \
            gspread pandas numpy scipy scikit-learn shap matplotlib seaborn
```

Runtime: **Google Colab** (recommended) or local Python 3.10+.

---

## Credentials (required)

Set the following Colab secrets before running:

| Secret key | Value |
|---|---|
| `MISTRAL_API_KEY` | Your Mistral AI key |
| `GOOGLE_SERVICE_ACCOUNT_CREDENTIALS` | JSON content of your GCP service account |

Never hardcode credentials — the notebook reads them via `userdata.get(...)`.

---

## Key design choices mapped to article equations

| Code | Equation | Description |
|---|---|---|
| `estimate_cl()` | Eq. 1 | CL = α·ĥ + β·ĝ + γ·ê + δ·r̂ |
| `get_fusion_weights()` | Eq. 3 | Dynamic weight schedule (cold-start / CL-critical / intra-rich) |
| `score_evolution()` | Eq. (§5) | S(t+1) = S(t) + η·(1−S(t))·(1−CL_t)·ξ, η=0.18, ξ~U(0.8,1.2) |
| `_check_mastery()` | Eq. 2 | Per-module threshold θ_m (MODULES[mid]['theta']) |
| `FEATURE_TO_CHANNEL` | Table 4 | score_norm→ĥ, tendance→ĝ, erreur_norm→ê, cl_value→r̂ |
| `_verify_exercise()` | §3.3 | κ≥0.70 deliver / [0.60,0.70) retry / <0.60 fallback |

---

## Cold-start note

The collaborative filtering recommender uses a **hybrid** profile database:
- **Real profiles**: n=34 classroom learners
- **Synthetic pre-training**: N=100 profiles (`POPULATION_CF`) calibrated to match real cohort proportions

This hybrid mode is required to reach Recall@3≥0.85 at the current cohort size (n=34). Results are reported separately for the real-only and real+synthetic conditions in Table CF of the article (§7.4).

---

## Reproducibility

- Mistral model pinned: `mistral-medium-2312` (snapshot 15 March 2025)
- Random seed: 42 (synthetic simulation)
- All statistics computed dynamically from the live Google Sheet via Cell 21
