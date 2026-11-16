# Dataset Selection Criteria
**Project:** Agentic Data Science Failure Analysis
**Last updated:** 2026-05-22
**Target:** ~150 curated tasks

---

## 1. Task Taxonomy

### Primary Task Types

| Code | Name | Description | Example Metrics |
|------|------|-------------|-----------------|
| `BC` | Binary Classification | Two-class prediction | ROC-AUC, Log Loss, F1, Accuracy |
| `MC` | Multi-class Classification | 3–20 discrete classes | Accuracy, Macro-F1, Log Loss |
| `FG` | Fine-grained Classification | 20+ discrete classes | Top-K Accuracy, Macro-F1 |
| `RG` | Regression | Single continuous target | RMSE, MAE, RMSLE, R² |
| `OR` | Ordinal Regression | Ordered discrete output (e.g., star ratings) | QWK, RMSE |
| `MTR` | Multi-target Regression | Multiple continuous targets | Average RMSE/MAE |
| `MTC` | Multi-target Classification | Multiple label columns | MAP, Mean F1 |
| `TS` | Time Series Forecasting | Predict future timesteps | SMAPE, RMSE, MAE |
| `RANK` | Ranking / Recommendation | Order items by relevance | MAP@K, NDCG, AP@K |

### Modality Types

| Code | Meaning |
|------|---------|
| `Tab` | Tabular / structured (CSV) |
| `Txt` | Text / NLP features |
| `Img` | Image data |
| `Multi` | Combination of the above |

### Domain Taxonomy

| Domain | Sub-domains |
|--------|------------|
| **Healthcare & Life Sciences** | Disease prediction, clinical trials, drug approval, medical imaging, bioinformatics |
| **Finance & Economics** | Credit risk, insurance, housing prices, stock/loan prediction |
| **Natural Language** | Sentiment, toxicity, topic classification, review rating |
| **Retail & E-commerce** | Product pricing, demand forecasting, customer churn |
| **Environment & Climate** | Natural disasters, energy consumption, environmental monitoring |
| **Transportation** | Traffic, vehicle classification, ride-sharing demand |
| **Social & Demographics** | Crime, education, employment, population statistics |
| **Science & Research** | Materials, biology, physics challenges, genomics |
| **Infrastructure & Government** | Building permits, grants, public datasets |
| **Computer Vision** | Image classification/regression (secondary priority for this project) |

---

## 2. Selection Criteria

### Hard Filters — Must satisfy ALL

| # | Criterion | Rule |
|---|-----------|------|
| H1 | **Modality** | Primarily tabular (CSV/structured). Text-tabular hybrids accepted. Pure CV/NLP only included if it has a structured prediction target. |
| H2 | **Task clarity** | Single or clearly defined prediction target(s). Evaluation must be fully automated (no human scoring). |
| H3 | **Accessibility** | Publicly downloadable without special institutional approval. Kaggle API, HuggingFace, or direct URL access. |
| H4 | **Minimum scale** | ≥ 500 training samples. |
| H5 | **Maximum scale** | ≤ 5,000,000 rows total (train + test). Keeps agent runs computationally feasible. |
| H6 | **Metric** | Standard automated evaluation metric (RMSE, AUC, Accuracy, F1, etc.). No subjective scoring. |
| H7 | **License** | Research-permissive: CC-BY, CC0, public domain, or Kaggle ToS (for research/non-commercial use). |
| H8 | **Reproducibility** | Fixed train/test split or defined split strategy available. Leaderboard or held-out ground truth accessible. |

### Soft Filters — Prefer, but not strict

| # | Criterion | Preference |
|---|-----------|------------|
| S1 | **Kaggle participation** | ≥ 100 teams on leaderboard (quality signal, richer human baseline) |
| S2 | **Competition category** | Getting Started / Playground / Featured / Research over pure Community personal uploads |
| S3 | **Feature richness** | 5–200 feature columns. Avoids trivially small or overwhelmingly wide datasets. |
| S4 | **Domain diversity** | Spread across ≥ 7 distinct domains across the full 150-task set |
| S5 | **Task diversity** | Balanced across task types per targets in Section 3 |
| S6 | **Temporal coverage** | Mix of pre-2022 and post-2022 competition release dates (controls for LLM contamination) |
| S7 | **Difficulty range** | Include a range from beginner ("Getting Started") to expert-level (Featured/Research with prize pools) |
| S8 | **Data quality** | Reasonably clean or documented quality issues. Avoid datasets with known severe corruption or missing target. |

### Exclusion Rules

| Reason | Rule |
|--------|------|
| Pure image or audio classification | Exclude unless it has accompanying tabular metadata for the prediction |
| Duplicate/near-duplicate tasks | If two datasets represent the same underlying data with minor wording changes, keep only one |
| Synthetic/toy datasets only | Exclude if the data is entirely simulated with no real-world component |
| Requires live API or real-time data | Exclude (agents cannot reproduce the original conditions) |
| Ambiguous target | Exclude if the label column definition is unclear or contested |

---

## 3. Diversity & Balance Targets (~150 tasks)

| Task Type | Target Count | Notes |
|-----------|-------------|-------|
| Binary Classification (`BC`) | 35 | Include both balanced and imbalanced cases |
| Multi-class Classification (`MC`/`FG`) | 25 | Range from 3-class to 100+ class |
| Regression (`RG`) | 45 | Largest category; includes housing, prices, scores |
| Time Series Forecasting (`TS`) | 15 | Temporal structure; demand, energy, weather |
| Multi-target Regression/Classification (`MTR`/`MTC`) | 10 | Tests agent ability to handle multi-output |
| NLP-Tabular Hybrid (`Tab+Txt`) | 10 | Text features + structured metadata |
| Ordinal / Ranking (`OR`/`RANK`) | 10 | Rating prediction, recommendation |
| **Total** | **150** | |

### Domain Balance Targets

| Domain | Target Count |
|--------|-------------|
| Finance & Economics | 20 |
| Healthcare & Life Sciences | 20 |
| Retail & E-commerce | 15 |
| Environment & Climate | 15 |
| Social & Demographics | 15 |
| Natural Language (hybrid) | 10 |
| Science & Research | 15 |
| Transportation | 10 |
| Infrastructure & Government | 15 |
| Other / Mixed | 15 |

---

## 4. Source Provenance

All datasets must have a recorded provenance entry in `dataset_registry.md` with the following fields:

| Field | Description |
|-------|-------------|
| `id` | Short slug identifier |
| `source_paper` | Reference paper that used or listed this dataset (`AgentK`, `STRABLE`, `new`) |
| `origin` | Where the data comes from (`Kaggle`, `HuggingFace`, `UCI`, `OpenML`, `Other`) |
| `url` | Canonical URL for the dataset |
| `task_type` | One of the codes in Section 1 |
| `modality` | `Tab`, `Txt`, `Img`, or `Multi` |
| `domain` | Domain tag from Section 1 |
| `metric` | Primary evaluation metric |
| `n_rows` | Approximate number of training rows |
| `n_features` | Approximate number of feature columns |
| `kaggle_category` | For Kaggle only: `GS`, `P`, `F`, `C`, `RS` |
| `status` | `included`, `excluded`, `pending` |
| `notes` | Any caveats (imbalance, known issues, LLM contamination risk, etc.) |

---

## 5. Criteria Version History

| Version | Date | Changes |
|---------|------|---------|
| v0.1 | 2026-05-22 | Initial criteria drafted from Agent-K and STRABLE paper review |
