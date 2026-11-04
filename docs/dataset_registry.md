# Dataset Registry
**Project:** Agentic Data Science Failure Analysis
**Last updated:** 2026-05-27
**Criteria reference:** See `dataset_criteria.md`

---

## Summary

| Source | Benchmark size | Tasks fetched | Non-open-license |
|--------|---------------|---------------|-----------------|
| STRABLE (arXiv 2605.12292) | 108 | 108 | 0 |
| Kaggle (curated) | N/A | 36 | 36 |
| MLE-bench (arXiv 2410.07095) | 75 | 8 | 8 |
| OpenML | N/A | 38 | 0 |
| UCI ML Repository | N/A | 24 | 0 |
| **Total** | — | **214** | **44** |

Selection criteria: tabular modality only (no images, audio, or pure text); supervised prediction task (regression, binary classification, or multiclass classification); minimum 100 training rows after 80/20 split; for Kaggle and MLE-bench, training data publicly accessible with competition terms accepted and test labels present in the downloadable training split. Tasks failing these criteria or with permanently inaccessible data were excluded.

### Publication strategy (for open sourcing)
- **Ship freely**: all code, scripts, `task.json`, `description.md`, `evaluation/metric.py`, `sample_submission.csv`
- **Do not ship**: `data/train.parquet`, `data/test.parquet`, `evaluation/labels.parquet` for Kaggle/MLE-bench tasks
- **For STRABLE**: reference `inria-soda/STRABLE-benchmark` on HuggingFace; users fetch via `fetch_strable.py`
- **For Kaggle/MLE-bench**: users run `fetch_kaggle.py` with their own API key (they accept terms themselves)
- **For OpenML/UCI**: can include data directly or have users run fetch scripts (either is fine)

### Task-type coverage (actual — from fetched task.json files)

| Task type | STRABLE | Kaggle (curated) | MLE-bench | OpenML | UCI | Total |
|-----------|---------|-----------------|-----------|--------|-----|-------|
| Regression | 76 | 16 | 6 | 11 | 12 | **121** |
| Binary Classification | 13 | 14 | 2 | 16 | 7 | **52** |
| Multi-class Classification | 19 | 6 | 0 | 11 | 5 | **41** |
| **Total** | **108** | **36** | **8** | **38** | **24** | **214** |

### Domain distribution (all 214 tasks)

| Domain | Count |
|--------|-------|
| Healthcare | 46 |
| Finance | 45 |
| Science | 26 |
| Retail | 23 |
| Social | 22 |
| Energy | 16 |
| Transportation | 14 |
| Education | 11 |
| Environment | 8 |
| Mixed | 3 |

### Dataset statistics (across all 214 tasks)

**Rows**

| Stat | Value |
|------|-------|
| Min | 137 (restaurant-revenue-prediction) |
| Median | 8,665 |
| Mean | 507,617 |
| Max | 55,423,856 (new-york-city-taxi-fare-prediction) |

**Columns (features, excluding target)**

| Stat | Value |
|------|-------|
| Min | 3 |
| Median | 18 |
| Mean | 53 |
| Max | 1,776 (bioresponse) |

**Modality**

| Modality | Count |
|----------|-------|
| Tabular only | 209 |
| Tabular + text columns | 5 |

**Number of tables**

All 214 tasks are presented to agents as a single train/test parquet pair. 7 tasks originate from multi-table competitions and were flattened to the primary table: home-credit-default-risk (7), store-sales-time-series-forecasting (6), ashrae-energy-prediction (5), competitive-data-science-predict-future-sales (3), walmart-recruiting-store-sales-forecasting (3), airbnb-recruiting-new-user-bookings (2), petfinder-adoption-prediction (2).

---

## Registry Table

Fields: `id` | `source_paper` | `origin` | `url` | `task_type` | `modality` | `domain` | `metric` | `n_rows` | `n_features` | `kaggle_category` | `status` | `notes`

---

### A — Agent-K Competitions (81 total, from arxiv 2411.03562)

Source paper: **AgentK**
Origin: **Kaggle**
Base URL: `https://www.kaggle.com/competitions/`

| id | task_type | modality | domain | metric | n_rows | n_features | kaggle_category | status | notes |
|----|-----------|----------|--------|--------|--------|------------|-----------------|--------|-------|
| nlp-getting-started | BC | Txt | Natural Language | F1 Score | ~10k | text | GS | pending | NLP — Titanic-style for NLP |
| sentiment-analysis-on-movie-reviews | MC | Txt | Natural Language | Accuracy | ~8.5k | text | P | pending | 5-class sentiment (Rotten Tomatoes) |
| dogs-vs-cats-redux-kernels-edition | BC | Img | Computer Vision | Log Loss | ~25k | image | P | pending | CV; include only if multimodal version needed |
| dog-breed-identification | MC | Img | Computer Vision | Log Loss | ~10k | image | P | pending | 120 dog breeds; pure CV |
| state-farm-distracted-driver-detection | MC | Img | Computer Vision | Log Loss | ~22k | image | F | pending | Pure CV |
| histopathologic-cancer-detection | BC | Img | Healthcare | ROC AUC | ~220k | image | P | pending | Medical imaging; pure CV |
| whereami | MC | Img | Computer Vision | Accuracy | ~? | image | C | pending | Community; pure CV |
| 5-flowers-image-classification | MC | Img | Computer Vision | Accuracy | ~3.6k | image | C | pending | Community; pure CV |
| sign-language-image-classification | MC | Img | Computer Vision | Accuracy | ~? | image | C | pending | Community; pure CV |
| ml-olympiad-landscape-image-classification | MC | Img | Environment | Accuracy | ~? | image | C | pending | Community; pure CV |
| noaa-right-whale-recognition | MC | Img | Science | Accuracy | ~4k | image | RS | pending | Conservation; pure CV |
| humpback-whale-identification | MC | Img | Science | MAP@5 | ~25k | image | F | pending | Few-shot; pure CV |
| planttraits2024 | MTR | Tab+Img | Science | R² | ~? | mixed | RS | pending | Multimodal: image + tabular plant traits |
| invasive-species-monitoring | BC | Img | Environment | ROC AUC | ~2.3k | image | P | pending | Pure CV |
| tabular-playground-series-feb-2021 | RG | Tab | Retail | RMSE | ~300k | 10 | P | pending | Synthetic tabular; playground series |
| playground-series-s3e11 | RG | Tab | Finance | RMSE | ~? | ~10 | P | pending | Playground series S3 |
| playground-series-s3e14 | RG | Tab | Finance | RMSE | ~? | ~10 | P | pending | Playground series S3 |
| playground-series-s3e15 | RG | Tab | Finance | RMSE | ~? | ~10 | P | pending | Playground series S3 |
| home-data-for-ml-course | RG | Tab | Finance | RMSE | ~1.5k | 79 | GS | pending | House prices for ML course; classic starter |
| bike-sharing-demand | RG | Tab | Transportation | RMSLE | ~10.9k | 12 | P | pending | Time-structured demand prediction |
| tabular-playground-series-aug-2021 | RG | Tab | Mixed | RMSE | ~1M | ~100 | P | pending | Synthetic; large scale |
| forest-cover-type-prediction | MC | Tab | Environment | Accuracy | ~15k | 54 | P | pending | 7-class; cartographic features |
| mercedes-benz-greener-manufacturing | RG | Tab | Transportation | R² | ~4.2k | ~380 | F | pending | High-dimensional; feature selection challenge |
| goodreads-books-reviews-290312 | RG | Tab+Txt | Retail | RMSE | ~290k | mixed | P | pending | Text + metadata; book ratings |
| sberbank-russian-housing-market | RG | Tab | Finance | RMSLE | ~30k | ~292 | F | pending | Housing prices; many features |
| restaurant-revenue-prediction | RG | Tab | Retail | RMSE | ~137 | 41 | F | pending | Very small training set — interesting challenge |
| scrabble-player-rating | RG | Tab | Social | MAE | ~1.9M | ~30 | P | pending | Large scale; player skill rating |
| nlp1000-ml-challenge | BC | Txt | Natural Language | Accuracy | ~? | text | C | pending | Community NLP challenge |
| nlpsci | MC | Txt | Science | Accuracy | ~? | text | C | pending | Community NLP challenge |
| sf-crime | MC | Tab | Social | Log Loss | ~878k | 9 | F | pending | Crime category prediction; San Francisco |
| otto-group-product-classification-challenge | MC | Tab | Retail | Log Loss | ~61k | 93 | F | pending | 9-class product type classification |
| nitro-language-processing-3 | BC | Txt | Natural Language | Accuracy | ~? | text | C | pending | Community NLP |
| sbu-ai-finalproject | MC | Txt | Natural Language | Accuracy | ~? | text | C | pending | Community; small |
| avito-prohibited-content | BC | Tab+Txt | Retail | ROC AUC | ~4.2M | mixed | F | pending | Ad content moderation; very large |
| playground-series-s3e1 | BC | Tab | Finance | ROC AUC | ~? | ~10 | P | pending | S3 binary classification |
| playground-series-s3e9 | BC | Tab | Mixed | ROC AUC | ~? | ~10 | P | pending | S3 binary classification |
| playground-series-s3e16 | RG | Tab | Mixed | RMSE | ~? | ~10 | P | pending | S3 regression |
| playground-series-s3e25 | RG | Tab | Mixed | RMSE | ~? | ~10 | P | pending | S3 regression |
| tabular-playground-series-jan-2021 | RG | Tab | Mixed | RMSE | ~300k | 14 | P | pending | Synthetic tabular |
| tabular-playground-series-aug-2022 | BC | Tab | Mixed | ROC AUC | ~? | ~10 | F | pending | Larger playground |
| tabular-playground-series-apr-2021 | RG | Tab | Mixed | RMSE | ~? | ~10 | P | pending | Synthetic tabular |
| playground-series-s4e4 | RG | Tab | Mixed | RMSE | ~? | ~10 | P | pending | S4; ended May 2024 — post-LLM |
| ventilator-pressure-prediction | RG | Tab | Healthcare | MAE | ~6.5M | 8 | RS | pending | Time series structure; medical ventilator |
| playground-series-s4e5 | RG | Tab | Mixed | RMSE | ~? | ~10 | P | pending | S4 regression |
| porto-seguro-safe-driver-prediction | BC | Tab | Finance | Norm. Gini | ~595k | 57 | F | pending | Insurance; imbalanced classes |
| playground-series-s3e3 | BC | Tab | Mixed | ROC AUC | ~? | ~10 | P | pending | S3 binary |
| santander-customer-satisfaction | BC | Tab | Finance | ROC AUC | ~76k | ~370 | F | pending | Customer churn; anonymous features |
| playground-series-s3e5 | BC | Tab | Mixed | ROC AUC | ~? | ~10 | P | pending | S3 binary |
| cat-in-the-dat | BC | Tab | Mixed | ROC AUC | ~300k | ~24 | P | pending | Categorical encoding challenge |
| playground-series-s4e1 | RG | Tab | Mixed | RMSE | ~? | ~10 | P | pending | S4 regression |
| tabular-playground-series-sep-2022 | BC | Tab | Mixed | ROC AUC | ~? | ~10 | P | pending | Synthetic tabular |
| cat-in-the-dat-ii | BC | Tab | Mixed | ROC AUC | ~200k | ~46 | P | pending | Categorical encoding challenge v2 |
| playground-series-s3e22 | BC | Tab | Mixed | ROC AUC | ~? | ~10 | P | pending | S3 binary |
| playground-series-s3e26 | BC | Tab | Mixed | ROC AUC | ~? | ~10 | P | pending | S3 binary |
| playground-series-s3e7 | RG | Tab | Mixed | RMSE | ~? | ~10 | P | pending | S3 regression |
| jigsaw-toxic-comment-classification-challenge | MTC | Txt | Natural Language | Mean AUC | ~160k | text | F | pending | Multi-label toxicity; NLP |
| kaggle-pog-series-s01e03 | MC | Tab | Mixed | Accuracy | ~? | ~10 | C | pending | Community tabular |
| vehicle | MC | Tab | Transportation | Accuracy | ~? | ~15 | C | pending | Community; vehicle classification |
| plant-seedlings-classification | MC | Img | Science | Accuracy | ~4.7k | image | P | pending | Pure CV |
| sp-society-camera-model-identification | BC | Img | Computer Vision | Accuracy | ~? | image | F | pending | Camera fingerprinting; pure CV |
| plant-pathology-2020-fgvc7 | MC | Img | Science | ROC AUC | ~1.8k | image | RS | pending | Plant disease; pure CV |
| world-championship-2023-embryo-classification | BC | Img | Healthcare | Accuracy | ~? | image | C | pending | Medical imaging; pure CV |
| leaf-classification | MC | Tab+Img | Science | Log Loss | ~990 | 192 | P | pending | Tabular leaf features; usable without images |
| nitro-language-processing-2 | MC | Txt | Natural Language | Accuracy | ~? | text | C | pending | Community NLP |
| playground-series-s3e24 | BC | Tab | Mixed | ROC AUC | ~? | ~10 | P | pending | S3 binary |
| playground-series-s4e2 | MC | Tab | Mixed | Log Loss | ~? | ~10 | P | pending | S4 multiclass |
| tmdb-box-office-prediction | RG | Tab+Txt | Retail | RMSLE | ~3k | mixed | P | pending | Movie metadata + text; box office revenue |
| playground-series-s3e19 | RG | Tab | Mixed | RMSE | ~? | ~10 | P | pending | S3 regression |
| tabular-playground-series-may-2022 | BC | Tab | Mixed | ROC AUC | ~? | ~10 | P | pending | Synthetic tabular |
| tabular-playground-series-jan-2022 | RG | Tab | Mixed | RMSE | ~? | ~10 | P | pending | Synthetic tabular |
| dont-overfit-ii | BC | Tab | Mixed | ROC AUC | ~250 | 300 | P | pending | Overfitting challenge; very few rows |
| statoil-iceberg-classifier-challenge | BC | Tab+Img | Environment | Log Loss | ~1.6k | SAR image | F | pending | Radar image data; multimodal |
| diabetic-retinopathy-detection | MC | Img | Healthcare | QWK | ~35k | image | F | pending | Ordinal; medical imaging |
| stumbleupon | BC | Tab+Txt | Retail | ROC AUC | ~7.4k | mixed | F | pending | Web page classification; text + metadata |
| h2oai-predict-the-llm | MTC | Txt | Science | SMAPE | ~? | text | F | pending | Predict LLM benchmark scores; NLP |
| human-protein-atlas-image-classification | MC | Img | Healthcare | Mean F1 | ~31k | image | F | pending | Multi-label protein classification; pure CV |
| predict-who-is-more-influential-in-a-social-network | BC | Tab | Social | Accuracy | ~5.8k | 12 | F | pending | Social network features; tabular |
| bioresponse | BC | Tab | Healthcare | ROC AUC | ~3.7k | 1776 | P | pending | High-dimensional biological features |
| GiveMeSomeCredit | BC | Tab | Finance | ROC AUC | ~150k | 10 | P | pending | Credit default; imbalanced classes |
| galaxy-zoo-the-galaxy-challenge | MTC | Img | Science | RMSE | ~61k | image | F | pending | Galaxy morphology; pure CV |
| playground-series-s5e5 | RG | Tab | Mixed | RMSE | ~? | ~10 | P | pending | S5; ended June 2025 — most recent |

---

### B — STRABLE Datasets (108 total, from arxiv 2605.12292)

Source paper: **STRABLE**
Origin: **HuggingFace** (`inria-soda/STRABLE-benchmark`)
Base URL: `https://huggingface.co/datasets/inria-soda/STRABLE-benchmark/tree/main/`
Task types: Mix of BC (13), MC (19), RG (76) — individual assignments pending config.json review.

| id | task_type | modality | domain | metric | n_rows | n_features | kaggle_category | status | notes |
|----|-----------|----------|--------|--------|--------|------------|-----------------|--------|-------|
| aca-federal-upper-limits-wide | RG | Tab | ? | rmse | 1,976 | 12 | N/A | pending | origin: Medicaid |
| aijob_ai-ml-ds-salaries | RG | Tab | ? | rmse | 5,672 | 8 | N/A | pending | origin: aijobs.net |
| animalandveterinary-event | BC | Tab | ? | roc_auc | 1,345 | 19 | N/A | pending | origin: fda |
| antenna-structure-registration | RG | Tab | ? | rmse | 17,633 | 28 | N/A | pending | origin: HIFLD |
| awarded-grants | RG | Tab | ? | rmse | 63,815 | 41 | N/A | pending | origin: HRSA |
| beer-ratings | RG | Tab | ? | rmse | 3,197 | 20 | N/A | pending | origin: BeerAdvocate.com |
| broadband-availability | RG | Tab | ? | rmse | 3,219 | 23 | N/A | pending | origin: Institute of Museum and Library Services |
| california-houses | RG | Tab | ? | rmse | 44,913 | 40 | N/A | pending | origin: kaggle |
| child-adult-healthcare-quality | RG | Tab | ? | rmse | 29,367 | 14 | N/A | pending | origin: Medicaid |
| china-overseas-finance-inventory | RG | Tab | ? | rmse | 550 | 50 | N/A | pending | origin: world-resource-institute |
| chocolate-bar-ratings | MC | Tab | ? | accuracy | 2,789 | 8 | N/A | pending | origin: flavorsofcacao.com |
| clear-corpus | RG | Tab | ? | rmse | 4,724 | 30 | N/A | pending | origin: commonlit.org |
| cohort-default-rate | BC | Tab | ? | roc_auc | 4,700 | 13 | N/A | pending | origin: FSA |
| college-creditcard-marketing | RG | Tab | ? | rmse | 6,283 | 12 | N/A | pending | origin: Consumer-Financial-Protection-Bureau |
| college-deposit-product-marketing | RG | Tab | ? | rmse | 1,359 | 17 | N/A | pending | origin: Consumer-Financial-Protection-Bureau |
| colleges-and-universities | RG | Tab | ? | rmse | 6,128 | 40 | N/A | pending | origin: HIFLD |
| commitments-in-trust-funds | RG | Tab | ? | rmse | 41,243 | 10 | N/A | pending | origin: worldbankfinancesone |
| community-banking_wide | RG | Tab | ? | rmse | 5,253 | 30 | N/A | pending | origin: Federal-Deposit-Insurance-Corporation |
| conflict-events_wide | RG | Tab | ? | rmse | 1,520 | 12 | N/A | pending | origin: OHCA |
| contract-awards-investment-project-financing | RG | Tab | ? | rmse | 75,000 | 20 | N/A | pending | origin: worldbankfinancesone |
| contributions-to-financial-intermediary-funds | RG | Tab | ? | rmse | 2,990 | 10 | N/A | pending | origin: worldbankfinancesone |
| corporate-procurement-contract-awards | RG | Tab | ? | rmse | 3,295 | 11 | N/A | pending | origin: worldbankfinancesone |
| cosmetic-event | MC | Tab | ? | accuracy | 75,000 | 12 | N/A | pending | origin: fda |
| covid-clinical-trials | RG | Tab | ? | rmse | 4,227 | 25 | N/A | pending | origin: ClinicalTrials.gov |
| device-classification | MC | Tab | ? | accuracy | 6,522 | 9 | N/A | pending | origin: fda |
| device-covid19serology | BC | Tab | ? | roc_auc | 13,420 | 10 | N/A | pending | origin: fda |
| device-pma | BC | Tab | ? | roc_auc | 54,636 | 19 | N/A | pending | origin: fda |
| disbursements-in-trust-funds | RG | Tab | ? | rmse | 75,000 | 10 | N/A | pending | origin: worldbankfinancesone |
| discretionary-grant | RG | Tab | ? | rmse | 13,218 | 36 | N/A | pending | origin: Institute of Museum and Library Services |
| drug-drugsfda | BC | Tab | ? | roc_auc | 27,075 | 7 | N/A | pending | origin: fda |
| drug-enforcement | MC | Tab | ? | accuracy | 4,504 | 41 | N/A | pending | origin: fda |
| drug-ndc | MC | Tab | ? | accuracy | 583 | 21 | N/A | pending | origin: fda |
| drug-shortages | MC | Tab | ? | accuracy | 1,180 | 9 | N/A | pending | origin: fda |
| electric-generating-plants | RG | Tab | ? | rmse | 8,849 | 37 | N/A | pending | origin: HIFLD |
| electric-retail-service-territories | RG | Tab | ? | rmse | 2,909 | 19 | N/A | pending | origin: HIFLD |
| external-clinician-dashboard | RG | Tab | ? | rmse | 28,399 | 20 | N/A | pending | origin: HRSA |
| financial-intermediary-funds-cash-transfers | RG | Tab | ? | rmse | 16,630 | 5 | N/A | pending | origin: worldbankfinancesone |
| financial-intermediary-funds-commitments | RG | Tab | ? | rmse | 12,875 | 6 | N/A | pending | origin: worldbankfinancesone |
| financial-intermediary-funds-funding-decisions | RG | Tab | ? | rmse | 14,750 | 9 | N/A | pending | origin: worldbankfinancesone |
| financial-management | RG | Tab | ? | rmse | 15,511 | 8 | N/A | pending | origin: Medicaid |
| financial-product-complaint | MC | Tab | ? | accuracy | 75,000 | 15 | N/A | pending | origin: Consumer-Financial-Protection-Bureau |
| first-time-nadac-rates | RG | Tab | ? | rmse | 1,130 | 9 | N/A | pending | origin: Medicaid |
| food-enforcement | MC | Tab | ? | accuracy | 7,654 | 21 | N/A | pending | origin: fda |
| food-event | MC | Tab | ? | accuracy | 23,713 | 8 | N/A | pending | origin: fda |
| food-prices_wide | RG | Tab | ? | rmse | 50,443 | 14 | N/A | pending | origin: OHCA |
| foreign-gift-and-contract | RG | Tab | ? | rmse | 75,000 | 17 | N/A | pending | origin: FSA |
| fts-funding | RG | Tab | ? | rmse | 13,153 | 34 | N/A | pending | origin: OHCA |
| fts-requirement-and-funding | RG | Tab | ? | rmse | 7,009 | 10 | N/A | pending | origin: OHCA |
| gainful-employment | RG | Tab | ? | rmse | 8,637 | 12 | N/A | pending | origin: FSA |
| global-dams-database | RG | Tab | ? | rmse | 3,978 | 56 | N/A | pending | origin: energydata.info |
| global-power-plant | RG | Tab | ? | rmse | 34,936 | 18 | N/A | pending | origin: world-resource-institute |
| grant | RG | Tab | ? | rmse | 22,424 | 9 | N/A | pending | origin: Institute of Museum and Library Services |
| health-professional-shortage-areas | RG | Tab | ? | rmse | 9,691 | 56 | N/A | pending | origin: HRSA |
| historic-perimeters-wildfires | RG | Tab | ? | rmse | 23,776 | 20 | N/A | pending | origin: HIFLD |
| historical-earthquake-locations | MC | Tab | ? | accuracy | 3,998 | 61 | N/A | pending | origin: HIFLD |
| historical-volcanic-locations | MC | Tab | ? | accuracy | 603 | 49 | N/A | pending | origin: HIFLD |
| hospitals | RG | Tab | ? | rmse | 7,883 | 31 | N/A | pending | origin: HIFLD |
| hypertension-control-wide | BC | Tab | ? | roc_auc | 1,363 | 37 | N/A | pending | origin: HRSA |
| ibrd-statement-loans-guarantees | RG | Tab | ? | rmse | 9,331 | 19 | N/A | pending | origin: worldbankfinancesone |
| ida-statement-credits-grants-guarantees | RG | Tab | Finance | RMSE | ~? | ~? | N/A | pending | World Bank IDA credits |
| ifc-advisory-services-projects | RG | Tab | ? | rmse | 1,885 | 12 | N/A | pending | origin: worldbankfinancesone |
| ifc-investment-service-projects | RG | Tab | ? | rmse | 6,328 | 18 | N/A | pending | origin: worldbankfinancesone |
| industry-payments-entity | RG | Tab | ? | rmse | 31,448 | 8 | N/A | pending | origin: energydata.info |
| industry-payments-project | RG | Tab | ? | rmse | 32,327 | 8 | N/A | pending | origin: energydata.info |
| insurance-company-complaints | RG | Tab | ? | rmse | 14,509 | 11 | N/A | pending | origin: data.ct.gov |
| journal-ranking_wide | RG | Tab | ? | rmse | 31,136 | 13 | N/A | pending | origin: SCIMAGO |
| kickstarter-projects | BC | Tab | ? | roc_auc | 75,000 | 18 | N/A | pending | origin: webrobots.io |
| lending-club-loan | BC | Tab | ? | roc_auc | 38,577 | 33 | N/A | pending | origin: lendingclub.com |
| local-government-renewable-action | RG | Tab | ? | rmse | 1,333 | 28 | N/A | pending | origin: world-resource-institute |
| local-law-enforcements | RG | Tab | ? | rmse | 16,159 | 31 | N/A | pending | origin: HIFLD |
| managed-care-enrollment | RG | Tab | ? | rmse | 2,726 | 7 | N/A | pending | origin: Medicaid |
| media-ranking_wide | RG | Tab | ? | rmse | 6,009 | 6 | N/A | pending | origin: SCIMAGO |
| medically-underserved-areas-populations | RG | Tab | ? | rmse | 3,882 | 50 | N/A | pending | origin: HRSA |
| medicines | BC | Tab | ? | roc_auc | 2,559 | 25 | N/A | pending | origin: European-Medicines-Agency |
| mercari | RG | Tab | ? | rmse | 75,000 | 7 | N/A | pending | origin: mercari.com |
| meta-critic_whisky | RG | Tab | ? | rmse | 1,812 | 7 | N/A | pending | origin: whiskyanalysis.com |
| michelin-ratings | MC | Tab | ? | accuracy | 18,839 | 17 | N/A | pending | origin: Michelin |
| miga-issued-projects | RG | Tab | ? | rmse | 619 | 13 | N/A | pending | origin: worldbankfinancesone |
| mlr-summary-reports | RG | Tab | ? | rmse | 1,665 | 13 | N/A | pending | origin: Medicaid |
| mobile-home-parks | MC | Tab | ? | accuracy | 56,094 | 25 | N/A | pending | origin: HIFLD |
| museums | RG | Tab | ? | rmse | 20,351 | 54 | N/A | pending | origin: Institute of Museum and Library Services |
| national-average-drug-acquisition-cost | RG | Tab | ? | rmse | 8,947 | 10 | N/A | pending | origin: Medicaid |
| oil-natural-gas-platform | RG | Tab | ? | rmse | 7,137 | 27 | N/A | pending | origin: HIFLD |
| orphan-designations | BC | Tab | ? | roc_auc | 3,083 | 9 | N/A | pending | origin: European-Medicines-Agency |
| osha-accidents | BC | Tab | ? | roc_auc | 4,847 | 26 | N/A | pending | origin: osha.gov |
| paediatric-investigation-plan | MC | Tab | ? | accuracy | 3,056 | 15 | N/A | pending | origin: European-Medicines-Agency |
| pol-terminal | RG | Tab | ? | rmse | 2,091 | 51 | N/A | pending | origin: HIFLD |
| power-plants | RG | Tab | ? | rmse | 23,093 | 20 | N/A | pending | origin: HIFLD |
| prepaid-financial-product | BC | Tab | ? | roc_auc | 3,336 | 12 | N/A | pending | origin: Consumer-Financial-Protection-Bureau |
| prison-boundaries | RG | Tab | ? | rmse | 3,755 | 22 | N/A | pending | origin: HIFLD |
| ramen-ratings | MC | Tab | ? | accuracy | 5,006 | 4 | N/A | pending | origin: theramenrater.com |
| rasff_window | MC | Tab | ? | accuracy | 25,702 | 13 | N/A | pending | origin: European-Commission |
| rasnf_notification_list | BC | Tab | ? | roc_auc | 42,186 | 25 | N/A | pending | origin: European-Commission |
| recipient-executed-grants-commitments-disbursements | RG | Tab | ? | rmse | 7,390 | 14 | N/A | pending | origin: worldbankfinancesone |
| schools | RG | Tab | ? | rmse | 75,000 | 26 | N/A | pending | origin: HIFLD |
| sf-building-permits | RG | Tab | ? | rmse | 75,000 | 37 | N/A | pending | origin: data.sfgov.org |
| summary-of-deposit_wide | RG | Tab | ? | rmse | 4,546 | 61 | N/A | pending | origin: Federal-Deposit-Insurance-Corporation |
| tax-incentives | RG | Tab | ? | rmse | 908 | 20 | N/A | pending | origin: fueleconomy.gov |
| terms-cc-plans | RG | Tab | ? | rmse | 9,486 | 11 | N/A | pending | origin: Consumer-Financial-Protection-Bureau |
| tobacco-problem | MC | Tab | ? | accuracy | 1,314 | 7 | N/A | pending | origin: fda |
| total-contributions-ibrd-ida-ifc | RG | Tab | ? | rmse | 7,826 | 9 | N/A | pending | origin: worldbankfinancesone |
| transmission-lines | MC | Tab | ? | accuracy | 75,000 | 10 | N/A | pending | origin: HIFLD |
| transmission-towers | RG | Tab | ? | rmse | 75,000 | 27 | N/A | pending | origin: HIFLD |
| us-school-bus-fleet | RG | Tab | ? | rmse | 17,011 | 18 | N/A | pending | origin: world-resource-institute |
| vehicles | RG | Tab | ? | rmse | 1,238 | 28 | N/A | pending | origin: fueleconomy.gov |
| wine-dataset | RG | Tab | ? | rmse | 1,290 | 16 | N/A | pending | origin: majestic.co.uk |
| workforce-demographics-wide | RG | Tab | ? | rmse | 1,588 | 19 | N/A | pending | origin: HRSA |
| yelp_business | MC | Tab | ? | accuracy | 75,000 | 13 | N/A | pending | origin: Yelp Open Dataset |

---

### C — Additional Sources (to be filled)

---

### C — Kaggle Competitions (36 fetched)

Source: **Kaggle** (competition training data — labeled)
Config: `scripts/config/kaggle_tasks.json`
Fetch: `python scripts/fetch_kaggle.py`
Output: `datasets/kaggle/`
Removed (inaccessible): `stumbleupon-evergreen-classification-challenge` (page gone), `commonlit-readability-prize` (page gone).
Note: MLE-bench competitions are listed separately in Section F.

| id | source_paper | task_type | modality | domain | metric | kaggle_category | status | notes |
|----|-------------|-----------|----------|--------|--------|-----------------|--------|-------|
| titanic | new | BC | Tab | social | accuracy | GS | fetched ✓ | Classic benchmark; survival |
| house-prices-advanced-regression-techniques | new | RG | Tab | finance | rmsle | GS | fetched ✓ | 79 features; log-normal target |
| spaceship-titanic | new | BC | Tab | social | accuracy | GS | fetched ✓ | Modern GS; mixed feature types |
| home-data-for-ml-course | AgentK | RG | Tab | finance | rmse | GS | fetched ✓ | Simplified house prices |
| bike-sharing-demand | AgentK | RG | Tab | transportation | rmsle | P | fetched ✓ | Temporal; Washington DC bikes |
| forest-cover-type-prediction | AgentK | MC | Tab | environment | accuracy | P | fetched ✓ | 7-class; 54 cartographic features |
| mercedes-benz-greener-manufacturing | AgentK | RG | Tab | transportation | r2 | F | fetched ✓ | ~380 features; anonymized |
| sberbank-russian-housing-market | AgentK | RG | Tab | finance | rmsle | F | fetched ✓ | 292 features; macroeconomic |
| restaurant-revenue-prediction | AgentK | RG | Tab | retail | rmse | F | fetched ✓ | Only ~137 train rows |
| sf-crime | AgentK | MC | Tab | social | log_loss | F | fetched ✓ | 39-class; 878k rows |
| otto-group-product-classification-challenge | AgentK | MC | Tab | retail | log_loss | F | fetched ✓ | 9-class; 93 anon. features |
| porto-seguro-safe-driver-prediction | AgentK | BC | Tab | finance | normalized_gini | F | fetched ✓ | Imbalanced ~3.6%; insurance |
| santander-customer-satisfaction | AgentK | BC | Tab | finance | roc_auc | F | fetched ✓ | ~370 anon. features; churn |
| cat-in-the-dat | AgentK | BC | Tab | mixed | roc_auc | P | fetched ✓ | Categorical encoding challenge |
| cat-in-the-dat-ii | AgentK | BC | Tab | mixed | roc_auc | P | fetched ✓ | Harder categorical encoding |
| dont-overfit-ii | AgentK | BC | Tab | mixed | roc_auc | P | fetched ✓ | 250 rows / 300 features |
| bioresponse | AgentK | BC | Tab | healthcare | log_loss | P | fetched ✓ | 1776 molecular features |
| GiveMeSomeCredit | AgentK | BC | Tab | finance | roc_auc | P | fetched ✓ | Credit default; missing values |
| predict-who-is-more-influential-in-a-social-network | AgentK | BC | Tab | social | accuracy | F | fetched ✓ | Pairwise social network |
| tmdb-box-office-prediction | AgentK | RG | Tab+Txt | retail | rmsle | P | fetched ✓ | JSON-encoded metadata cols |
| goodreads-books-reviews-290312 | AgentK | RG | Tab+Txt | retail | rmse | P | fetched ✓ | Book metadata + reviews |
| scrabble-player-rating | AgentK | RG | Tab | social | mae | P | fetched ✓ | 1.9M rows; player skill |
| store-sales-time-series-forecasting | new | RG/TS | Tab | retail | rmsle | F | fetched ✓ | Hierarchical grocery sales TS |
| ashrae-energy-prediction | new | RG/TS | Tab | energy | rmsle | F | fetched ✓ | Building energy; timestamp |
| competitive-data-science-predict-future-sales | new | RG/TS | Tab | retail | rmse | F | fetched ✓ | Multi-table; Russian sales TS |
| petfinder-adoption-prediction | new | OR/MC | Tab | social | qwk | F | fetched ✓ | Ordinal 0–4 adoption speed |
| prudential-life-insurance-assessment | new | OR/MC | Tab | finance | qwk | F | fetched ✓ | Ordinal risk 1–8; insurance |
| ieee-fraud-detection | new | BC | Tab | finance | roc_auc | F | fetched ✓ | Highly imbalanced; 590k rows |
| home-credit-default-risk | new | BC | Tab | finance | roc_auc | F | fetched ✓ | 7 relational tables; 307k rows |
| airbnb-recruiting-new-user-bookings | new | MC | Tab | retail | ndcg | F | fetched ✓ | 12-class destination prediction |
| avito-demand-prediction | new | RG | Tab+Txt | retail | rmse | F | fetched ✓ | Ad listing deal probability |
| allstate-claims-severity | new | RG | Tab | finance | mae | F | fetched ✓ | 130 anon. features; claim cost |
| bnp-paribas-cardif-claims-management | new | BC | Tab | finance | log_loss | F | fetched ✓ | Fast/slow claim; many nulls |
| microsoft-malware-prediction | new | BC | Tab | science | roc_auc | F | fetched ✓ | 8.9M rows; system telemetry |
| nyc-taxi-trip-duration | new | RG | Tab | transportation | rmsle | F | fetched ✓ | 1.4M rows; GPS + datetime |
| walmart-recruiting-store-sales-forecasting | new | RG/TS | Tab | retail | wmae | F | fetched ✓ | 45 stores; holiday weighting |

---

### F — MLE-bench Kaggle Competitions (8 fetched, from arxiv 2410.07095)

Source: **Kaggle** (competition training data — labeled)
Origin paper: **MLE-bench** (OpenAI, arXiv 2410.07095)
Config: `scripts/config/kaggle_tasks.json` (entries with `source_paper: "mlebench"`)
Fetch: `python scripts/fetch_kaggle.py`
Output: `datasets/mlebench/`
Removed: `detecting-insults-in-social-commentary` (403 Forbidden — permanently inaccessible), `h-and-m-personalized-fashion-recommendations` (35 GB images; no usable tabular target).

| id | task_type | modality | domain | metric | complexity | status | notes |
|----|-----------|----------|--------|--------|------------|--------|-------|
| new-york-city-taxi-fare-prediction | RG | Tab | transportation | rmse | low | fetched ✓ | ~55M rows; GPS + datetime |
| nomad2018-predict-transparent-conductors | RG | Tab | science | rmsle | low | fetched ✓ | ~3k rows; primary target: formation_energy_ev_natom |
| champs-scalar-coupling | RG | Tab | science | mae | medium | fetched ✓ | 4.7M rows; multi-table molecular features |
| osic-pulmonary-fibrosis-progression | RG | Tab | healthcare | mae | medium | fetched ✓ | ~1.5k rows; clinical lung capacity |
| petfinder-pawpularity-score | RG | Tab | social | rmse | medium | fetched ✓ | ~9.9k rows; image metadata (no images) |
| jigsaw-unintended-bias-in-toxicity-classification | BC | Tab+Txt | social | roc_auc | medium | fetched ✓ | 1.8M rows; float target binarized at 0.5 |
| random-acts-of-pizza | BC | Tab+Txt | social | roc_auc | low | fetched ✓ | ~5k rows; JSON format; Reddit posts |
| stanford-covid-vaccine | RG | Tab | science | mcrmse | high | fetched ✓ | ~2.4k rows; JSON format; multi-target RNA reactivity |

---

### D — OpenML Datasets (38 fetched)

Source: **OpenML** (public API, no credentials needed)
Config: `scripts/config/openml_tasks.json`
Fetch: `python scripts/fetch_openml.py`
Output: `datasets/openml/`

| id | did | task_type | modality | domain | metric | status | notes |
|----|-----|-----------|----------|--------|--------|--------|-------|
| adult | 1590 | BC | Tab | social | accuracy | fetched ✓ | Income >50K; fairness benchmark |
| bank-marketing | 1461 | BC | Tab | finance | roc_auc | fetched ✓ | Phone campaign; imbalanced |
| credit-g | 31 | BC | Tab | finance | roc_auc | fetched ✓ | German credit; cost-sensitive |
| diabetes | 37 | BC | Tab | healthcare | roc_auc | fetched ✓ | Pima Indian; classic |
| blood-transfusion-service-center | 1464 | BC | Tab | healthcare | roc_auc | fetched ✓ | 748 rows; RFMTC model |
| ilpd | 1480 | BC | Tab | healthcare | roc_auc | fetched ✓ | Liver disease; gender feature |
| spambase | 44 | BC | Tab | science | roc_auc | fetched ✓ | 57 word-frequency features |
| kc1 | 1067 | BC | Tab | science | roc_auc | fetched ✓ | Software defect; code metrics |
| qsar-biodegradation | 1494 | BC | Tab | science | roc_auc | fetched ✓ | 41 molecular descriptors |
| internet_advertisement | 40978 | BC | Tab | retail | roc_auc | fetched ✓ | 1558 sparse features |
| electricity | 151 | BC | Tab | energy | accuracy | fetched ✓ | 45k rows; concept drift |
| madelon | 1485 | BC | Tab | science | accuracy | fetched ✓ | 500 features; 5 informative |
| KDDCup09_appetency | 1111 | BC | Tab | retail | roc_auc | fetched ✓ | 50k rows; 230 features; many missing |
| phoneme | 1489 | BC | Tab | science | accuracy | fetched ✓ | 5 features; nasal vs oral |
| churn | 40701 | BC | Tab | retail | roc_auc | fetched ✓ | Telecom; imbalanced ~14% |
| climate-model-simulation-crashes | 40994 | BC | Tab | environment | roc_auc | fetched ✓ | Predict simulation failure |
| vehicle | 54 | MC | Tab | transportation | accuracy | fetched ✓ | 4 vehicle types; silhouette |
| car | 40975 | MC | Tab | transportation | accuracy | fetched ✓ | 4 acceptability classes; ordinal |
| satimage | 182 | MC | Tab | environment | accuracy | fetched ✓ | 6 land-use classes; spectral |
| segment | 40984 | MC | Tab | science | accuracy | fetched ✓ | 7 image region classes |
| har | 1478 | MC | Tab | healthcare | accuracy | fetched ✓ | 561 sensor features; 6 activities |
| steel-plates-fault | 40982 | MC | Tab | science | accuracy | fetched ✓ | 7-class fault type (target col); manufacturing |
| MiceProtein | 40966 | MC | Tab | healthcare | accuracy | fetched ✓ | 77 protein levels; many missing |
| cmc | 23 | MC | Tab | healthcare | accuracy | fetched ✓ | Contraceptive choice; ordinal |
| wall-robot-navigation | 1497 | MC | Tab | science | accuracy | fetched ✓ | 24 sensor readings; 4 directions |
| connect-4 | 40668 | MC | Tab | science | accuracy | fetched ✓ | 67k rows; 42 board features |
| eucalyptus | 188 | MC | Tab | environment | accuracy | fetched ✓ | 5 utility classes; ordinal |
| california_housing | 44977 | RG | Tab | finance | rmse | fetched ✓ | 20k rows; 8 features |
| diamonds | 42225 | RG | Tab | retail | rmse | fetched ✓ | 54k rows; ordinal quality features |
| house_sales | 42092 | RG | Tab | finance | rmse | fetched ✓ | King County; GPS + date |
| concrete-compressive-strength | 44959 | RG | Tab | science | rmse | fetched ✓ | 1030 rows; 8 ingredients |
| kin8nm | 189 | RG | Tab | science | rmse | fetched ✓ | Robot arm kinematics |
| cpu_act | 197 | RG | Tab | science | rmse | fetched ✓ | CPU activity; 21 features |
| energy-efficiency | 44960 | RG | Tab | energy | rmse | fetched ✓ | Heating load; 8 parameters |
| superconductivity | 44964 | RG | Tab | science | rmse | fetched ✓ | 81 elemental features |
| miami-housing | 43093 | RG | Tab | finance | rmse | fetched ✓ | 14k rows; GPS coordinates |
| abalone | 183 | RG | Tab | science | rmse | fetched ✓ | Ordinal-like ring count target |
| house_prices_ames | 41211 | RG | Tab | finance | rmse | fetched ✓ | 80 features; richer house prices |

---

### E — UCI Datasets (24 fetched)

Source: **UCI ML Repository** (ucimlrepo API, no credentials needed)
Config: `scripts/config/uci_tasks.json`
Fetch: `python scripts/fetch_uci.py`
Output: `datasets/uci/`

| id | uci_id | task_type | modality | domain | metric | status | notes |
|----|--------|-----------|----------|--------|--------|--------|-------|
| breast-cancer-wisconsin | 17 | BC | Tab | healthcare | roc_auc | fetched ✓ | 30 nucleus features; malignant/benign |
| magic-gamma-telescope | 159 | BC | Tab | science | roc_auc | fetched ✓ | 19k rows; particle physics |
| default-of-credit-card-clients | 350 | BC | Tab | finance | roc_auc | fetched ✓ | 30k rows; Taiwan credit default |
| heart-failure-clinical | 519 | BC | Tab | healthcare | roc_auc | fetched ✓ | 299 rows; survival; small dataset |
| occupancy-detection | 357 | BC | Tab | energy | roc_auc | fetched ✓ | TS; sensor-based; 20k rows |
| electrical-grid-stability | 471 | BC | Tab | energy | roc_auc | fetched ✓ | Simulation; 4-node star topology |
| adult | 2 | BC | Tab | social | accuracy | fetched ✓ | Use if OpenML version fails |
| estimation-of-obesity-levels | 544 | MC | Tab | healthcare | accuracy | fetched ✓ | 7 ordinal obesity classes |
| dry-bean-dataset | 602 | MC | Tab | science | accuracy | fetched ✓ | 13k rows; 16 morphological features |
| diabetes-130 | 296 | MC | Tab | healthcare | accuracy | fetched ✓ | 100k rows; readmission; many missing |
| car-evaluation | 19 | MC | Tab | transportation | accuracy | fetched ✓ | 4 ordinal acceptability classes |
| drug-consumption | 373 | MC | Tab | healthcare | accuracy | fetched ✓ | 7 ordinal usage classes; NEO-FFI |
| wine-quality | 186 | OR/RG | Tab | retail | rmse | fetched ✓ | 6497 rows; ordinal 0–10 quality |
| student-performance | 320 | OR/RG | Tab | education | rmse | fetched ✓ | Ordinal 0–20 grade; 649 rows |
| communities-and-crime | 183 | RG | Tab | social | rmse | fetched ✓ | 128 features; many missing; fairness |
| online-news-popularity | 332 | RG | Tab | social | rmse | fetched ✓ | 60 NLP features; skewed shares |
| appliances-energy-prediction | 374 | RG/TS | Tab | energy | rmse | fetched ✓ | TS; 10-min interval; smart home |
| air-quality | 360 | RG/TS | Tab | environment | rmse | fetched ✓ | Hourly air quality; -200 missing encoding |
| bike-sharing-dataset | 275 | RG/TS | Tab | transportation | rmse | fetched ✓ | Daily DC bikes; seasonal |
| beijing-pm25 | 381 | RG/TS | Tab | environment | rmse | fetched ✓ | Hourly PM2.5; wind + weather |
| metro-interstate-traffic-volume | 492 | RG/TS | Tab | transportation | rmse | fetched ✓ | Hourly traffic; holiday features |
| seoul-bike-sharing-demand | 560 | RG/TS | Tab | transportation | rmse | fetched ✓ | Hourly Korean bike rentals |
| istanbul-stock-exchange | 247 | RG/TS | Tab | finance | rmse | fetched ✓ | Daily returns; 9 markets; TS |
| superconductivity | 464 | RG | Tab | science | rmse | fetched ✓ | Separate from OpenML DID 44964 (same domain, different split) |

---

## Provenance Notes

| Label | Full Citation |
|-------|--------------|
| AgentK | Guo et al., "Kolb-Based Experiential Learning for Generalist Agents with Human-Level Kaggle Data Science Performance", arXiv 2411.03562v3 |
| STRABLE | "STRABLE: A Benchmark for Tabular Learning with Strings and Numbers", arXiv 2605.12292v1. Dataset: `inria-soda/STRABLE-benchmark` on HuggingFace. |
| mlebench | Chan et al., "MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering", arXiv 2410.07095. 75 Kaggle competitions across low/medium/high complexity splits. |
| new | Added during curation for this project |
