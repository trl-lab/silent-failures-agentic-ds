"""
LLM-as-judge specification: failure-mode taxonomy, structure schema, applicability gates and
the two prompts, standalone (no project imports). Two LLM calls per task, temperature 0.
Used by run_judge.py.

  1. STRUCTURE EXTRACTION (EXTRACTION_PROMPT): reads only the final code, returns 8 yes/no
     facts about which pipeline stages are present (STRUCTURE_SCHEMA).
  2. GATED JUDGING (GATED_JUDGE_PROMPT): those facts, plus 2 dataset-level facts computed from
     task.json (DATASET_LEVEL_FACT_RULES), decide which of the 15 FAILURE_MODES are applicable
     (FAILURE_MODE_PRECONDITIONS) -- a gated-out mode is never asked about. The judge assesses
     only the applicable modes, plus a free-form "other_failure_modes" list.
"""

JUDGE_MODEL = "gemini/gemini-3.1-pro-preview"   # key in config/model_registry.json

# =============================================================================================
# 1. FAILURE MODE TAXONOMY (15 modes). `gate` = the fact that must be True for the mode to be
#    applicable (None = always applicable).
# =============================================================================================

FAILURE_MODES = {
    "FM02": {
        "name": "Peeking at test data during EDA",
        "gate": None,
        "definition": (
            "The agent uses test data (features or distribution statistics) during "
            "data exploration or preprocessing design. E.g. inspecting test_df.describe(), "
            "test_df.isnull().sum(), or fitting any transformer on test data before the "
            "train/test split is established."
        ),
    },
    "FM03": {
        "name": "Not cleaning data",
        "gate": None,
        "definition": (
            "The agent makes no effort to handle missing values, duplicate rows, or obvious "
            "outliers/inconsistencies in the training data before model building. Dropping all "
            "rows with NaN counts, or relying entirely on the model to handle missing values "
            "without any explicit strategy, counts as this failure."
        ),
    },
    "FM08": {
        "name": "Scaling/normalising before splitting",
        "gate": None,
        "definition": (
            "A scaler (StandardScaler, MinMaxScaler, RobustScaler, etc.) is fit on the "
            "combined or full dataset BEFORE the train/test split — causing test statistics "
            "to leak into the training distribution. Safe pattern: fit scaler only on "
            "X_train, then transform X_test separately (or use a sklearn Pipeline)."
        ),
    },
    "FM09": {
        "name": "Feature selection before splitting",
        "gate": None,
        "definition": (
            "Feature selection (SelectKBest, VarianceThreshold, correlation-based dropping, "
            "PCA, etc.) is applied to the full dataset before splitting, leaking test "
            "information into feature choice."
        ),
    },
    "FM04": {
        "name": "Imputation before splitting",
        "gate": None,
        "definition": (
            "An imputer (SimpleImputer, KNNImputer, IterativeImputer, etc.) is fit on the "
            "full dataset before splitting, leaking test missing-value statistics into "
            "the training data."
        ),
    },
    "FM05": {
        "name": "Ignoring temporal order",
        "gate": "has_temporal",
        "definition": (
            "The dataset is a time series or has a temporal component (timestamps, "
            "date columns, sequential IDs) but the agent splits randomly rather than "
            "using a time-based split (e.g. GroupTimeSeriesSplit, or a manual cutoff "
            "date), introducing look-ahead bias."
        ),
    },
    "FM10": {
        "name": "Wrong model type for task",
        "gate": None,
        "definition": (
            "The agent uses a model that is fundamentally mismatched to the task type: "
            "e.g. a regression model for a classification task, a classifier for regression, "
            "or a single-label model for multi-label output."
        ),
    },
    "FM11": {
        "name": "No regularisation in deep learning",
        "gate": "used_deep_learning",
        "definition": (
            "The agent trains a deep learning model (PyTorch, Keras, TensorFlow) without "
            "any regularisation: no dropout layers, no weight decay / L2 penalty, no "
            "early stopping. Only applicable when DL is actually used."
        ),
    },
    "FM06": {
        "name": "No train/validation split",
        "gate": None,
        "definition": (
            "The agent trains and reports performance on the same data — no holdout set or "
            "cross-validation is used to estimate generalisation. Note: if the agent trains "
            "on the full training set and predicts on a separate provided test set WITHOUT "
            "any internal validation, this still counts if no val score is computed."
        ),
    },
    "FM13": {
        "name": "Reusing test set across model variants",
        "gate": "model_comparison",
        "definition": (
            "The agent evaluates multiple models or parameter settings on the same held-out "
            "test set and then selects the best — effectively using the test set as a "
            "validation set, causing sequential overfitting."
        ),
    },
    "FM14": {
        "name": "Single evaluation of stochastic model",
        "gate": "uses_stochastic_model",
        "definition": (
            "The agent trains and evaluates a model with a random element (random seed, "
            "random initialisation, random subsampling) exactly once and reports a single "
            "score, without averaging over multiple runs to account for variance."
        ),
    },
    "FM15": {
        "name": "Using only one evaluation metric",
        "gate": None,
        "definition": (
            "The agent reports or optimises using only one metric and does not verify that "
            "the chosen metric is appropriate for the task (e.g. accuracy on imbalanced "
            "data, RMSE when MAE would be more robust to outliers)."
        ),
    },
    "FM01": {
        "name": "Wrong task formulation",
        "gate": None,
        "definition": (
            "The agent frames the modelling problem in a way that discards signal the "
            "task actually provides. E.g. collapsing a survival-analysis / time-to-event "
            "problem into plain binary classification (ignoring the time-to-event target "
            "entirely), or treating a ranking task as independent regression per item, or "
            "ignoring a multi-target structure the task description makes explicit."
        ),
    },
    "FM07": {
        "name": "No feature engineering",
        "gate": None,
        "definition": (
            "The agent uses raw columns as-is (at most basic imputation/encoding) with no "
            "derived features, domain-informed transforms, or interaction terms -- even "
            "where the task description or column names suggest obvious derived features "
            "(ratios, date parts, text/categorical decomposition)."
        ),
    },
    "FM12": {
        "name": "Naive ensembling",
        "gate": "ensembling",
        "definition": (
            "When the agent combines multiple models, it uses an unweighted/equal-weight "
            "average or simple majority vote with no tuning of the combination itself "
            "(e.g. no stacking, no weight search, no blend-ratio optimisation), where a "
            "more sophisticated combination is straightforward given what's already computed."
        ),
    },
}

# =============================================================================================
# 2. STRUCTURE-EXTRACTION SCHEMA (stage 1: code -> 8 yes/no pipeline-stage facts). Descriptions
#    are transcribed verbatim, including the literal "DRAFT -- " prefix -- that text is sent to
#    the judge exactly as written.
# =============================================================================================

STRUCTURE_SCHEMA = [
    {"name": "data_cleaning",
     "description": "DRAFT -- does the code handle missing values, duplicates, or obvious "
                     "data-quality issues at all (any explicit strategy, not just relying on "
                     "the model)?"},
    {"name": "feature_engineering",
     "description": "DRAFT -- does the code derive any features beyond raw columns (ratios, "
                     "interactions, date parts, domain-informed transforms, encodings beyond "
                     "basic type conversion)?"},
    {"name": "validation",
     "description": "DRAFT -- does the code hold out any data (train/val split or "
                     "cross-validation) to estimate generalization before final test "
                     "evaluation?"},
    {"name": "hyperparameter_tuning",
     "description": "DRAFT -- does the code search over more than one hyperparameter setting "
                     "(grid/random search, manual sweep, Optuna, etc.)?"},
    {"name": "model_comparison",
     "description": "DRAFT -- does the code train/evaluate more than one model type or "
                     "configuration before selecting a final one?"},
    {"name": "ensembling",
     "description": "DRAFT -- does the final solution combine predictions from multiple "
                     "models?"},
    {"name": "used_deep_learning",
     "description": "DRAFT -- does the code use a deep learning framework (PyTorch, "
                     "TensorFlow, Keras)?"},
    {"name": "uses_stochastic_model",
     "description": "DRAFT -- does the final model/training process have an inherent random "
                     "element (e.g. random forest, gradient boosting with subsampling, neural "
                     "net weight init, SGD, bootstrap/bagging, k-means) -- regardless of "
                     "whether a random_state/seed is fixed for this particular run? Gates FM14 "
                     "(single evaluation of a stochastic model): a fixed-seed run can still be "
                     "the ONE evaluation that failure mode is about."},
]

# =============================================================================================
# 3. DATASET-LEVEL FACTS (from task.json, no LLM call) and PRECONDITIONS (gate -> which fact)
# =============================================================================================

DATASET_LEVEL_FACT_RULES = {
    "has_temporal": (
        "True if any column name contains 'date', 'time', 'timestamp', 'dt', 'year', 'month', "
        "or 'week' (case-insensitive substring). Gates FM05."
    ),
}

FAILURE_MODE_PRECONDITIONS = {
    "FM05": {"applicable_if": {"has_temporal": True}},
    "FM11": {"applicable_if": {"used_deep_learning": True}},
    "FM13": {"applicable_if": {"model_comparison": True}},
    "FM14": {"applicable_if": {"uses_stochastic_model": True}},
    "FM12": {"applicable_if": {"ensembling": True}},
}


def is_applicable(fid: str, facts: dict) -> bool:
    cond = FAILURE_MODE_PRECONDITIONS.get(fid, {}).get("applicable_if")
    return True if not cond else all(facts.get(k) == v for k, v in cond.items())


# =============================================================================================
# 4. THE TWO PROMPTS, VERBATIM. {placeholders} filled per task:
#    EXTRACTION_PROMPT: {code}
#    GATED_JUDGE_PROMPT: {task_id} {description} {task_type} {target_column} {metric}
#      {metric_direction} {extracted_facts} {code} {process_log} {failure_mode_list}
#      (failure_mode_list = only the modes that passed is_applicable() for this task)
# =============================================================================================

EXTRACTION_PROMPT = """\
You are analyzing a data science solution's FINAL delivered code to extract structural \
facts about what the pipeline actually does -- not whether it's correct, just what \
steps/techniques are present IN THIS CODE. This will be used to determine which quality \
checks even apply to this solution.

Judge only what this code does. Do not infer a fact from general knowledge of how such \
solutions are typically produced (e.g. an agent's search/reasoning process) -- if a step \
isn't reflected in the code below, the fact is false, even if you'd expect that kind of \
step to have happened somewhere along the way.

## Final code
```python
{code}
```

## Facts to extract
{field_list}

## Output format
Return ONLY a valid JSON object, one boolean per field:
{{
{field_json_template}
}}
"""

GATED_JUDGE_PROMPT = """\
You are an expert ML engineer reviewing an agent's data science solution for a predictive \
modelling task. Your job is to identify concrete failure modes -- bad data science \
practices that could harm model quality.

## Task context
- Task ID: {task_id}
- Description: {description}
- Task type: {task_type}
- Target column: {target_column}
- Metric: {metric} ({metric_direction} is better)

## Extracted structural facts about this solution
{extracted_facts}

## Final code solution
```python
{code}
```

## Process log (reasoning / search trace -- supporting context only, may include discarded attempts)
```
{process_log}
```

## Failure modes to evaluate
Only the modes below are applicable to this solution given the structural facts above --
modes gated out by a missing precondition are excluded, don't guess about them.
{failure_mode_list}

## Output format
Return ONLY a valid JSON object:

{{
  "task": "{task_id}",
  "failure_modes": {{
    "<ID>": {{
      "present": true | false | "cannot_determine",
      "severity": 1 | 2 | 3,
      "confidence": "high" | "medium" | "low",
      "evidence": "<direct quote from the code, or null>",
      "reasoning": "<one sentence>"
    }}
  }},
  "other_failure_modes": [
    {{
      "name": "<short name for a failure mode NOT in the list above>",
      "description": "<1-2 sentences, written generally so it could apply to other tasks too>",
      "severity": 1 | 2 | 3,
      "confidence": "high" | "medium" | "low",
      "evidence": "<direct quote>"
    }}
  ],
  "summary": "<2-3 sentences>"
}}

Rules:
- Only include keys in "failure_modes" for the IDs listed above as applicable
- "other_failure_modes": list anything else you notice that doesn't fit the fixed list above, including execution failures -- empty list if nothing else found
- Severity: 1 = minor, 2 = moderate, 3 = critical (invalidates results or causes data leakage)
- Do not judge a failure mode as present solely because it appeared in a discarded earlier attempt in the process log
"""
