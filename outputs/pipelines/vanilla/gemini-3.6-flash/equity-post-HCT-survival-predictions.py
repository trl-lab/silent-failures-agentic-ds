look at the core modeling strategy:
What models and loss functions should we train?

Let's consider 4 complementary model architectures:
1. LightGBM Cox / Survival Regression model
   - Option A: LightGBM regressor predicting `efs_time` (or log(`efs_time`)) weighted or unweighted.
   - Option B: LightGBM binary classifier predicting `efs` (predicting $1 - P(efs=1)$, i.e. event-free probability).
   - Option C: LightGBM Cox model (`objective='cox'`).

2. XGBoost AFT (Accelerated Failure Time) Model
   - `objective='survival:aft'`
   - `eval_metric='aft-nloglik'`
   - Target bounds:
     `y_lower = np.maximum(efs_time, 0.001)`
     `y_upper = np.where(efs == 1, y_lower, np.inf)`
   - XGBoost AFT model directly models survival time distribution with interval censoring!

3. XGBoost Cox Model
   - `objective='survival:cox'`
   - Target $y = efs\_time \times (1 \text{ if } efs==1 \text{ else } -1)$
   - Predict log hazard score $h$, rank survival as $-h$.

4. CatBoost Cox Model
   - `loss_function='Cox'`
   - Target $y = efs