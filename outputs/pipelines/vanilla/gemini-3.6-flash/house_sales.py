from scipy.optimize import minimize

OOF_matrix = np.column_stack([
    oof_lgb_raw,
    oof_lgb_log,
    oof_cb_raw,
    oof_cb_log,
    oof_xgb_raw,
    oof_xgb_log,
])
TEST_matrix = np.column_stack([
    test_lgb_raw,
    test_lgb_log,
    test_cb_raw,
    test_cb_log,
    test_xgb_raw,
    test_xgb_log,
])


def rmse_obj(weights):
  pred = OOF_matrix @ weights
  return np.sqrt(np.mean((y_train - pred) ** 2))


# Constraints: weights sum to 1, non-negative
bounds = [(0, 1) for _ in range(6)]
constraints = {"type": "eq", "fun": lambda w: 1.0 - np.sum(w)}
init_weights = np.ones(6) / 6

res = minimize(
    rmse_obj, init_weights, method="SLSQP", bounds=bounds, constraints=constraints
)
best_weights = res.x

final_test_preds = TEST_matrix @ best_weights