import os
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from scipy.special import softmax
from scipy.optimize import minimize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# 1. Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Stars"

# Fit LabelEncoder on target values
le = LabelEncoder()
y = le.fit_transform(train_df[target_col].astype(str))


# 2. Build combined text field
def create_text_representation(df):
    brand = df["Brand"].fillna("unknown").astype(str).str.strip().str.lower()
    style = df["Style"].fillna("unknown").astype(str).str.strip().str.lower()
    country = df["Country"].fillna("unknown").astype(str).str.strip().str.lower()
    variety = df["Variety"].fillna("unknown").astype(str).str.strip().str.lower()

    return (
        "brand: "
        + brand
        + " | style: "
        + style
        + " | country: "
        + country
        + " | variety: "
        + variety
    )


train_df["text"] = create_text_representation(train_df)
test_df["text"] = create_text_representation(test_df)

# 3. 5-Fold Stratified Cross-Validation Setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
num_classes = len(le.classes_)

oof_mlp = np.zeros((len(train_df), num_classes))
oof_lr = np.zeros((len(train_df), num_classes))
oof_ridge = np.zeros((len(train_df), num_classes))
oof_cnb = np.zeros((len(train_df), num_classes))

test_mlp = np.zeros((len(test_df), num_classes))
test_lr = np.zeros((len(test_df), num_classes))
test_ridge = np.zeros((len(test_df), num_classes))
test_cnb = np.zeros((len(test_df), num_classes))

for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, y)):
    X_train_text = train_df["text"].iloc[train_idx]
    y_train = y[train_idx]
    X_val_text = train_df["text"].iloc[val_idx]
    y_val = y[val_idx]

    # Feature extraction: Word and Char N-Grams
    word_vec = TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True, min_df=2)
    char_vec = TfidfVectorizer(
        ngram_range=(2, 5), analyzer="char_wb", sublinear_tf=True, min_df=2
    )

    X_tr_word = word_vec.fit_transform(X_train_text)
    X_val_word = word_vec.transform(X_val_text)
    X_te_word = word_vec.transform(test_df["text"])

    X_tr_char = char_vec.fit_transform(X_train_text)
    X_val_char = char_vec.transform(X_val_text)
    X_te_char = char_vec.transform(test_df["text"])

    X_train = hstack([X_tr_word, X_tr_char]).tocsr()
    X_val = hstack([X_val_word, X_val_char]).tocsr()
    X_test = hstack([X_te_word, X_te_char]).tocsr()

    # Model 1: Multi-Layer Perceptron
    mlp = MLPClassifier(
        hidden_layer_sizes=(128,),
        activation="relu",
        solver="adam",
        alpha=1e-3,
        max_iter=250,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42 + fold,
    )
    mlp.fit(X_train, y_train)
    oof_mlp[val_idx] = mlp.predict_proba(X_val)
    test_mlp += mlp.predict_proba(X_test) / 5.0

    # Model 2: Logistic Regression
    lr = LogisticRegression(
        C=1.5,
        max_iter=1000,
        solver="lbfgs",
        random_state=42 + fold,
    )
    lr.fit(X_train, y_train)
    oof_lr[val_idx] = lr.predict_proba(X_val)
    test_lr += lr.predict_proba(X_test) / 5.0

    # Model 3: Ridge Classifier
    ridge = RidgeClassifier(alpha=1.0, random_state=42 + fold)
    ridge.fit(X_train, y_train)
    oof_ridge[val_idx] = softmax(ridge.decision_function(X_val), axis=1)
    test_ridge += softmax(ridge.decision_function(X_test), axis=1) / 5.0

    # Model 4: Complement Naive Bayes
    cnb = ComplementNB(alpha=0.5)
    cnb.fit(X_train, y_train)
    oof_cnb[val_idx] = cnb.predict_proba(X_val)
    test_cnb += cnb.predict_proba(X_test) / 5.0


# 4. Out-of-Fold Blending Weight Optimization
def objective(weights):
    w = np.maximum(np.array(weights), 0)
    if w.sum() > 0:
        w /= w.sum()
    blended = w[0] * oof_mlp + w[1] * oof_lr + w[2] * oof_ridge + w[3] * oof_cnb
    preds = np.argmax(blended, axis=1)
    return -accuracy_score(y, preds)


init_weights = [0.3, 0.3, 0.2, 0.2]
opt_result = minimize(objective, init_weights, method="Nelder-Mead")
opt_weights = np.maximum(opt_result.x, 0)
opt_weights /= opt_weights.sum()

oof_final_probs = (
    opt_weights[0] * oof_mlp
    + opt_weights[1] * oof_lr
    + opt_weights[2] * oof_ridge
    + opt_weights[3] * oof_cnb
)
oof_classes = np.argmax(oof_final_probs, axis=1)
acc_score = accuracy_score(y, oof_classes)
print(f"Out-of-Fold Accuracy: {acc_score:.4f}")

# 5. Save Submission File
test_final_probs = (
    opt_weights[0] * test_mlp
    + opt_weights[1] * test_lr
    + opt_weights[2] * test_ridge
    + opt_weights[3] * test_cnb
)
test_classes = np.argmax(test_final_probs, axis=1)
predicted_labels = le.inverse_transform(test_classes)

if pd.api.types.is_numeric_dtype(train_df[target_col]):
    predicted_labels = predicted_labels.astype(train_df[target_col].dtype)

submission = pd.DataFrame({"id": np.arange(len(test_df)), "Stars": predicted_labels})

os.makedirs("./working", exist_ok=True)
submission.to_csv("./working/submission.csv", index=False)
