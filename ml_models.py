import numpy as np
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

from features import extract_features
from preprocessing import preprocess
from dataset import build_dataset

# ================================================================
# XGBOOST GPU CHECK
# ================================================================
def get_xgb_tree_method(X_tr, y_tr, scaler):
    try:
        test_model = XGBClassifier(tree_method="gpu_hist")
        test_model.fit(scaler.fit_transform(X_tr[:10]), y_tr[:10])
        return "gpu_hist"
    except:
        return "hist"


# ================================================================
# OPTUNA MODEL TRIAL
# ================================================================
def model_objective(trial, X_tr, X_te, y_tr, y_te, scaler):

    model_name = trial.suggest_categorical(
        "model",
        ["LR", "SVM", "NN", "RF", "GB", "XGB"]
    )

    # Logistic Regression
    if model_name == "LR":
        C = trial.suggest_float("lr_C", 0.01, 10.0, log=True)
        clf = LogisticRegression(C=C, max_iter=500)

    # SVM
    elif model_name == "SVM":
        C = trial.suggest_float("svm_C", 0.1, 10.0, log=True)
        gamma = trial.suggest_float("svm_gamma", 1e-4, 1e-1, log=True)
        kernel = trial.suggest_categorical("svm_kernel", ["rbf", "linear"])
        clf = SVC(C=C, gamma=gamma, kernel=kernel, probability=True)

    # Neural Network
    elif model_name == "NN":
        h1 = trial.suggest_int("nn_h1", 32, 256)
        h2 = trial.suggest_int("nn_h2", 16, 128)
        act = trial.suggest_categorical("nn_act", ["relu", "tanh"])
        alpha = trial.suggest_float("nn_alpha", 1e-5, 1e-2, log=True)
        clf = MLPClassifier(
            hidden_layer_sizes=(h1, h2),
            activation=act,
            alpha=alpha,
            max_iter=400
        )

    # Random Forest
    elif model_name == "RF":
        n = trial.suggest_int("rf_n_estimators", 100, 400)
        depth = trial.suggest_int("rf_depth", 3, 15)
        mf = trial.suggest_categorical("rf_features", ["sqrt", "log2"])
        clf = RandomForestClassifier(
            n_estimators=n,
            max_depth=depth,
            max_features=mf
        )

    # Gradient Boosting
    elif model_name == "GB":
        lr = trial.suggest_float("gb_lr", 0.01, 0.3)
        n = trial.suggest_int("gb_n", 50, 300)
        depth = trial.suggest_int("gb_depth", 2, 6)
        clf = GradientBoostingClassifier(
            learning_rate=lr,
            n_estimators=n,
            max_depth=depth
        )

    # XGBoost
    else:
        tree_method = get_xgb_tree_method(X_tr, y_tr, scaler)
        clf = XGBClassifier(
            max_depth=trial.suggest_int("xgb_max_depth", 3, 10),
            learning_rate=trial.suggest_float("xgb_lr", 0.01, 0.3),
            n_estimators=trial.suggest_int("xgb_n", 100, 400),
            subsample=trial.suggest_float("xgb_sub", 0.7, 1.0),
            colsample_bytree=trial.suggest_float("xgb_col", 0.7, 1.0),
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method=tree_method,
            random_state=42
        )

    # Train model
    clf.fit(scaler.fit_transform(X_tr), y_tr)
    preds = clf.predict_proba(scaler.transform(X_te))[:, 1]
    auc = roc_auc_score(y_te, preds)

    return auc


# ================================================================
# MAIN TRAINING FUNCTION CALLED BY GUI
# ================================================================
def run_optuna_training(state, raw_patient_eeg):

    X, y = build_dataset(state)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    scaler = StandardScaler()

    # Run Optuna
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: model_objective(trial, X_tr, X_te, y_tr, y_te, scaler),
                   n_trials=40)

    best_params = study.best_params
    best_model = best_params["model"]  # ALWAYS STRING

    # Train final model
    # ------------------------------------------------------------
    if best_model == "LR":
        clf_final = LogisticRegression(
            C=best_params["lr_C"],
            max_iter=500
        )

    elif best_model == "SVM":
        clf_final = SVC(
            C=best_params["svm_C"],
            gamma=best_params["svm_gamma"],
            kernel=best_params["svm_kernel"],
            probability=True
        )

    elif best_model == "NN":
        clf_final = MLPClassifier(
            hidden_layer_sizes=(best_params["nn_h1"], best_params["nn_h2"]),
            activation=best_params["nn_act"],
            alpha=best_params["nn_alpha"],
            max_iter=400
        )

    elif best_model == "RF":
        clf_final = RandomForestClassifier(
            n_estimators=best_params["rf_n_estimators"],
            max_depth=best_params["rf_depth"],
            max_features=best_params["rf_features"]
        )

    elif best_model == "GB":
        clf_final = GradientBoostingClassifier(
            learning_rate=best_params["gb_lr"],
            n_estimators=best_params["gb_n"],
            max_depth=best_params["gb_depth"]
        )

    else:
        tree_method = get_xgb_tree_method(X_tr, y_tr, scaler)
        clf_final = XGBClassifier(
            max_depth=best_params["xgb_max_depth"],
            learning_rate=best_params["xgb_lr"],
            n_estimators=best_params["xgb_n"],
            subsample=best_params["xgb_sub"],
            colsample_bytree=best_params["xgb_col"],
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method=tree_method,
            random_state=42
        )

    # Calibrated model
    base = clf_final.fit(scaler.fit_transform(X_tr), y_tr)
    calibrated = CalibratedClassifierCV(base, cv=5)
    calibrated.fit(scaler.transform(X_tr), y_tr)

    # Evaluate
    y_pred = calibrated.predict(scaler.transform(X_te))
    y_prob = calibrated.predict_proba(scaler.transform(X_te))[:, 1]

    acc = accuracy_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred)
    auc = roc_auc_score(y_te, y_prob)
    cm = confusion_matrix(y_te, y_pred)

    # Predict Alzheimer probability for selected patient
    fv = extract_features(preprocess(raw_patient_eeg)).reshape(1, -1)
    patient_prob = calibrated.predict_proba(scaler.transform(fv))[0, 1] * 100

    # RETURN IN CORRECT ORDER for GUI
    return (
        best_model,   # STRING (good)
        acc,
        f1,
        auc,
        cm,
        best_params,  # DICTIONARY
        patient_prob
    )
