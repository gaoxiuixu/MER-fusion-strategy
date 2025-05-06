import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    confusion_matrix, roc_curve
)

# ============ 1. 加载数据 ============
def load_data():
    return X_trains, y_trains, X_val, y_val, X_test, y_test


def get_models_dict():
    """
    Return a dictionary of base classifiers.

    Note:
    - Each classifier here is expected to be used in combination with the DDPO oversampling method.
    - The hyperparameters should be the optimal ones obtained from prior tuning,
      where each classifier was trained on a DDPO-augmented (balanced) dataset.
    - These parameter settings aim to maximize AUC on the validation set.
    - The current instantiations use default parameters and should be replaced 
      with the tuned settings based on your specific experiment.
    """
    return {
        "LR": LogisticRegression(),          # Replace with tuned parameters
        "SVM": SVC(),                        # Replace with tuned parameters
        "RF": RandomForestClassifier(),      # Replace with tuned parameters
        "XGB": XGBClassifier(),              # Replace with tuned parameters
        "LGBM": LGBMClassifier()             # Replace with tuned parameters
    }

def calculate_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob[:, 1])
    f1 = f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    bs1 = np.mean((y_prob[y_true == 1, 1] - 1) ** 2) 
    bs0 = np.mean((y_prob[y_true == 0, 1] - 0) ** 2)

    return {
        "Accuracy": round(acc, 4),
        "AUC": round(auc, 4),
        "F1": round(f1, 4),
        "BS1": round(bs1, 4),
        "BS0": round(bs0, 4)
    }

def compute_metrics_binary(y_true, y_pred, y_score=None, note=""):
    if y_score is not None:
        metrics_dict = calculate_metrics(
            y_true,
            y_pred,
            np.column_stack([1 - y_score, y_score])
        )
    else:
        metrics_dict = calculate_metrics(
            y_true,
            y_pred,
            np.column_stack([1 - y_pred, y_pred])
        )
    metrics_dict["Method"] = note
    return metrics_dict

def er_evidence_fusion(models, X_trains, y_trains, X_val, y_val, X_test):
    model_perf = {}
    val_probs = {}
    test_probs = {}

    for idx, (name, model) in enumerate(models.items()):
        model.fit(X_trains[idx], y_trains[idx])
        val_probs[name] = model.predict_proba(X_val)
        test_probs[name] = model.predict_proba(X_test)

        y_val_pred = np.argmax(val_probs[name], axis=1)
        auc_val = roc_auc_score(y_val, val_probs[name][:, 1])
        tn, fp, fn, tp = confusion_matrix(y_val, y_val_pred).ravel()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        model_perf[name] = {
            "AUC": auc_val,
            "Recall": recall,
            "Specificity": specificity
        }

    total_auc = sum(p["AUC"] for p in model_perf.values())
    for name in model_perf:
        if total_auc > 0:
            model_perf[name]["Weight"] = model_perf[name]["AUC"] / total_auc
        else:
            model_perf[name]["Weight"] = 0.2

    def fuse_evidence(probs_dict):
        n_samples = len(next(iter(probs_dict.values())))
        fused_result = []

        for i in range(n_samples):
            names = list(models.keys())
            first = names[0]
            prob = probs_dict[first][i]
            w = model_perf[first]["Weight"]
            r1 = model_perf[first]["Recall"]
            r0 = model_perf[first]["Specificity"]

            w1 = w / (1 + w - r1) if (1 + w - r1) != 0 else w
            w0 = w / (1 + w - r0) if (1 + w - r0) != 0 else w

            s = {
                "1": w1 * prob[1],
                "0": w0 * prob[0],
                "u": max(0.0, 1 - (w1 * prob[1] + w0 * prob[0]))
            }

            for name in names[1:]:
                prob = probs_dict[name][i]
                w = model_perf[name]["Weight"]
                r1 = model_perf[name]["Recall"]
                r0 = model_perf[name]["Specificity"]

                w1 = w / (1 + w - r1) if (1 + w - r1) != 0 else w
                w0 = w / (1 + w - r0) if (1 + w - r0) != 0 else w

                s_new = {
                    "1": w1 * prob[1],
                    "0": w0 * prob[0],
                    "u": max(0.0, 1 - (w1 * prob[1] + w0 * prob[0]))
                }

                s_hat = {
                    "1": (1 - r1) * s["1"] + s["u"] * s_new["1"] + s["1"] * s_new["1"],
                    "0": (1 - r0) * s["0"] + s["u"] * s_new["0"] + s["0"] * s_new["0"],
                    "u": s["u"] * s_new["u"] + s["1"] * s_new["0"] + s["0"] * s_new["1"]
                }

                total_hat = sum(s_hat.values())
                if total_hat > 0:
                    for k in s_hat:
                        s_hat[k] /= total_hat
                s = s_hat

            p_1 = s["1"]
            p_0 = s["0"]
            total_final = p_1 + p_0
            fused_prob = p_1 / total_final if total_final > 0 else 0.5
            fused_result.append(fused_prob)

        return np.array(fused_result)

    val_er_prob = fuse_evidence(val_probs)
    test_er_prob = fuse_evidence(test_probs)

    return val_er_prob, test_er_prob, model_perf, val_probs, test_probs

def save_single_model_metrics(test_probs_dict, y_test, model_names, model_perf):
    test_metrics = []
    for name in model_names:
        prob = test_probs_dict[name][:, 1]
        pred = (prob >= 0.5).astype(int)
        metrics = calculate_metrics(y_test, pred, np.column_stack([1 - prob, prob]))
        metrics["Model"] = name
        test_metrics.append(metrics)

    test_metrics_df = pd.DataFrame(test_metrics)
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    test_metrics_path = os.path.join(desktop, "Base_Learner_Test_Metrics3.xlsx")
    test_metrics_df.to_excel(test_metrics_path, index=False)
    print(f"Single model test metrics saved to: {test_metrics_path}")

if __name__ == "__main__":
    X_trains, y_trains, X_val, y_val, X_test, y_test = load_data()
    models = get_models_dict()
    model_names = list(models.keys())

    val_er_prob, test_er_prob, model_perf, val_probs_dict, test_probs_dict = er_evidence_fusion(
        models, X_trains, y_trains, X_val, y_val, X_test
    )

    val_er_pred = (val_er_prob >= 0.5).astype(int)
    test_er_pred = (test_er_prob >= 0.5).astype(int)

    val_metrics_er = calculate_metrics(
        y_val, val_er_pred, np.column_stack([1 - val_er_prob, val_er_prob])
    )
    test_metrics_er = calculate_metrics(
        y_test, test_er_pred, np.column_stack([1 - test_er_prob, test_er_prob])
    )

    print(pd.DataFrame(model_perf).T.round(4))

    print("Validation ER metrics:")
    print(val_metrics_er)
    print("Test ER metrics:")
    print(test_metrics_er)

    save_single_model_metrics(test_probs_dict, y_test, model_names, model_perf)

    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    results_df = pd.DataFrame({
        "Dataset": ["Validation", "Test"],
        "Accuracy": [val_metrics_er["Accuracy"], test_metrics_er["Accuracy"]],
        "AUC": [val_metrics_er["AUC"], test_metrics_er["AUC"]],
        "F1": [val_metrics_er["F1"], test_metrics_er["F1"]],
        "BS1": [val_metrics_er["BS1"], test_metrics_er["BS1"]],
        "BS0": [val_metrics_er["BS0"], test_metrics_er["BS0"]]
    })
    excel_path = os.path.join(desktop, "ER_Ensemble_ResultsER3.xlsx")
    results_df.to_excel(excel_path, index=False)

    evidence_df = pd.DataFrame(model_perf).T[["Weight", "Recall", "Specificity", "AUC"]]
    evidence_path = os.path.join(desktop, "ER_Evidence_WeightsER3.xlsx")
    evidence_df.to_excel(evidence_path, index_label="Model")

   