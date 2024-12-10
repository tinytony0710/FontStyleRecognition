from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score
import numpy as np
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# y_test = _y
# predictions = _y_prediction
# proba = _y_prediction_prob

def evaluate(y_test, predictions, proba):
    # Calculate performance metrics
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average=None)
    precision = precision_score(y_test, predictions, average=None)
    recall = recall_score(y_test, predictions, average=None)

    # Check if the model supports predict_proba method for AUC calculation
    if len(np.unique(y_test)) == 2:  # Binary classification
        auc = roc_auc_score(y_test, proba)
    else:  # Multiclass classification
        auc = roc_auc_score(y_test, proba, multi_class='ovo', average=None)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"AUC: {auc:.4f}")
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auc': auc
    }
