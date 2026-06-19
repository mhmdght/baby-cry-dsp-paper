import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.metrics import roc_curve

from sklearn.metrics import auc

from sklearn.preprocessing import label_binarize

import numpy as np


def plot_confusion_matrix(
        y_true,
        y_pred,
        classes
):

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    plt.figure(figsize=(8,6))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes
    )

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.title("Confusion Matrix")

    plt.show()


def print_report(
        y_true,
        y_pred,
        classes
):

    print(

        classification_report(
            y_true,
            y_pred,
            target_names=classes
        )
    )


def plot_roc(
        model,
        X_test,
        y_test,
        n_classes
):

    y_score = model.predict_proba(
        X_test
    )

    y_bin = label_binarize(
        y_test,
        classes=np.arange(n_classes)
    )

    plt.figure(figsize=(8,6))

    for i in range(n_classes):

        fpr, tpr, _ = roc_curve(
            y_bin[:,i],
            y_score[:,i]
        )

        roc_auc = auc(
            fpr,
            tpr
        )

        plt.plot(
            fpr,
            tpr,
            label=f'Class {i} AUC={roc_auc:.2f}'
        )

    plt.plot(
        [0,1],
        [0,1],
        '--'
    )

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title("ROC Curves")

    plt.legend()

    plt.show()