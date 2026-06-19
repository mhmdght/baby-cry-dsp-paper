import numpy as np

from sklearn.model_selection import train_test_split

from src.model_ann import run_grid_search
from src.model_ann import cross_validation

from src.evaluation import plot_confusion_matrix
from src.evaluation import print_report
from src.evaluation import plot_roc


# ---------------------------------
# LOAD FEATURES
# ---------------------------------

X = np.load("X.npy")

y = np.load("y.npy")

classes = [
    "Belly Pain",
    "Burping",
    "Discomfort",
    "Hungry",
    "Tired"
]

# ---------------------------------
# SPLIT
# ---------------------------------

X_train, X_temp, y_train, y_temp = train_test_split(

    X,
    y,

    test_size=0.25,

    stratify=y,

    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(

    X_temp,
    y_temp,

    test_size=0.60,

    stratify=y_temp,

    random_state=42
)

# ---------------------------------
# GRID SEARCH
# ---------------------------------

best_model = run_grid_search(
    X_train,
    y_train
)

# ---------------------------------
# TEST
# ---------------------------------

y_pred = best_model.predict(
    X_test
)

print_report(
    y_test,
    y_pred,
    classes
)

plot_confusion_matrix(
    y_test,
    y_pred,
    classes
)

plot_roc(
    best_model,
    X_test,
    y_test,
    len(classes)
)

# ---------------------------------
# 10-FOLD CV
# ---------------------------------

cross_validation(
    X,
    y
)