import numpy as np

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import SGD

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

from scikeras.wrappers import KerasClassifier


# ==========================================
# MODEL CREATOR
# ==========================================

def create_model(
        optimizer='adam',
        dropout_rate=0.5
):

    model = Sequential()

    # Layer 1
    model.add(Dense(
        128,
        activation='relu',
        input_shape=(13,)
    ))

    model.add(BatchNormalization())

    model.add(Dropout(dropout_rate))

    # Layer 2
    model.add(Dense(
        128,
        activation='relu'
    ))

    model.add(BatchNormalization())

    model.add(Dropout(dropout_rate))

    # Layer 3
    model.add(Dense(
        64,
        activation='relu'
    ))

    model.add(BatchNormalization())

    model.add(Dropout(dropout_rate))

    # Layer 4
    model.add(Dense(
        64,
        activation='relu'
    ))

    model.add(BatchNormalization())

    model.add(Dropout(dropout_rate))

    # Layer 5
    model.add(Dense(
        32,
        activation='relu'
    ))

    model.add(BatchNormalization())

    model.add(Dropout(dropout_rate))

    # Output

    model.add(Dense(
        5,
        activation='softmax'
    ))

    if optimizer == 'adam':
        opt = Adam()

    elif optimizer == 'rmsprop':
        opt = RMSprop()

    else:
        opt = SGD()

    model.compile(
        optimizer=opt,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ==========================================
# GRID SEARCH
# ==========================================

def run_grid_search(X_train, y_train):

    model = KerasClassifier(
        model=create_model,
        verbose=0
    )

    param_grid = {

        "optimizer": [
            "adam",
            "rmsprop",
            "sgd"
        ],

        "batch_size": [
            16,
            32,
            64
        ],

        "epochs": [
            50,
            100
        ],

        "dropout_rate": [
            0.3,
            0.5
        ]
    }

    grid = GridSearchCV(

        estimator=model,

        param_grid=param_grid,

        cv=3,

        scoring='accuracy',

        n_jobs=-1
    )

    grid.fit(
        X_train,
        y_train
    )

    print("\nBEST PARAMETERS")

    print(grid.best_params_)

    print("\nBEST SCORE")

    print(grid.best_score_)

    return grid.best_estimator_


# ==========================================
# CROSS VALIDATION
# ==========================================

def cross_validation(X, y):

    model = KerasClassifier(
        model=create_model,
        optimizer='adam',
        dropout_rate=0.5,
        batch_size=32,
        epochs=100,
        verbose=0
    )

    cv = StratifiedKFold(
        n_splits=10,
        shuffle=True,
        random_state=42
    )

    scores = []

    for train_idx, test_idx in cv.split(X, y):

        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        model.fit(
            X_train,
            y_train
        )

        score = model.score(
            X_test,
            y_test
        )

        scores.append(score)

    print("\n10-FOLD ACCURACY")

    print(np.mean(scores))

    return scores