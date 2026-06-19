import matplotlib.pyplot as plt
import pickle

with open(
    "history.pkl",
    "rb"
) as f:

    history = pickle.load(f)

# Accuracy

plt.figure(figsize=(8,5))

plt.plot(
    history["accuracy"],
    label="Train"
)

plt.plot(
    history["val_accuracy"],
    label="Validation"
)

plt.title(
    "Accuracy Curve"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid()

plt.show()

# Loss

plt.figure(figsize=(8,5))

plt.plot(
    history["loss"],
    label="Train"
)

plt.plot(
    history["val_loss"],
    label="Validation"
)

plt.title(
    "Loss Curve"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid()

plt.show()