import os
import glob
import librosa
import pandas as pd

DATASET_PATH = "data"

results = []

classes = sorted(os.listdir(DATASET_PATH))

for class_name in classes:

    durations = []

    files = glob.glob(
        os.path.join(
            DATASET_PATH,
            class_name,
            "*.wav"
        )
    )

    for file in files:

        y, sr = librosa.load(
            file,
            sr=None
        )

        duration = len(y) / sr

        durations.append(duration)

    results.append({

        "Class": class_name,

        "Min Duration":
            round(min(durations),2),

        "Max Duration":
            round(max(durations),2),

        "Mean Duration":
            round(sum(durations)/len(durations),2)
    })

df = pd.DataFrame(results)

print(df)

df.to_csv(
    "duration_report.csv",
    index=False
)