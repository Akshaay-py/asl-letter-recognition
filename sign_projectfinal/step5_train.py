import os
import csv
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_FOLDER, "hand_data.csv")
MODEL_FILE = os.path.join(BASE_FOLDER, "sign_model.joblib")

if not os.path.exists(DATA_FILE):
    print("ERROR: hand_data.csv not found. Run step4_collect.py first.")
else:
    features = []   # the 42 numbers for each example
    labels = []     # the correct letter for each example

    with open(DATA_FILE) as file:
        for row in csv.reader(file):
            if len(row) != 43:      # skip any broken row
                continue
            labels.append(row[0])
            features.append([float(n) for n in row[1:]])

    print("Total samples:", len(labels))
    for letter in sorted(set(labels)):
        print("  Letter", letter, ":", labels.count(letter), "samples")

    if len(set(labels)) < 2:
        print("ERROR: Need at least 2 different letters. Record more data.")
    else:
        # Keep 20% of the data hidden to test the model fairly
        x_train, x_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )

        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(x_train, y_train)

        accuracy = model.score(x_test, y_test)
        print("Accuracy on test data:", round(accuracy * 100, 1), "%")

        # Now train again using ALL the data and save the model
        model.fit(features, labels)
        joblib.dump(model, MODEL_FILE)
        print("Model saved to:", MODEL_FILE)