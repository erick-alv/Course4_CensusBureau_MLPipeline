# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference
import joblib

# Add code to load in the data.

data = pd.read_csv("../data/clean_census.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [ #TODO better in separate file with constants
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

# Train and save a model.
model = train_model(X_train, y_train)
preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)

print(precision, recall, fbeta)
#todo save model + lb(?) + encoder

joblib.dump(model, "../model/model.pkl")
joblib.dump(encoder, "../model/encoder.pkl")

# TODO
# Write unit tests for at least 3 functions in the model code.
# Write a function that outputs the performance of the model on slices of the data.
# Suggestion: for simplicity, the function can just output the performance on slices of just the categorical features.
# Write a model card using the provided template.
#TODO delete
if __name__ == "__main__":
    print("hello")