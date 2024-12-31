# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference, save_component
from ml.constants import cat_features

# Add code to load in the data.

data = pd.read_csv("../data/clean_census.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

# saving the test split for future slice testing
test.to_csv("../data/clean_census_test_split.csv", index=False)

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

print("Model performance:")
print(f"precision: {precision}\nrecall: {recall}\nF1: {fbeta}")

save_component(model, "../model/model.pkl")
save_component(encoder, "../model/encoder.pkl")
save_component(lb, "../model/lb.pkl")