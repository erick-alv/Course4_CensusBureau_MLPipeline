import argparse
from starter.ml.data import process_data
from starter.ml.model import load_component, inference, compute_model_metrics
from starter.ml.constants import cat_features
import pandas as pd


def slice_testing(model, encoder, lb, test_data):
    for cat_f in cat_features:
        print(f"======= Performing test on feature {cat_f} =======")
        for value in test_data[cat_f].unique():
            print(f"Metrics on slice '{cat_f} = {value}':")
            test_slice = test_data[test_data[cat_f] == value]
            X_test, y_test, _, _ = process_data(
                test_slice, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
            )
            preds = inference(model, X_test)
            precision, recall, fbeta = compute_model_metrics(y_test, preds)
            print(f"precision: {precision}\nrecall: {recall}\nF1: {fbeta}")


def overall_performance(model, encoder, lb, test_data):
    X_test, y_test, _, _ = process_data(
        test_data, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )
    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    print("Overall model performance:")
    print(f"precision: {precision}\nrecall: {recall}\nF1: {fbeta}")


def main(model_path, encoder_path, lb_path, test_split_path):
    model = load_component(model_path)
    encoder = load_component(encoder_path)
    lb = load_component(lb_path)
    test_data = pd.read_csv(test_split_path)
    # slice testing
    slice_testing(model, encoder, lb, test_data)
    # overall
    overall_performance(model, encoder, lb, test_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="The path to the trained model")
    parser.add_argument("--encoder_path", type=str, required=True, help="The path to the one hot encoder")
    parser.add_argument("--lb_path", type=str, required=True, help="The path to the label binarizer")
    parser.add_argument("--test_split_path", type=str, required=True, help="Path to test data for the model")
    args = parser.parse_args()
    main(args.model_path, args.encoder_path, args.lb_path, args.test_split_path)