import pytest

from starter.ml.model import train_model, compute_model_metrics, inference
import numpy as np
import sklearn


MOCK_TRAIN_SIZE = 100
MOCK_FEAT_DIM = 4


@pytest.fixture
def mock_train_data():
    mock_X_train = np.random.rand(MOCK_TRAIN_SIZE, MOCK_FEAT_DIM)
    mock_y_train = np.random.randint(2, size=MOCK_TRAIN_SIZE)
    return mock_X_train, mock_y_train


@pytest.fixture
def tr_model(mock_train_data):
    mock_X_train, mock_y_train = mock_train_data
    trained_model = train_model(mock_X_train, mock_y_train)
    return trained_model


def test_compute_model_metrics():
    size = 5
    y_t = np.random.randint(2, size=size)
    y_preds_t = np.random.randint(2, size=size)
    precision, recall, fbeta = compute_model_metrics(y_t, y_preds_t)
    assert type(precision.item()) == float
    assert type(recall.item()) == float
    assert type(fbeta.item()) == float


def test_train_model(mock_train_data):
    mock_X_train, mock_y_train = mock_train_data
    trained_model = train_model(mock_X_train, mock_y_train)
    assert type(trained_model) == sklearn.linear_model._logistic.LogisticRegression


def test_inference(tr_model):
    size = 20
    mock_X = np.random.rand(size, MOCK_FEAT_DIM)
    prediction = inference(tr_model, mock_X)
    assert type(prediction) == np.ndarray
    assert prediction.dtype == np.int64
    assert prediction.size == size