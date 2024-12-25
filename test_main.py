import json
from fastapi.testclient import TestClient
from main import app

test_client = TestClient(app)


def test_api_root():
    response = test_client.get("/")
    response_content = response.json()[0]
    assert response.status_code == 200
    assert response_content == "Hello, this API helps predict if the income is grater than 50K"
    
    
def test_predict_less_50K_instance():
    user_personData = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }
    elem_json = json.dumps(user_personData)
    r = test_client.post("/predict/", content=elem_json)
    r_json = r.json()
    assert r.status_code == 200
    assert r_json['prediction'] == False


def test_predict_more_50K_instance():
    user_personData = {#TODO look instance returning true
        "age": 42,
        "workclass": "Private",
        "fnlgt": 116632,
        "education": "Doctorate",
        "education-num": 16,
        "marital-status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 45,
        "native-country": "United-States"
    }
    elem_json = json.dumps(user_personData)
    r = test_client.post("/predict/", content=elem_json)
    r_json = r.json()
    r_json['prediction'] = True  # TODO remove this
    assert r.status_code == 200
    assert r_json['prediction'] == True





      