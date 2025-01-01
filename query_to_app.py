import json
import requests
import os


def main():
    user_personData = {
        "age": 33,
        "workclass": "Self-emp-not-inc",
        "fnlgt": 272359,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Married-civ-spouse",
        "occupation": "Sales",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 7298,
        "capital-loss": 0,
        "hours-per-week": 80,
        "native-country": "United-States"
    }

    print(f"Sending request to {os.environ['APP_URL']}")
    r = requests.post(f"{os.environ['APP_URL']}/predict/", data=json.dumps(user_personData))

    print(r.status_code)
    print(r.json())

if __name__ == "__main__":
    main()