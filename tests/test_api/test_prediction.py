from fastapi import FastAPI
from starlette.testclient import TestClient

app= FastAPI()

client = TestClient(app)

@app.get("/")
async def read_main():
    return {"msg": "Hello World"}

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"msg": "Hello World"}


def test_prediction_nopayload(test_client) -> None:
    # note test_client is our app - loaded from conftest.py
    response = test_client.post(
        "/api/predict",
        json={},
        headers={"token": None 
        }
    )
    assert response.status_code == 422


def test_prediction(test_client) -> None:
    response = test_client.post(
        "/api/predict",
        json={
            "text": "test"
        }
    )
    assert response.status_code == 200
    assert "result" in response.json()