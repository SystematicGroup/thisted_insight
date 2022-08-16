from starlette.testclient import TestClient
from observation_insight.app.main import app

def test_prediction_nopayload(test_client) -> None:
    response = test_client.post(
        "/api/to_sentences",
        json={},
        headers={"token": None 
        }
    )
    assert response.status_code == 422

def test_prediction(test_client) -> None:
    response = test_client.post(
        "/api/to_sentences",
        json={
            "username":"name",
            "text": "Jeg skriver noget meget seriøs dokumentation. Bo er faldet. Dragons are here. Det gør ondt i hans knæ.",
            "obstype_chosen": [
                "Psykosocialt (hverdagsobservation)",             
                "Kontakt til læge",                                
                "Funktionsniveau/egenomsorg (hverdagsobservation)" 
            ]
        }
    )
    assert response.status_code == 200
    assert "result" in response.json()