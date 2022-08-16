def test_final_logging(test_client) -> None:
    response = test_client.post(
        "/api/sentencescoring",
        json={
            "username": "Thomas Sønderborg",
            "text": "This is a test. We are working.",
            "observations": [ "obs1", "obs2", "obs3"],
            "sentences": [
                {
                    "sentence":"This is ",
                    "observations":[]
                },
                {
                    'sentence':'a ',
                    'observations':['obs1']
                },
                {
                    'sentence':'test.',
                    'observations':['obs1','obs2']
                },
                {
                    'sentence':'We are working.',
                    'observations':['obs3']
                }                
            ]
        }
    )
    assert response.status_code == 200