import os
from observation_insight.app.main import get_app
import observation_insight.app.core.config as cfg

app = get_app()

def test_heartbeat(test_client) -> None:
    response = test_client.get('/api/health/heartbeat')
    assert response.status_code == 200
    assert response.json() == {"is_alive": True}


def test_default_route(test_client) -> None:
    response = test_client.get('/')
    expected_status_code = 404
    if cfg.MAUS_FILE_DIR is not None:
        if os.path.exists(cfg.MAUS_FILE_DIR) and os.path.exists(os.path.join(cfg.MAUS_FILE_DIR, "index.html")):
            expected_status_code = 200
    assert response.status_code == expected_status_code


def test_wrong_route(test_client) -> None:
    response = test_client.get('/api/health/xxx')
    expected_status_code = 404
    assert response.status_code == expected_status_code