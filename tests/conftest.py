import pytest
from starlette.config import environ
from starlette.testclient import TestClient

environ["API_KEY"] = "sample_api_key"

from observation_insight.app.main import get_app 

# scope="session" - only create one app for full session - this is slightly a hack but used for now to make tests run faster
@pytest.fixture(scope="session")
def test_client():
    app = get_app()
    with TestClient(app) as test_client:
        yield test_client