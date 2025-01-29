import pytest
from app import app, AsyncCamera, camera_registry
from config import Config

@pytest.fixture
def test_client():
    return app.test_client()

@pytest.mark.asyncio
async def test_add_camera_endpoint(test_client):
    response = await test_client.post('/add_camera', json={
        'source': 0,
        'name': 'test_camera'
    })
    assert response.status_code == 200
    data = await response.get_json()
    assert data['status'] == 'success'
    assert 'camera_id' in data

@pytest.mark.asyncio
async def test_list_cameras_endpoint(test_client):
    response = await test_client.get('/list_cameras')
    assert response.status_code == 200
    data = await response.get_json()
    assert 'cameras' in data

@pytest.mark.asyncio
async def test_camera_config_integration():
    config = Config.load_from_file()
    camera = AsyncCamera(source=0, name="test_camera")
    assert camera.max_fps == config.camera.max_fps
    assert camera.frame_shape[0] == config.camera.frame_height
    assert camera.frame_shape[1] == config.camera.frame_width 