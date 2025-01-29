import pytest
import asyncio
from app import AsyncCamera, CameraRegistry, CameraError
from config import Config, CameraConfig, RecordingConfig, ServerConfig

@pytest.fixture
def config():
    return Config(
        camera=CameraConfig(),
        recording=RecordingConfig(),
        server=ServerConfig()
    )

@pytest.fixture
def camera_registry():
    return CameraRegistry()

@pytest.mark.asyncio
async def test_camera_creation():
    camera = AsyncCamera(source=0, max_fps=20, name="test_camera")
    assert camera.source == 0
    assert camera.max_fps == 20
    assert camera.name == "test_camera"
    assert not camera.is_recording

@pytest.mark.asyncio
async def test_camera_registry_add_remove(camera_registry):
    camera = AsyncCamera(source=0, max_fps=20, name="test_camera")
    camera_id = await camera_registry.add_camera(camera)
    
    assert camera_id == 0
    assert camera_registry.get_camera(camera_id) == camera
    
    success = await camera_registry.remove_camera(camera_id)
    assert success
    assert camera_registry.get_camera(camera_id) is None

@pytest.mark.asyncio
async def test_camera_setup():
    camera = AsyncCamera(source=0, max_fps=20, name="test_camera")
    success = camera.setup(camera_id=0)
    assert success
    assert camera.camera_id == 0
    assert camera.frame_array is not None
    assert camera.status_array is not None
    await camera.stop()

@pytest.mark.asyncio
async def test_invalid_camera_source():
    with pytest.raises(ValueError):
        camera = AsyncCamera(source=-1)
        await camera.start() 