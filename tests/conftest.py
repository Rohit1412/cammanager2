import pytest
import asyncio
from app import app
from camera_registry import CameraRegistry

# Only keep the asyncio marker registration
def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers",
        "asyncio: mark test as async"
    )

@pytest.fixture(scope="function")
async def camera_registry():
    """Provide a clean CameraRegistry instance for each test"""
    registry = CameraRegistry()
    # Store old registry if exists
    old_registry = getattr(app, 'camera_registry', None)
    app.camera_registry = registry
    
    yield registry
    
    # Cleanup
    try:
        await asyncio.wait_for(registry.cleanup(), timeout=5.0)
    except asyncio.TimeoutError:
        print("Registry cleanup timed out")
    except Exception as e:
        print(f"Error cleaning up registry: {e}")
    finally:
        # Restore old registry or remove
        if old_registry:
            app.camera_registry = old_registry
        else:
            delattr(app, 'camera_registry')

@pytest.fixture
async def test_client(camera_registry):
    """Provide test client with registry"""
    async with app.test_client() as client:
        yield client

@pytest.fixture(autouse=True)
async def cleanup_between_tests():
    """Cleanup before and after each test"""
    yield
    await asyncio.sleep(0.1)  # Allow pending tasks to complete 