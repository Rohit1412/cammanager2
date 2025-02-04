import pytest
from app import app, AsyncCamera, camera_registry
from config import Config
from unittest.mock import patch, Mock, AsyncMock
import numpy as np

@pytest.fixture
def test_client():
    return app.test_client()

@pytest.mark.asyncio
async def test_add_camera_endpoint():
    """Test adding a camera via API endpoint"""
    test_data = {
        'source': 0,
        'name': 'test_camera'
    }
    
    with patch('app.AsyncCamera') as mock_camera:
        # Create async mock for camera methods
        mock_instance = AsyncMock()
        mock_instance.setup.return_value = True
        mock_instance.start.return_value = True
        mock_instance.validate_source.return_value = True
        mock_camera.return_value = mock_instance
        
        response = await app.test_client().post('/add_camera', json=test_data)
        assert response.status_code == 200
        data = await response.get_json()
        assert data['status'] == 'success'
        assert 'camera_id' in data

@pytest.mark.asyncio
async def test_list_cameras_endpoint():
    """Test listing cameras via API endpoint"""
    with patch('app.camera_registry') as mock_registry:
        # Create a mock camera
        mock_camera = AsyncMock()
        mock_camera.name = "test_camera"
        mock_camera.is_recording = False
        mock_camera.frame_shape = (480, 640, 3)
        
        # Setup camera stats
        mock_camera.get_camera_stats.return_value = {
            'status': 'active',
            'fps': 30.0,
            'frames_processed': 0,
            'dropped_frames': 0,
            'errors': [],
            'performance': {
                'cpu_usage': 5.0,
                'memory_usage': 50.0,
                'frame_times': 0.033
            }
        }
        
        # Setup registry mock
        mock_registry._cameras = {0: mock_camera}
        
        # Test list endpoint
        response = await app.test_client().get('/list_cameras')
        assert response.status_code == 200
        data = await response.get_json()
        
        # Verify response structure
        assert 'cameras' in data
        assert isinstance(data['cameras'], list)
        assert len(data['cameras']) > 0
        
        # Verify camera info
        camera_info = data['cameras'][0]
        assert camera_info['name'] == "test_camera"
        assert camera_info['status'] == "active"
        assert camera_info['fps'] == 30.0
        assert 'performance' in camera_info
        assert camera_info['performance']['cpu_usage'] == 5.0

@pytest.mark.asyncio
async def test_camera_config_integration():
    """Test camera configuration integration"""
    test_config = {
        'max_fps': 30,
        'frame_width': 1280,
        'frame_height': 720
    }
    
    with patch('app.AsyncCamera') as mock_camera:
        mock_instance = AsyncMock()
        mock_instance.setup.return_value = True
        mock_instance.start.return_value = True
        mock_instance.validate_source.return_value = True
        mock_camera.return_value = mock_instance
        
        response = await app.test_client().post('/add_camera', json={
            'source': 0,
            'name': 'test_camera',
            'config': test_config
        })
        
        assert response.status_code == 200
        data = await response.get_json()
        assert data['status'] == 'success' 