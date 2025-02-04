import pytest
from app import app
from camera_registry import CameraRegistry
from unittest.mock import patch, AsyncMock, MagicMock
import json
import numpy as np
import asyncio
import psutil

@pytest.fixture
async def mock_camera():
    """Create a mock camera with all required attributes and methods"""
    camera = AsyncMock()
    camera.name = "test_camera"
    camera.is_recording = False
    camera.frame_shape = (480, 640, 3)
    camera.setup.return_value = True
    camera.start.return_value = True
    camera.validate_source.return_value = True
    camera.get_camera_stats.return_value = {
        'status': 'active',
        'fps': 30.0,
        'frames_processed': 100,
        'dropped_frames': 0,
        'errors': [],
        'performance': {
            'cpu_usage': 5.0,
            'memory_usage': 50.0,
            'frame_times': 0.033
        }
    }
    camera.cap = MagicMock()
    camera.cap.isOpened.return_value = True
    camera.cap.release = MagicMock()
    yield camera

@pytest.mark.asyncio
async def test_add_camera_endpoint(test_client, mock_camera):
    """Test adding a camera"""
    with patch('app.AsyncCamera', return_value=mock_camera):
        try:
            response = await test_client.post('/add_camera', 
                json={
                    'source': 0,
                    'name': 'test_camera'
                }
            )
            
            assert response.status_code == 200
            data = await response.get_json()
            assert data['status'] == 'success'
            assert 'camera_id' in data
        except Exception as e:
            pytest.fail(f"Test failed with error: {str(e)}")

@pytest.mark.asyncio
async def test_list_cameras_empty(test_client):
    """Test listing cameras when none exist"""
    try:
        response = await test_client.get('/list_cameras')
        assert response.status_code == 200
        data = await response.get_json()
        assert data['status'] == 'success'
        assert data['cameras'] == []
    except Exception as e:
        pytest.fail(f"Test failed with error: {str(e)}")

@pytest.mark.asyncio
async def test_list_cameras_with_camera(test_client, camera_registry, mock_camera):
    """Test listing cameras with one camera"""
    try:
        async with asyncio.timeout(5.0):
            # Add mock camera to registry
            camera_registry._cameras[0] = mock_camera
            
            response = await test_client.get('/list_cameras')
            assert response.status_code == 200
            data = await response.get_json()
            
            assert data['status'] == 'success'
            assert len(data['cameras']) == 1
            camera_info = data['cameras'][0]
            assert camera_info['name'] == 'test_camera'
            assert camera_info['status'] == 'active'
            
    except asyncio.TimeoutError:
        pytest.fail("Test timed out")
    except Exception as e:
        pytest.fail(f"Test failed with error: {str(e)}")
    finally:
        camera_registry._cameras.clear()

@pytest.mark.asyncio
async def test_remove_camera(test_client, camera_registry, mock_camera):
    """Test removing a camera"""
    # Add camera first
    camera_registry._cameras[0] = mock_camera
    
    response = await test_client.post('/remove_camera/0')
    assert response.status_code == 200
    data = await response.get_json()
    assert data['status'] == 'success'
    
    # Verify camera was removed
    assert len(camera_registry._cameras) == 0

@pytest.mark.asyncio
async def test_start_recording(test_client, camera_registry, mock_camera):
    """Test starting recording"""
    # Setup mock
    mock_camera.start_recording.return_value = (True, "Recording started")
    mock_camera.current_recording = "test_recording.mp4"
    camera_registry._cameras[0] = mock_camera
    
    response = await test_client.post('/start_recording',
        json={'camera_id': 0}
    )
    
    assert response.status_code == 200
    data = await response.get_json()
    assert data['status'] == 'success'
    assert data['recording_file'] == "test_recording.mp4"

@pytest.mark.asyncio
async def test_stop_recording(test_client, camera_registry, mock_camera):
    """Test stopping recording"""
    # Setup mock
    mock_camera.is_recording = True
    mock_camera.stop_recording.return_value = (True, "Recording stopped")
    camera_registry._cameras[0] = mock_camera
    
    response = await test_client.post('/stop_recording',
        json={'camera_id': 0}
    )
    
    assert response.status_code == 200
    data = await response.get_json()
    assert data['status'] == 'success'

@pytest.mark.asyncio
async def test_get_frame(test_client, camera_registry, mock_camera):
    """Test getting a frame"""
    try:
        # Setup mock frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_camera.get_frame.return_value = frame
        
        # Add camera to registry first
        camera_id = await camera_registry.add_camera(mock_camera)
        
        response = await test_client.get(f'/frame/{camera_id}')
        assert response.status_code == 200
        assert response.headers['Content-Type'] == 'image/jpeg'
    except Exception as e:
        pytest.fail(f"Test failed with error: {str(e)}")
    finally:
        await camera_registry.remove_camera(camera_id)

@pytest.mark.asyncio
async def test_error_handling(test_client):
    """Test error handling for invalid requests"""
    try:
        # Test invalid camera ID
        response = await test_client.get('/frame/999')
        assert response.status_code == 404
        data = await response.get_json()
        assert data['status'] == 'error'
        
        # Test invalid JSON
        response = await test_client.post('/add_camera', 
            data="invalid json",
            headers={'Content-Type': 'application/json'}
        )
        assert response.status_code == 400
        data = await response.get_json()
        assert data['status'] == 'error'
        assert 'Invalid JSON data' in data['message']
        
        # Test missing required fields
        response = await test_client.post('/add_camera',
            json={},  # Missing required source field
            headers={'Content-Type': 'application/json'}
        )
        assert response.status_code == 400
        data = await response.get_json()
        assert data['status'] == 'error'
        assert 'Missing required field' in data['message']
    except Exception as e:
        pytest.fail(f"Test failed with error: {str(e)}")

@pytest.mark.asyncio
async def test_camera_reconnection(test_client, camera_registry, mock_camera):
    """Test camera reconnection after connection loss"""
    try:
        # Setup mock for connection loss and recovery
        mock_camera.is_connected = True
        mock_camera.reconnect_attempts = 0
        mock_camera.max_reconnect_attempts = 3
        
        # Add camera to registry first
        camera_id = await camera_registry.add_camera(mock_camera)
        
        # Simulate connection loss
        mock_camera.get_frame.side_effect = [
            ConnectionError("Connection lost"),  # First call fails
            None,  # Second call during reconnect
            np.zeros((480, 640, 3))  # Third call succeeds
        ]
        
        # First request - connection lost
        response = await test_client.get(f'/frame/{camera_id}')
        assert response.status_code == 503  # Service Unavailable
        
        # Wait for reconnection attempt
        await asyncio.sleep(0.1)
        
        # Reset mock for next call
        mock_camera.get_frame.side_effect = None
        mock_camera.get_frame.return_value = np.zeros((480, 640, 3))
        
        # Second request - should succeed after reconnection
        response = await test_client.get(f'/frame/{camera_id}')
        assert response.status_code == 200
        
    except Exception as e:
        pytest.fail(f"Test failed with error: {str(e)}")
    finally:
        await camera_registry.remove_camera(camera_id)

@pytest.mark.asyncio
async def test_multiple_camera_operations(test_client, camera_registry):
    """Test handling multiple cameras simultaneously"""
    # Make separate mocks:
    cameras = []
    for i in range(3):
        cam = AsyncMock()
        cam.name = f"camera_{i}"
        cam.is_recording = False
        cam.current_recording = f"recording_{i}.mp4"
        cam.start_recording.return_value = (True, f"Recording started for camera_{i}")
        cam.stop_recording.return_value = (True, f"Stopped camera_{i}")
        cam.get_camera_stats.return_value = {'status': 'active', 'fps': 30.0}
        camera_registry._cameras[i] = cam
        cameras.append(cam)

    # Now test listing, recordings, etc.
    response = await test_client.get('/list_cameras')
    data = await response.get_json()
    assert len(data['cameras']) == 3

    for i in range(3):
        response = await test_client.post('/start_recording', json={'camera_id': i})
        assert response.status_code == 200

@pytest.mark.asyncio
async def test_camera_performance_monitoring(test_client, camera_registry, mock_camera):
    """Test camera performance monitoring"""
    from unittest.mock import AsyncMock
    mock_camera.get_camera_stats = AsyncMock(side_effect=[{
            'status': 'active',
            'fps': 30.0,
            'performance': {
                'cpu_usage': 50.0,
                'memory_usage': 75.0,
                'frame_times': 0.05
            }
        }, {
            'status': 'warning',
            'fps': 15.0,
            'performance': {
                'cpu_usage': 90.0,
                'memory_usage': 95.0,
                'frame_times': 0.1
            }
        }])
    
    camera_registry._cameras[0] = mock_camera
    
    # First check - normal performance
    response = await test_client.get('/camera_status/0')
    data = await response.get_json()
    assert data['status'] == 'active'
    
    # Second check - degraded performance
    response = await test_client.get('/camera_status/0')
    data = await response.get_json()
    assert data['status'] == 'warning'

@pytest.mark.asyncio
async def test_recording_storage_management(test_client, camera_registry, mock_camera):
    """Test recording storage management"""
    import os
    from datetime import datetime, timedelta
    
    # Setup mock for recording management
    mock_camera.get_recordings = AsyncMock(return_value=[
        {
            'path': 'recordings/cam0/old_recording.mp4',
            'size': 1024 * 1024 * 100,  # 100MB
            'date': datetime.now() - timedelta(days=7)
        },
        {
            'path': 'recordings/cam0/recent_recording.mp4',
            'size': 1024 * 1024 * 50,  # 50MB
            'date': datetime.now()
        }
    ])
    
    camera_registry._cameras[0] = mock_camera
    
    # Test listing recordings
    response = await test_client.get('/list_recordings/0')
    data = await response.get_json()
    assert len(data['recordings']) == 2
    
    # Test storage cleanup
    response = await test_client.post('/cleanup_recordings/0',
        json={'max_age_days': 5}
    )
    assert response.status_code == 200
    data = await response.get_json()
    assert data['files_removed'] == 1

@pytest.mark.asyncio
async def test_concurrent_operations(test_client, camera_registry):
    """Test handling concurrent operations on the same camera"""
    # Setup 2 cameras with proper start/stop recording mock return values
    for i in range(2):
        cam = AsyncMock()
        cam.name = f"camera_{i}"
        cam.current_recording = f"recording_{i}.mp4"
        cam.start_recording.return_value = (True, f"Recording started for camera_{i}")
        cam.stop_recording.return_value = (True, f"Stopped camera_{i}")
        camera_registry._cameras[i] = cam

    async def start_recording():
        return await test_client.post('/start_recording',
            json={'camera_id': 0}
        )
    
    async def stop_recording():
        return await test_client.post('/stop_recording',
            json={'camera_id': 0}
        )
    
    # Try to start and stop recording concurrently
    responses = await asyncio.gather(
        start_recording(),
        stop_recording(),
        return_exceptions=True
    )
    
    # Verify that one operation succeeded and one failed
    assert any(r.status_code == 200 for r in responses)
    assert any(r.status_code == 409 for r in responses)

@pytest.mark.asyncio
async def test_camera_configuration(test_client, camera_registry, mock_camera):
    """Test camera configuration updates"""
    camera_registry._cameras[0] = mock_camera
    response = await test_client.get('/camera_config/0')
    assert response.status_code == 200 

@pytest.mark.asyncio
async def test_system_capacity(test_client, camera_registry):
    """Test maximum camera capacity under different resolutions"""
    try:
        from unittest.mock import AsyncMock
        import psutil
        
        # Test configurations
        configurations = [
            {
                'name': '1080p (CCTV/Dome)',
                'resolution': (1920, 1080),
                'fps': 30,
                'bitrate': '4Mbps',
                'expected_max': 8,
                'type': 'CCTV'
            },
            {
                'name': '720p (Drone)',
                'resolution': (1280, 720), 
                'fps': 60,
                'bitrate': '3Mbps',
                'expected_max': 12,
                'type': 'Aerial'
            },
            {
                'name': '360p (BWC)',
                'resolution': (640, 360),
                'fps': 25,
                'bitrate': '1Mbps',
                'expected_max': 25,
                'type': 'Body-Worn'
            },
            {
                'name': '4K (PTZ)',
                'resolution': (3840, 2160),
                'fps': 30,
                'bitrate': '8Mbps',
                'expected_max': 3,
                'type': 'Pan-Tilt-Zoom'
            }
        ]

        results = []
        
        for config in configurations:
            # Reset registry for each test
            camera_registry._cameras.clear()
            
            # Create mock cameras
            for i in range(1, 50):  # Test up to 50 instances
                cam = AsyncMock()
                cam.name = f"{config['type']}_{i}"
                cam.resolution = config['resolution']
                cam.fps = config['fps']
                cam.bitrate = config['bitrate']
                cam.get_frame.return_value = np.zeros((*config['resolution'][::-1], 3))
                
                try:
                    await camera_registry.add_camera(cam)
                except Exception as e:
                    break
                    
                # Check system resources
                cpu = psutil.cpu_percent()
                ram = psutil.virtual_memory().percent
                
                if cpu > 90 or ram > 85:
                    break

            max_supported = len(camera_registry._cameras)
            results.append({
                **config,
                'actual_max': max_supported,
                'status': 'PASS' if max_supported >= config['expected_max'] else 'FAIL'
            })

        # Print capacity table
        print("\nCamera Capacity Test Results:")
        print("+-----------------+----------------+-------+---------+-----------+-----------+--------+")
        print("| Resolution       | Camera Type    | FPS   | Bitrate | Expected  | Actual    | Status |")
        print("+-----------------+----------------+-------+---------+-----------+-----------+--------+")
        for result in results:
            print(f"| {result['name']:15} | {result['type']:14} | {result['fps']:5} | {result['bitrate']:7} | {result['expected_max']:9} | {result['actual_max']:9} | {result['status']:6} |")
        print("+-----------------+----------------+-------+---------+-----------+-----------+--------+")
        
    except Exception as e:
        pytest.fail(f"Capacity test failed: {str(e)}") 