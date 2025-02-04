import os
import sys
import pytest
import cv2
import glob
import shutil

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 

@pytest.fixture(autouse=True)
async def cleanup_cameras():
    """Cleanup any remaining camera resources after each test"""
    yield
    try:
        # Clean up test recordings directory
        test_recordings = os.path.join('recordings', 'test_camera_*')
        for path in glob.glob(test_recordings):
            try:
                shutil.rmtree(path)
            except:
                pass
                
        # Release any remaining cameras
        for i in range(10):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    await cap.release()
            except Exception:
                pass
    except Exception as e:
        print(f"Error in cleanup: {e}") 

@pytest.fixture(autouse=True)
async def cleanup_between_tests():
    """Clean up shared memory between tests"""
    from app import cleanup_shared_memory, camera_registry
    
    # Clean up before test
    for i in range(10):
        cleanup_shared_memory(i)
    
    yield
    
    # Clean up cameras properly
    cameras = list(camera_registry._cameras.values())  # Create copy of values
    for camera in cameras:
        try:
            if camera.is_recording:
                await camera.stop_recording()
            await camera.stop()
        except Exception as e:
            print(f"Error cleaning up camera: {e}")
    
    # Clear registry
    camera_registry._cameras.clear()
    
    # Clean up after test
    for i in range(10):
        cleanup_shared_memory(i)
        
    # Clean up test recordings
    if os.path.exists('recordings'):
        try:
            shutil.rmtree('recordings')
        except Exception as e:
            print(f"Error cleaning up recordings: {e}") 