import logging
import asyncio
from typing import Dict, Optional

logger = logging.getLogger('camera_manager')

class CameraRegistry:
    """Manages camera instances"""
    def __init__(self):
        self._cameras: Dict[int, 'AsyncCamera'] = {}
        self._camera_id_counter = 0
        self._lock = asyncio.Lock()
        self._shutting_down = False

    async def add_camera(self, camera: 'AsyncCamera') -> int:
        """Add a camera to the registry"""
        async with self._lock:
            camera_id = self._camera_id_counter
            self._camera_id_counter += 1
            self._cameras[camera_id] = camera
            logger.info(f"Added camera {camera.name} with ID {camera_id}")
            return camera_id

    def get_camera(self, camera_id: int) -> Optional['AsyncCamera']:
        """Get a camera by ID"""
        return self._cameras.get(camera_id)

    async def remove_camera(self, camera_id: int) -> bool:
        """Remove a camera from the registry"""
        async with self._lock:
            if camera_id in self._cameras:
                camera = self._cameras[camera_id]
                try:
                    await camera.stop()
                except Exception as e:
                    logger.error(f"Error stopping camera {camera_id}: {e}")
                del self._cameras[camera_id]
                return True
            return False

    async def cleanup(self):
        """Cleanup all cameras"""
        if self._shutting_down:
            return

        self._shutting_down = True
        try:
            async with self._lock:
                for camera_id in list(self._cameras.keys()):
                    try:
                        await self.remove_camera(camera_id)
                    except Exception as e:
                        logger.error(f"Error cleaning up camera {camera_id}: {e}")
                        
                # Clear the cameras dict
                self._cameras.clear()
                
        except Exception as e:
            logger.error(f"Error during registry cleanup: {e}")
        finally:
            self._shutting_down = False 