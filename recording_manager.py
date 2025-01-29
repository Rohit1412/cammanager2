import asyncio
from collections import deque
import logging
from datetime import datetime
import os

logger = logging.getLogger('camera_manager')

class RecordingManager:
    def __init__(self):
        self.recording_queue = {}  # camera_id -> deque of recording tasks
        self.active_recordings = {}  # camera_id -> current recording info
        self._locks = {}  # camera_id -> asyncio.Lock()

    async def get_lock(self, camera_id):
        if camera_id not in self._locks:
            self._locks[camera_id] = asyncio.Lock()
        return self._locks[camera_id]

    async def queue_recording(self, camera, duration=None, quality=None):
        """Queue a recording request"""
        camera_id = camera.camera_id
        if camera_id not in self.recording_queue:
            self.recording_queue[camera_id] = deque()

        recording_info = {
            'duration': duration,
            'quality': quality,
            'timestamp': datetime.now(),
            'status': 'queued'
        }

        self.recording_queue[camera_id].append(recording_info)
        logger.info(f"Queued recording for camera {camera_id}")
        
        # Start processing queue if not already running
        asyncio.create_task(self._process_recording_queue(camera))
        return recording_info

    async def _process_recording_queue(self, camera):
        """Process queued recordings"""
        camera_id = camera.camera_id
        lock = await self.get_lock(camera_id)
        
        if not self.recording_queue[camera_id]:
            return

        async with lock:
            while self.recording_queue[camera_id]:
                recording_info = self.recording_queue[camera_id][0]
                
                try:
                    # Start recording
                    success, message = await camera.start_recording(
                        duration=recording_info.get('duration'),
                        quality=recording_info.get('quality')
                    )
                    
                    if success:
                        recording_info['status'] = 'recording'
                        recording_info['start_time'] = datetime.now()
                        self.active_recordings[camera_id] = recording_info
                        
                        # If duration specified, schedule stop
                        if recording_info['duration']:
                            asyncio.create_task(
                                self._auto_stop_recording(
                                    camera, 
                                    recording_info['duration']
                                )
                            )
                    else:
                        recording_info['status'] = 'failed'
                        recording_info['error'] = message
                        
                except Exception as e:
                    logger.error(f"Error processing recording: {str(e)}")
                    recording_info['status'] = 'failed'
                    recording_info['error'] = str(e)
                finally:
                    self.recording_queue[camera_id].popleft()

    async def _auto_stop_recording(self, camera, duration):
        """Automatically stop recording after duration"""
        await asyncio.sleep(duration)
        await self.stop_recording(camera)

    async def stop_recording(self, camera):
        """Stop current recording"""
        camera_id = camera.camera_id
        lock = await self.get_lock(camera_id)
        
        async with lock:
            if camera_id in self.active_recordings:
                recording_info = self.active_recordings[camera_id]
                success, message = await camera.stop_recording()
                
                if success:
                    recording_info['status'] = 'completed'
                    recording_info['end_time'] = datetime.now()
                else:
                    recording_info['status'] = 'failed'
                    recording_info['error'] = message
                    
                del self.active_recordings[camera_id]
                return success, message
            
            return False, "No active recording"

    def get_recording_status(self, camera_id):
        """Get current recording status"""
        active = self.active_recordings.get(camera_id)
        queued = len(self.recording_queue.get(camera_id, [])) if camera_id in self.recording_queue else 0
        
        return {
            'active_recording': active,
            'queued_recordings': queued
        } 