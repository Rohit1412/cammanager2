import os
import cv2
from quart import Quart, websocket, render_template, jsonify, request
import asyncio
from multiprocessing import Process, shared_memory, freeze_support, Value, Manager
import numpy as np
import time
from datetime import datetime
import logging
from vidgear.gears import CamGear
import socket
from contextlib import closing
import re
from logging_config import setup_logging
from error_handlers import handle_camera_errors, CameraError, CameraInitError, StreamError, RecordingError
from recording_manager import RecordingManager
import signal
import sys
import atexit
from config import Config
import shutil
from pathlib import Path
from utils import GPU_AVAILABLE

# Move app creation to top, after imports
app = Quart(__name__)

logger = setup_logging()

# Replace hardcoded values with config
config = Config.load_from_file()

# Protocol patterns
PROTOCOL_PATTERNS = {
    'rtsp': r'^rtsp://',
    'rtmp': r'^rtmp://',
    'http': r'^http://',
    'https': r'^https://',
    'udp': r'^udp://',
    'tcp': r'^tcp://',
    'ip': r'^(?:\d{1,3}\.){3}\d{1,3}'
}

def determine_camera_type(source):
    """Determine the type of camera/stream from the source."""
    logging.info(f"Attempting to determine camera type for source: {source}")
    
    # Handle integer or string number for USB cameras
    if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
        try:
            source_int = int(source)
            logging.info(f"Testing USB camera at index {source_int}")
            
            # Test if camera is accessible
            cap = cv2.VideoCapture(source_int)
            if cap.isOpened():
                # Get camera properties
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = cap.get(cv2.CAP_PROP_FPS)
                logging.info(f"Successfully opened camera. Properties: width={width}, height={height}, fps={fps}")
                
                cap.release()
                return 'usb', source_int  # Return both type and source
            else:
                available_cameras = []
                # Try to find available cameras
                for i in range(10):
                    temp_cap = cv2.VideoCapture(i)
                    if temp_cap.isOpened():
                        available_cameras.append(i)
                        temp_cap.release()
                
                if available_cameras:
                    logging.error(f"Camera index {source} not available. Available cameras: {available_cameras}")
                else:
                    logging.error("No cameras found on system")
                raise ValueError(f"Unable to open USB camera at index {source}")
        except Exception as e:
            logging.error(f"Error accessing USB camera: {str(e)}")
            raise ValueError(f"Error accessing USB camera: {str(e)}")
    
    # Handle URL-based sources
    source_str = str(source).lower()
    for protocol, pattern in PROTOCOL_PATTERNS.items():
        if re.match(pattern, source_str):
            return protocol, source  # Return both type and source
    
    raise ValueError(f"Unsupported camera source: {source}")

class CameraCapture:
    """Factory class to create appropriate camera capture instance"""
    @staticmethod
    def create_capture(source):
        camera_type, source = determine_camera_type(source)  # Unpack both values
        
        if camera_type == 'usb':
            # Convert to int for USB cameras
            source = int(source) if isinstance(source, str) else source
            capture = cv2.VideoCapture(source)
            if not capture.isOpened():
                raise ValueError(f"Failed to open USB camera at index {source}")
            return capture, camera_type  # Return both capture and type
            
        elif camera_type in ['rtmp', 'http', 'https', 'ip']:
            return CamGear(source=source).start(), camera_type
            
        elif camera_type in ['rtsp', 'udp', 'tcp']:
            return cv2.VideoCapture(source), camera_type
            
        else:
            raise ValueError(f"Unsupported camera type: {camera_type}")

    @staticmethod
    def read_frame(capture, camera_type):
        """Read a frame from the capture based on camera type"""
        if camera_type in ['rtmp', 'http', 'https', 'ip']:
            return capture.read()
        else:  # OpenCV capture
            ret, frame = capture.read()
            return frame if ret else None

class CameraManager:
    def __init__(self):
        self.camera_id_counter = 0
        self.lock = asyncio.Lock()
    
    async def generate_camera_id(self):
        async with self.lock:
            camera_id = self.camera_id_counter
            self.camera_id_counter += 1
            return camera_id

# Global constants
RECORDING_DIR = 'recordings'

def cleanup_shared_memory(camera_id):
    """Clean up any existing shared memory for a camera ID"""
    shm_names = [
        f"camera_frame_{camera_id}",
        f"camera_status_{camera_id}",
        f"recording_status_{camera_id}"
    ]
    
    for shm_name in shm_names:
        try:
            # First try to attach to existing shared memory
            shm = shared_memory.SharedMemory(name=shm_name)
            try:
                shm.unlink()
            except Exception as e:
                logger.warning(f"Error unlinking shared memory {shm_name}: {e}")
            finally:
                try:
                    shm.close()
                except Exception as e:
                    logger.warning(f"Error closing shared memory {shm_name}: {e}")
        except FileNotFoundError:
            # Shared memory doesn't exist, which is fine
            continue
        except Exception as e:
            logger.warning(f"Error accessing shared memory {shm_name}: {e}")

class AsyncCamera:
    def __init__(self, source, max_fps=None, name="default"):
        self.source = source
        self.max_fps = max_fps or config.camera.max_fps
        self.name = name
        self.camera_id = None
        
        # Use config values for frame dimensions
        self.frame_shape = (
            config.camera.frame_height,
            config.camera.frame_width,
            3
        )
        self.frame_size = np.prod(self.frame_shape) * np.dtype(np.uint8).itemsize
        
        # Don't create shared memory here - move to setup method
        self.frame_shm = None
        self.status_shm = None
        self.frame_array = None
        self.status_array = None
        self.recording_dir = None
        
        # Add these attributes
        self.recording_process = None
        self.is_recording = False
        self.current_recording = None
        self.recording_quality = None
        self._is_stopping = False
        self._processes = []  # Keep track of all child processes
        self._cleanup_lock = asyncio.Lock()
        self._shared_memory = []  # Track shared memory objects
        
    def _track_shared_memory(self, shm):
        """Track shared memory for cleanup"""
        self._shared_memory.append(shm)

    def setup(self, camera_id):
        """Initialize shared memory after camera_id is assigned"""
        self.camera_id = camera_id
        
        # Clean up any existing shared memory first
        cleanup_shared_memory(camera_id)
        
        try:
            # Initialize frame shape and shared memory using config values
            self.frame_size = np.prod(self.frame_shape) * np.dtype(np.uint8).itemsize
            self.frame_shm = shared_memory.SharedMemory(
                name=f"camera_frame_{camera_id}", 
                create=True, 
                size=self.frame_size
            )
            self._track_shared_memory(self.frame_shm)
            
            # Status array
            self.status_shm = shared_memory.SharedMemory(
                name=f"camera_status_{camera_id}",
                create=True,
                size=5 * np.dtype(np.float32).itemsize
            )
            self._track_shared_memory(self.status_shm)

            # Initialize arrays
            self.frame_array = np.ndarray(
                self.frame_shape,
                dtype=np.uint8,
                buffer=self.frame_shm.buf
            )
            self.status_array = np.ndarray(
                (5,),
                dtype=np.float32,
                buffer=self.status_shm.buf
            )
            self.status_array.fill(0)
            
            self.recording_dir = os.path.join(RECORDING_DIR, f"{self.name}_{camera_id}")
            os.makedirs(self.recording_dir, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Error in setup: {str(e)}")
            self.cleanup()
            raise e

    def _add_process(self, process):
        """Add process to tracking list"""
        self._processes.append(process)

    def _remove_process(self, process):
        """Remove process from tracking list"""
        if process in self._processes:
            self._processes.remove(process)

    async def _terminate_process(self, process, timeout=2.0):
        """Safely terminate a process"""
        try:
            if process and process.is_alive():
                process.terminate()
                process.join(timeout=timeout)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=1.0)
        except Exception as e:
            logger.error(f"Error terminating process: {str(e)}")
        finally:
            self._remove_process(process)

    async def start(self):
        self.status_array[2] = 1.0  # Set running flag
        self.capture_process = Process(
            target=self._capture_loop,
            args=(self.source, self.max_fps)
        )
        self._add_process(self.capture_process)
        self.capture_process.start()

    def _capture_loop(self, source, max_fps):
        capture = None
        consecutive_failures = 0
        max_failures = 10
        
        try:
            capture, camera_type = CameraCapture.create_capture(source)  # Get both values
            logging.info(f"Camera initialized: type={camera_type}, source={source}")
            
            last_frame_time = time.time()
            frame_interval = 1.0 / max_fps

            while self.status_array[2] > 0:
                current_time = time.time()
                if current_time - last_frame_time >= frame_interval:
                    frame = CameraCapture.read_frame(capture, camera_type)
                    
                    if frame is not None:
                        consecutive_failures = 0
                        if frame.shape != self.frame_shape:
                            frame = cv2.resize(frame, 
                                            (self.frame_shape[1], self.frame_shape[0]))
                        self.frame_array[:] = frame
                        self.status_array[1] = 1.0 / (current_time - last_frame_time)
                        last_frame_time = current_time
                    else:
                        consecutive_failures += 1
                        logging.warning(f"Failed to read frame. Attempt {consecutive_failures}/{max_failures}")
                        if consecutive_failures >= max_failures:
                            raise ValueError("Too many consecutive failures reading frames")
                        time.sleep(0.1)

        except Exception as e:
            logging.error(f"Error in capture loop: {str(e)}")
            self.status_array[2] = 0.0  # Signal process to stop
        finally:
            if capture is not None:
                if camera_type in ['rtmp', 'http', 'https', 'ip']:
                    capture.stop()
                else:
                    capture.release()

    async def start_recording(self, duration=None, quality=None):
        """Start recording with optional duration and quality settings"""
        if self.is_recording:
            return False, "Already recording"
            
        try:
            # Create date-based subdirectory
            date_dir = datetime.now().strftime('%Y-%m-%d')
            timestamp = datetime.now().strftime('cam_%H-%M-%S.mp4')
            
            # Create full directory path
            recording_subdir = os.path.join(self.recording_dir, date_dir)
            os.makedirs(recording_subdir, exist_ok=True)
            
            # Full path for the output file
            output_path = os.path.join(recording_subdir, timestamp)
            
            # Set recording quality
            if quality:
                self.recording_quality = quality
            
            # Clean up any existing shared memory
            await self._cleanup_recording_resources()
            
            # Initialize new recording
            await self._initialize_recording(output_path)
            
            logger.info(f"Recording started successfully: {output_path}")
            return True, f"Recording started: {output_path}"
        except Exception as e:
            logger.error(f"Failed to start recording: {str(e)}", exc_info=True)
            await self._cleanup_recording_resources()
            return False, str(e)

    async def _cleanup_recording_resources(self):
        """Clean up recording-related resources"""
        if hasattr(self, 'recording_status_shm'):
            try:
                self.recording_status_shm.close()
                self.recording_status_shm.unlink()
            except Exception as e:
                logger.warning(f"Error cleaning up recording resources: {str(e)}")

    async def _initialize_recording(self, output_path):
        """Initialize recording resources"""
        # Create shared memory for recording status
        self.recording_status_shm = shared_memory.SharedMemory(
            create=True,
            size=np.dtype(np.bool_).itemsize,
            name=f"recording_status_{self.camera_id}"
        )
        self.recording_status = np.ndarray(
            (1,), dtype=np.bool_, buffer=self.recording_status_shm.buf
        )
        self.recording_status[0] = True
        
        # Start recording process
        self.recording_process = Process(
            target=self._record_video,
            args=(
                output_path,
                f"camera_frame_{self.camera_id}",
                self.frame_shape,
                f"recording_status_{self.camera_id}",
                self.recording_quality
            )
        )
        self._add_process(self.recording_process)
        self.recording_process.start()
        
        # Update status
        self.is_recording = True
        self.status_array[0] = 1.0
        self.current_recording = output_path

    def _record_video(self, output_path, frame_shm_name, frame_shape, recording_status_name, quality):
        """Recording process function"""
        frame_shm = None
        recording_status_shm = None
        out = None
        
        try:
            # Reconnect to shared memory in the new process
            frame_shm = shared_memory.SharedMemory(name=frame_shm_name)
            recording_status_shm = shared_memory.SharedMemory(name=recording_status_name)
            
            frame_array = np.ndarray(frame_shape, dtype=np.uint8, buffer=frame_shm.buf)
            recording_status = np.ndarray((1,), dtype=np.bool_, buffer=recording_status_shm.buf)
            
            # Initialize video writer
            out = self.create_video_writer(output_path, 'mp4v', self.max_fps, (frame_shape[1], frame_shape[0]))

            while recording_status[0]:  # Check recording status
                try:
                    # Make a copy of the frame to avoid any race conditions
                    frame = np.copy(frame_array)
                    if frame is not None and frame.size > 0:
                        out.write(frame)
                    time.sleep(1.0 / self.max_fps)
                except Exception as e:
                    logging.error(f"Error writing frame: {str(e)}")
                    break
                    
        except Exception as e:
            logging.error(f"Recording process error: {str(e)}")
        finally:
            # Clean up resources
            if out is not None:
                out.release()
            if frame_shm is not None:
                frame_shm.close()
            if recording_status_shm is not None:
                recording_status_shm.close()

    def create_video_writer(self, output_path, codec, fps, resolution):
        """Create video writer with GPU support if available"""
        fourcc = cv2.VideoWriter_fourcc(*codec)
        
        if GPU_AVAILABLE and config.recording.hardware_acceleration:
            # Initialize GPU-accelerated writer
            writer = cv2.cudacodec.createVideoWriter(
                output_path,
                cv2.VideoWriter.fourcc(*codec),
                fps,
                resolution,
                True
            )
            return writer
        else:
            # Fallback to CPU writer
            return cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                resolution
            )

    async def stop_recording(self):
        """Stop the current recording"""
        if not self.is_recording:
            return False, "Not recording"
            
        try:
            # Signal recording process to stop
            if hasattr(self, 'recording_status'):
                self.recording_status[0] = False
            
            if self.recording_process:
                # Give the process a moment to finish cleanly
                await asyncio.sleep(0.5)
                
                # Check if process exists and is alive before trying to terminate
                if hasattr(self, 'recording_process') and self.recording_process and self.recording_process.is_alive():
                    await self._terminate_process(self.recording_process)
                
                self.recording_process = None
            
            # Clean up recording status shared memory
            if hasattr(self, 'recording_status_shm'):
                try:
                    self.recording_status_shm.close()
                    self.recording_status_shm.unlink()
                except Exception as e:
                    logger.warning(f"Error cleaning up recording status shared memory: {str(e)}")
                finally:
                    delattr(self, 'recording_status_shm')
                    if hasattr(self, 'recording_status'):
                        delattr(self, 'recording_status')
            
            self.is_recording = False
            recorded_file = self.current_recording
            self.current_recording = None
            self.status_array[0] = 0.0
            
            logger.info(f"Recording stopped successfully: {recorded_file}")
            return True, f"Recording stopped: {recorded_file}"
        except Exception as e:
            logger.error(f"Failed to stop recording: {str(e)}", exc_info=True)
            return False, str(e)

    async def stop(self):
        """Stop the camera and clean up resources"""
        if self._is_stopping:
            return True

        async with self._cleanup_lock:
            self._is_stopping = True
            try:
                # First stop recording if active
                if self.is_recording:
                    try:
                        await self.stop_recording()
                    except Exception as e:
                        logger.error(f"Error stopping recording during camera shutdown: {e}")

                # Signal processes to stop
                if hasattr(self, 'status_array') and self.status_array is not None:
                    try:
                        self.status_array[2] = 0.0
                    except Exception as e:
                        logger.error(f"Error updating status array during shutdown: {e}")

                # Wait briefly for processes to notice stop signal
                await asyncio.sleep(0.5)

                # Terminate processes
                processes = self._processes.copy()
                for process in processes:
                    try:
                        await self._terminate_process(process)
                    except Exception as e:
                        logger.error(f"Error terminating process during shutdown: {e}")

                # Clean up shared memory
                if hasattr(self, '_shared_memory'):
                    for shm in self._shared_memory.copy():
                        try:
                            shm.close()
                            shm.unlink()
                        except Exception as e:
                            logger.error(f"Error cleaning shared memory during shutdown: {e}")

                self._shared_memory.clear()
                return True
            except Exception as e:
                logger.error(f"Error during camera shutdown: {e}")
                return False
            finally:
                self._is_stopping = False

    async def _cleanup_shared_memory(self):
        """Clean up shared memory resources"""
        try:
            # Clean up frame shared memory
            if hasattr(self, 'frame_shm') and self.frame_shm:
                try:
                    self.frame_shm.close()
                    self.frame_shm.unlink()
                except Exception as e:
                    logger.warning(f"Error cleaning up frame shared memory: {str(e)}")

            # Clean up status shared memory
            if hasattr(self, 'status_shm') and self.status_shm:
                try:
                    self.status_shm.close()
                    self.status_shm.unlink()
                except Exception as e:
                    logger.warning(f"Error cleaning up status shared memory: {str(e)}")

            # Clean up recording status shared memory
            if hasattr(self, 'recording_status_shm') and self.recording_status_shm:
                try:
                    self.recording_status_shm.close()
                    self.recording_status_shm.unlink()
                except Exception as e:
                    logger.warning(f"Error cleaning up recording status shared memory: {str(e)}")
        except Exception as e:
            logger.error(f"Error in cleanup_shared_memory: {str(e)}")

    async def _safe_shared_memory_access(self, operation, max_retries=3):
        """Safely perform shared memory operations with retries"""
        retries = 0
        while retries < max_retries:
            try:
                return operation()
            except FileNotFoundError:
                logger.warning(f"Shared memory not found, retrying ({retries+1}/{max_retries})")
                await asyncio.sleep(0.1 * (retries + 1))
                retries += 1
            except Exception as e:
                logger.error(f"Shared memory error: {str(e)}")
                raise
        raise RuntimeError("Max retries exceeded for shared memory operation")

    async def get_frame(self):
        """Safely get current frame with error handling"""
        try:
            return await self._safe_shared_memory_access(
                lambda: np.copy(self.frame_array)
            )
        except Exception as e:
            logger.error(f"Frame access error: {str(e)}")
            raise StreamError("Failed to retrieve frame")

    async def get_stream_metadata(self):
        """Get metadata about the current stream"""
        return {
            'resolution': f"{self.frame_shape[1]}x{self.frame_shape[0]}",
            'fps': self.status_array[1],
            'codec': 'H264',
            'bitrate': self._calculate_bitrate(),
            'timestamp': datetime.now().isoformat()
        }

    def _calculate_bitrate(self):
        """Estimate current bitrate"""
        if self.frame_array is None:
            return 0
        frame_size = self.frame_array.nbytes
        return int(frame_size * self.status_array[1] * 8 / 1e6)  # Mbps

    async def process_frames(self):
        """Process frames with GPU acceleration if available"""
        try:
            if GPU_AVAILABLE:
                # Upload frame to GPU
                gpu_frame = cv2.cuda_GpuMat()
                while self.status_array[2] > 0:
                    frame = await self.get_frame()
                    gpu_frame.upload(frame)
                    # Process on GPU
                    processed = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2RGB)
                    # Download from GPU
                    frame = processed.download()
                    # ... rest of processing ...
            else:
                # CPU processing
                while self.status_array[2] > 0:
                    frame = await self.get_frame()
                    # ... rest of processing ...
        except Exception as e:
            logger.error(f"Frame processing error: {str(e)}")

@app.websocket('/stream/<int:camera_id>')
@handle_camera_errors
async def stream(camera_id):
    camera = camera_registry.get_camera(camera_id)
    if not camera:
        logger.warning(f"Attempt to stream non-existent camera ID: {camera_id}")
        raise CameraError(f'Camera {camera_id} not found', 404)
    
    try:
        logger.info(f"Starting stream for camera {camera_id}")
        while camera.status_array[2] > 0:
            try:
                frame = await camera.get_frame()
                _, buffer = cv2.imencode('.jpg', frame)
                await websocket.send(buffer.tobytes())
                await asyncio.sleep(1/30)
            except Exception as e:
                logger.error(f"Frame processing error for camera {camera_id}: {str(e)}")
                raise StreamError(f"Stream processing failed: {str(e)}")
    except Exception as e:
        logger.error(f"Streaming error for camera {camera_id}: {str(e)}", exc_info=True)
        raise StreamError(f"Stream failed: {str(e)}")

class CameraRegistry:
    def __init__(self):
        self._counter = Value('i', 0)
        self._cameras = {}
        self._lock = asyncio.Lock()
        self._cleanup_lock = asyncio.Lock()

    async def add_camera(self, camera):
        async with self._lock:
            camera_id = self._counter.value
            self._counter.value += 1
            self._cameras[camera_id] = camera
            return camera_id

    async def remove_camera(self, camera_id):
        """Remove a camera from the registry with proper cleanup"""
        async with self._cleanup_lock:
            async with self._lock:
                if camera_id in self._cameras:
                    camera = self._cameras[camera_id]
                    # Remove from registry first to prevent new operations
                    del self._cameras[camera_id]
                    
                    try:
                        # Stop the camera and clean up its resources
                        await camera.stop()
                        return True
                    except Exception as e:
                        logger.error(f"Error during camera cleanup: {str(e)}")
                        # Don't add the camera back - it's in an unknown state
                        return False
                return False

    def get_camera(self, camera_id):
        return self._cameras.get(camera_id)

    def list_cameras(self):
        return list(self._cameras.items())

# Replace global variables with registry
camera_registry = CameraRegistry()

# Add after camera_registry initialization
recording_manager = RecordingManager()

# Add this before the main() function
class ProcessManager:
    def __init__(self):
        self.manager = Manager()
        self.active_processes = self.manager.list()

    def register_process(self, process):
        self.active_processes.append(process)

    def unregister_process(self, process):
        if process in self.active_processes:
            self.active_processes.remove(process)

    async def cleanup_all(self):
        """Clean up all registered processes"""
        processes = list(self.active_processes)  # Create copy
        for process in processes:
            try:
                if process and process.is_alive():
                    process.terminate()
                    process.join(timeout=2.0)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=1.0)
            except Exception as e:
                logger.error(f"Error cleaning up process: {e}")
            finally:
                self.unregister_process(process)

# Create global process manager
process_manager = ProcessManager()

@app.route('/add_camera', methods=['POST'])
@handle_camera_errors
async def add_camera():
    try:
        if not request.is_json:
            raise CameraError('Request must be JSON', 400)

        data = await request.get_json()
        
        if not data:
            raise CameraError('Empty request body', 400)

        source = data.get('source')
        if source is None:
            raise CameraError('Camera source is required', 400)

        name = data.get('name', 'default')
        max_fps = data.get('max_fps', 20)
        
        if isinstance(source, str) and source.isdigit():
            source = int(source)
            
        # Create camera without ID first
        try:
            camera = AsyncCamera(source, max_fps, name)
        except Exception as e:
            raise CameraInitError(f'Failed to initialize camera: {str(e)}')
            
        # Get ID from registry
        camera_id = await camera_registry.add_camera(camera)
        
        # Setup camera with assigned ID
        try:
            camera.setup(camera_id)
        except Exception as e:
            await camera_registry.remove_camera(camera_id)
            raise CameraInitError(f'Failed to setup camera: {str(e)}')
            
        # Start the camera
        try:
            await camera.start()
        except Exception as e:
            await camera_registry.remove_camera(camera_id)
            raise CameraInitError(f'Failed to start camera: {str(e)}')
        
        logger.info(f'Camera {name} (ID: {camera_id}) added successfully')
        return jsonify({
            'status': 'success',
            'camera_id': camera_id,
            'message': f'Camera {name} added successfully'
        })
    except Exception as e:
        logger.error(f"Failed to add camera: {str(e)}", exc_info=True)
        raise

@app.route('/remove_camera/<int:camera_id>', methods=['POST'])
@handle_camera_errors
async def remove_camera(camera_id):
    """Remove a camera from the registry"""
    logger.info(f"Attempting to remove camera {camera_id}")
    
    if not await camera_registry.remove_camera(camera_id):
        logger.warning(f"Attempt to remove non-existent camera {camera_id}")
        raise CameraError(f'Camera {camera_id} not found', 404)
    
    logger.info(f"Successfully removed camera {camera_id}")
    return jsonify({
        'status': 'success',
        'message': f'Camera {camera_id} removed successfully'
    })

@app.route('/list_cameras', methods=['GET'])
async def list_cameras():
    cameras = camera_registry.list_cameras()
    return jsonify({
        'status': 'success',
        'cameras': [{
            'id': cid,
            'name': camera.name,
            'source': camera.source,
            'fps': float(camera.status_array[1]),
            'is_recording': bool(camera.status_array[0])
        } for cid, camera in cameras]
    })

@app.route('/')
async def index():
    return await render_template('index.html')

@app.route('/start_recording', methods=['POST'])
@handle_camera_errors
async def start_recording_route():
    """Start recording for a camera with optional parameters"""
    data = await request.get_json()
    camera_id = data.get('camera_id')
    duration = data.get('duration')  # Optional duration in seconds
    quality = data.get('quality')    # Optional quality settings
    
    if camera_id is None:
        raise CameraError('Camera ID is required', 400)
        
    camera = camera_registry.get_camera(camera_id)
    if not camera:
        logger.warning(f"Attempt to start recording for non-existent camera {camera_id}")
        raise CameraError(f'Camera {camera_id} not found', 404)
    
    recording_info = await recording_manager.queue_recording(
        camera, 
        duration=duration,
        quality=quality
    )
    
    return jsonify({
        'status': 'success',
        'message': 'Recording queued successfully',
        'recording_info': recording_info
    })

@app.route('/recording_status/<int:camera_id>', methods=['GET'])
@handle_camera_errors
async def get_recording_status(camera_id):
    """Get recording status for a camera"""
    camera = camera_registry.get_camera(camera_id)
    if not camera:
        raise CameraError(f'Camera {camera_id} not found', 404)
        
    status = recording_manager.get_recording_status(camera_id)
    return jsonify({
        'status': 'success',
        'recording_status': status
    })

@app.route('/stop_recording', methods=['POST'])
@handle_camera_errors
async def stop_recording_route():
    """Stop recording for a camera specified in request body"""
    try:
        data = await request.get_json()
        camera_id = data.get('camera_id')
        
        if camera_id is None:
            raise CameraError('Camera ID is required', 400)
            
        camera = camera_registry.get_camera(camera_id)
        if not camera:
            logger.warning(f"Attempt to stop recording for non-existent camera {camera_id}")
            raise CameraError(f'Camera {camera_id} not found', 404)
            
        success, message = await camera.stop_recording()
        if not success:
            logger.error(f"Failed to stop recording for camera {camera_id}: {message}")
            raise RecordingError(f"Failed to stop recording: {message}")
        
        logger.info(f"Stopped recording for camera {camera_id}: {message}")
        return jsonify({
            'status': 'success',
            'message': message
        })
    except Exception as e:
        logger.error(f"Error in stop_recording_route: {str(e)}", exc_info=True)
        raise

@app.route('/recordings/<int:camera_id>', methods=['GET'])
async def list_recordings(camera_id):
    camera = camera_registry.get_camera(camera_id)
    if camera:
        recordings = []
        try:
            for file in os.listdir(camera.recording_dir):
                if file.endswith('.mp4'):
                    file_path = os.path.join(camera.recording_dir, file)
                    recordings.append({
                        'filename': file,
                        'size': os.path.getsize(file_path),
                        'created': datetime.fromtimestamp(
                            os.path.getctime(file_path)
                        ).isoformat()
                    })
            return jsonify({
                'status': 'success',
                'recordings': recordings
            })
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Error listing recordings: {str(e)}'
            }), 500
    return jsonify({
        'status': 'error',
        'message': 'Camera not found'
    }), 404

@app.route('/check_cameras', methods=['GET'])
async def check_cameras():
    """Endpoint to check available cameras"""
    available_cameras = []
    for i in range(10):  # Check first 10 indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            properties = {
                'index': i,
                'width': cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                'height': cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                'fps': cap.get(cv2.CAP_PROP_FPS)
            }
            available_cameras.append(properties)
            cap.release()
    
    return jsonify({
        'status': 'success',
        'available_cameras': available_cameras
    })

@app.route('/api-docs', methods=['GET'])
async def api_docs():
    """Return OpenAPI documentation"""
    return jsonify({
        'paths': {
            '/add_camera': {
                'post': {
                    'summary': 'Add a new camera',
                    'parameters': [
                        {'name': 'source', 'type': 'string', 'required': True},
                        {'name': 'name', 'type': 'string'}
                    ]
                }
            },
            # Add documentation for other endpoints
        }
    })

def create_app():
    # Simplify create_app to avoid manager issues
    logging.basicConfig(level=logging.WARNING)
    os.makedirs(RECORDING_DIR, exist_ok=True)
    return app

def find_free_port(start_port=5000):
    """Find a free port starting from start_port"""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        port = start_port
        while port < 65535:
            try:
                s.bind(('', port))
                return port
            except OSError:
                port += 1
        raise RuntimeError("No free ports available")

def cleanup_on_exit():
    """Ensure proper cleanup on exit"""
    logger.info("Starting cleanup on exit...")
    try:
        # Create event loop for async cleanup
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Get list of cameras before cleanup
        camera_ids = list(camera_registry._cameras.keys())
        
        # First stop all cameras
        for camera_id in camera_ids:
            try:
                loop.run_until_complete(camera_registry.remove_camera(camera_id))
            except Exception as e:
                logger.error(f"Error during camera cleanup: {e}")

        # Clean up process manager
        try:
            loop.run_until_complete(process_manager.cleanup_all())
        except Exception as e:
            logger.error(f"Error cleaning up processes: {e}")

        # Final shared memory cleanup
        for camera_id in camera_ids:
            try:
                cleanup_shared_memory(camera_id)
            except Exception as e:
                logger.error(f"Error in final shared memory cleanup: {e}")

        # Close the event loop
        try:
            loop.close()
        except Exception as e:
            logger.error(f"Error closing event loop: {e}")

    except Exception as e:
        logger.error(f"Error during exit cleanup: {e}")
    finally:
        logger.info("Cleanup completed, exiting...")

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, initiating shutdown...")
    cleanup_on_exit()
    sys.exit(0)  # Move sys.exit here

# Register signal handlers
def setup_signal_handlers():
    signals = [signal.SIGINT, signal.SIGTERM]
    for sig in signals:
        try:
            signal.signal(sig, signal_handler)
        except Exception as e:
            logger.error(f"Error setting up signal handler for {sig}: {e}")

# Update main function
def main():
    try:
        setup_signal_handlers()
        port = config.server.port or find_free_port()
        app = create_app()
        logger.info(f"Starting server on port {port}")
        
        # Register cleanup handler
        atexit.register(cleanup_on_exit)
        
        # Start the application
        app.run(
            host=config.server.host,
            port=port,
            debug=config.server.debug
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        cleanup_on_exit()

if __name__ == '__main__':
    freeze_support()  # Add this back
    main()

class RecordingManager:
    async def cleanup_old_recordings(self):
        """Automatically cleanup old recordings based on config"""
        try:
            for camera_id in camera_registry._cameras.copy().keys():
                camera = camera_registry.get_camera(camera_id)
                if not camera:
                    continue
                
                # Check storage limits
                await self._enforce_storage_limits(camera)
                
                # Cleanup temporary files
                await self._cleanup_temp_files(camera)
                
        except Exception as e:
            logger.error(f"Error in automatic cleanup: {str(e)}")

    async def _enforce_storage_limits(self, camera):
        """Enforce storage limits for recordings"""
        try:
            total_size = 0
            all_recordings = []
            
            for root, _, files in os.walk(camera.recording_dir):
                for file in files:
                    if file.endswith('.mp4'):
                        path = Path(root) / file
                        stat = path.stat()
                        all_recordings.append((stat.st_mtime, path))
                        total_size += stat.st_size

            # Convert GB to bytes
            max_size = config.recording.max_storage_gb * 1024**3
            if total_size > max_size:
                # Sort by oldest first
                all_recordings.sort()
                while total_size > max_size and len(all_recordings) > 0:
                    oldest_mtime, oldest_path = all_recordings.pop(0)
                    try:
                        file_size = oldest_path.stat().st_size
                        oldest_path.unlink()
                        total_size -= file_size
                        logger.info(f"Deleted old recording: {oldest_path}")
                    except Exception as e:
                        logger.error(f"Error deleting recording: {str(e)}")
                        
        except Exception as e:
            logger.error(f"Storage limit enforcement error: {str(e)}")

    async def _cleanup_temp_files(self, camera):
        """Cleanup temporary files from failed recordings"""
        try:
            temp_dir = Path("temp")
            if temp_dir.exists():
                for file in temp_dir.glob("*.tmp"):
                    try:
                        file.unlink()
                        logger.info(f"Cleaned up temp file: {file}")
                    except Exception as e:
                        logger.error(f"Error cleaning temp file: {str(e)}")
        except Exception as e:
            logger.error(f"Temp file cleanup error: {str(e)}")

async def run_maintenance_tasks():
    """Run periodic maintenance tasks"""
    while True:
        try:
            # Run every 6 hours
            await asyncio.sleep(6 * 3600)
            
            logger.info("Running maintenance tasks...")
            
            # Cleanup old recordings
            await recording_manager.cleanup_old_recordings()
            
            # Cleanup temporary files
            cleanup_temp_files()
            
            # Check system resources
            await check_system_health()
            
            # Add to existing maintenance tasks
            await perform_database_maintenance()
            
        except Exception as e:
            logger.error(f"Maintenance task error: {str(e)}")

async def check_system_health():
    """Check system resources and log warnings"""
    try:
        # Get disk usage
        disk_usage = shutil.disk_usage("/")
        if disk_usage.free < 1e9:  # 1GB remaining
            logger.warning("Low disk space! Free space: {:.2f}GB".format(
                disk_usage.free / 1e9))
            
        # Get memory usage
        import psutil
        mem = psutil.virtual_memory()
        if mem.percent > 90:
            logger.warning("High memory usage: {:.1f}% used".format(mem.percent))
            
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")

def cleanup_temp_files():
    """Cleanup temporary files from failed recordings"""
    try:
        temp_dir = Path("temp")
        if temp_dir.exists():
            for file in temp_dir.glob("*.tmp"):
                try:
                    file.unlink()
                    logger.info(f"Cleaned up temp file: {file}")
                except Exception as e:
                    logger.error(f"Error cleaning temp file: {str(e)}")
    except Exception as e:
        logger.error(f"Temp file cleanup error: {str(e)}")

async def perform_database_maintenance():
    """Perform database cleanup and optimization"""
    try:
        logger.info("Running database maintenance...")
        # Example tasks:
        # - Cleanup old logs
        # - Optimize indexes
        # - Backup database
        # - Vacuum/defragment
        logger.info("Database maintenance completed")
    except Exception as e:
        logger.error(f"Database maintenance failed: {str(e)}")