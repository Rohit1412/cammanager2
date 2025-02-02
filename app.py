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
import psutil
from subprocess import Popen, PIPE
import shlex
import fcntl
import select
from collections import deque
import mmap
import ctypes

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
    if isinstance(source, (int, str)):
        try:
            if isinstance(source, str):
                if any(source.startswith(p) for p in ['rtsp://', 'rtmp://', 'http://']):
                    return 'network', source
                source = int(source)
            
            # Test camera accessibility
            cap = cv2.VideoCapture(source)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    cap.release()
                    return 'usb', source
            cap.release()
            
            # Try CamGear as fallback
            from vidgear.gears import CamGear
            cap = CamGear(source=source).start()
            frame = cap.read()
            if frame is not None:
                cap.stop()
                return 'usb', source
            cap.stop()
            
        except Exception as e:
            logging.error(f"Error accessing camera: {str(e)}")
            
        raise ValueError(f"Unable to access camera source: {source}")
    
    raise ValueError(f"Unsupported camera source type: {type(source)}")

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
        
        # Initialize capture based on source type
        try:
            if isinstance(source, str) and any(source.startswith(p) for p in ['rtsp://', 'rtmp://', 'http://']):
                # Network stream
                self._init_network_stream(source)
            else:
                # Local camera
                self._init_local_camera(source)
                
        except Exception as e:
            logger.error(f"Failed to initialize camera: {str(e)}")
            raise CameraInitError(f"Camera initialization failed: {str(e)}")
        
        # Use config values for frame dimensions
        self.frame_shape = (
            config.camera.frame_height,
            config.camera.frame_width,
            3
        )
        self.frame_size = np.prod(self.frame_shape) * np.dtype(np.uint8).itemsize
        
        # Add frame buffer
        self.frame_buffer = deque(maxlen=30)  # Store last 30 frames
        self.mmap_buffer = None
        self.buffer_size = self.frame_shape[0] * self.frame_shape[1] * 3
        
        # Add these attributes
        self.recording_process = None
        self.is_recording = False
        self.current_recording = None
        self.recording_quality = None
        self._is_stopping = False
        self._processes = []  # Keep track of all child processes
        self._cleanup_lock = asyncio.Lock()
        self._shared_memory = []  # Track shared memory objects
        self.adaptive_fps = self.max_fps
        self.last_cpu_check = time.time()
        self.cpu_check_interval = 5.0  # Check CPU usage every 5s
        self.frame_counter = 0  # For skipping frames
        self.ffmpeg_process = None
        self.encoding_preset = 'ultrafast'  # Can be adjusted based on CPU capacity
        self.recording_status = Value('i', 0)  # Shared value for recording status
        self.recording_start_time = None
        
        # Add object detection
        self.object_detector = None
        self.detection_enabled = False
        self.last_detections = []
        
    def _track_shared_memory(self, shm):
        """Track shared memory for cleanup"""
        self._shared_memory.append(shm)

    def setup(self, camera_id):
        """Setup camera with memory-optimized buffers"""
        self.camera_id = camera_id
        try:
            # Create memory-mapped buffer instead of shared memory
            self.mmap_buffer = mmap.mmap(-1, self.buffer_size,
                                       flags=mmap.MAP_SHARED,
                                       prot=mmap.PROT_READ | mmap.PROT_WRITE)
            
            # Create numpy array from memory map
            self.frame_array = np.frombuffer(self.mmap_buffer, 
                                           dtype=np.uint8).reshape(self.frame_shape)
            
            # Status array using ctypes
            self.status_array = (ctypes.c_float * 3).from_buffer(
                mmap.mmap(-1, 12, flags=mmap.MAP_SHARED)
            )
            
            self.recording_dir = os.path.join('recordings', str(camera_id))
            os.makedirs(self.recording_dir, exist_ok=True)
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup camera buffers: {str(e)}")
            return False

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

    def should_process_frame(self):
        """
        Determine if the current frame should be processed.
        Skip frames if CPU usage is high.
        """
        cpu_usage = psutil.cpu_percent()
        # If CPU > 60%, only process every second frame
        if cpu_usage > 60:
            self.frame_counter += 1
            return (self.frame_counter % 2) == 0
        return True

    def _adjust_fps_by_cpu(self):
        """
        Dynamically adjust capture FPS based on CPU usage.
        Drops fps if CPU > 80%, gradually restores if CPU < 60%.
        """
        current_time = time.time()
        if current_time - self.last_cpu_check >= self.cpu_check_interval:
            cpu_usage = psutil.cpu_percent()
            if cpu_usage > 80:
                self.adaptive_fps = max(10, self.adaptive_fps - 5)
                logger.info(f"Reducing FPS to {self.adaptive_fps} due to high CPU usage: {cpu_usage}%")
            elif cpu_usage < 60 and self.adaptive_fps < self.max_fps:
                self.adaptive_fps = min(self.max_fps, self.adaptive_fps + 2)
                logger.info(f"Increasing FPS to {self.adaptive_fps} due to lower CPU usage: {cpu_usage}%")
            self.last_cpu_check = current_time

    def _capture_loop(self, source, max_fps):
        """Modified capture loop with object detection"""
        try:
            last_frame_time = time.time()
            consecutive_failures = 0
            max_failures = 10
            
            while self.status_array[2] > 0:
                current_time = time.time()
                frame_interval = 1.0 / self.adaptive_fps
                
                self._adjust_fps_by_cpu()
                
                if current_time - last_frame_time >= frame_interval:
                    frame = self._read_frame()
                    
                    if frame is not None:
                        if self.should_process_frame():
                            # Process frame (object detection happens in websocket stream)
                            processed_frame = frame.copy()
                            
                            # Update shared memory with the new frame
                            try:
                                self.frame_array[:] = processed_frame
                                self.frame_buffer.append(processed_frame)
                                
                                # Update status
                                self.status_array[1] = 1.0 / (current_time - last_frame_time)
                                last_frame_time = current_time
                                
                                # Write to FFmpeg if recording
                                if self.is_recording:
                                    self._write_frame_to_ffmpeg(processed_frame)
                                    
                                consecutive_failures = 0
                            except Exception as e:
                                logger.error(f"Error updating frame buffer: {str(e)}")
                                
                    else:
                        consecutive_failures += 1
                        if consecutive_failures >= max_failures:
                            raise ValueError("Too many consecutive failures")
                        time.sleep(0.05)
                        
        except Exception as e:
            logger.error(f"Error in capture loop: {str(e)}")
            self.status_array[2] = 0.0
        finally:
            self._cleanup_capture()

    def _read_frame(self):
        """Optimized frame reading"""
        try:
            if isinstance(self.cap, cv2.VideoCapture):
                ret, frame = self.cap.read()
                if not ret:
                    return None
            else:  # CamGear
                frame = self.cap.read()
                if frame is None:
                    return None
            
            # Resize if necessary
            if frame.shape != self.frame_shape:
                frame = cv2.resize(frame, 
                                 (self.frame_shape[1], self.frame_shape[0]),
                                 interpolation=cv2.INTER_AREA)
            
            return frame
        except Exception as e:
            logger.error(f"Error reading frame: {str(e)}")
            return None

    def _cleanup_capture(self):
        """Enhanced cleanup with memory management"""
        try:
            if self.mmap_buffer:
                self.mmap_buffer.close()
            
            self.frame_buffer.clear()
            
            if isinstance(self.cap, cv2.VideoCapture):
                self.cap.release()
            else:
                self.cap.stop()
                
        except Exception as e:
            logger.error(f"Error in cleanup: {str(e)}")

    async def start_recording(self, output_dir=None):
        """Start recording with FFmpeg"""
        if self.is_recording:
            return False, "Already recording"
            
        try:
            # Setup output directory
            if output_dir is None:
                output_dir = os.path.join('recordings', str(self.camera_id))
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate output filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = os.path.join(output_dir, f'cam_{self.camera_id}_{timestamp}.mp4')
            
            # Initialize FFmpeg
            if not self._setup_ffmpeg_recording(output_file):
                raise RecordingError("Failed to initialize FFmpeg recording")
            
            self.is_recording = True
            self.current_recording = output_file
            self.recording_status.value = 1
            self.recording_start_time = time.time()
            
            logger.info(f"Started recording to {output_file}")
            return True, f"Recording started: {output_file}"
            
        except Exception as e:
            logger.error(f"Failed to start recording: {str(e)}")
            self.is_recording = False
            self.recording_status.value = 0
            self.recording_start_time = None
            return False, str(e)

    def _setup_ffmpeg_recording(self, output_file):
        """Setup FFmpeg for hardware-accelerated recording"""
        try:
            # Base FFmpeg command with hardware acceleration
            if GPU_AVAILABLE:
                # Try NVIDIA GPU acceleration first
                encoder = '-c:v h264_nvenc'
                hw_accel = '-hwaccel cuda -hwaccel_output_format cuda'
            else:
                # Fallback to CPU with optimized settings
                encoder = '-c:v libx264'
                hw_accel = ''
            
            # Construct FFmpeg command
            cmd = f'ffmpeg {hw_accel} -f rawvideo -pix_fmt bgr24 ' \
                  f'-s {self.frame_shape[1]}x{self.frame_shape[0]} ' \
                  f'-r {self.adaptive_fps} -i pipe: ' \
                  f'{encoder} -preset {self.encoding_preset} ' \
                  f'-b:v 2M -maxrate 2M -bufsize 4M ' \
                  f'-g {int(self.adaptive_fps*2)} ' \
                  f'-threads 8 {output_file}'
            
            # Start FFmpeg process
            self.ffmpeg_process = Popen(
                shlex.split(cmd),
                stdin=PIPE,
                stderr=PIPE
            )
            
            # Set non-blocking mode for stderr
            flags = fcntl.fcntl(self.ffmpeg_process.stderr, fcntl.F_GETFL)
            fcntl.fcntl(self.ffmpeg_process.stderr, fcntl.F_SETFL, flags | os.O_NONBLOCK)
            
            logger.info(f"Started FFmpeg recording with command: {cmd}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup FFmpeg recording: {str(e)}")
            return False

    def _write_frame_to_ffmpeg(self, frame):
        """Write frame to FFmpeg process"""
        if self.ffmpeg_process and self.is_recording:
            try:
                self.ffmpeg_process.stdin.write(frame.tobytes())
            except Exception as e:
                logger.error(f"Error writing frame to FFmpeg: {str(e)}")
                asyncio.create_task(self.stop_recording())

    async def stop_recording(self):
        """Stop the current recording"""
        if not self.is_recording:
            return False, "Not recording"
            
        try:
            # Close FFmpeg process if it exists
            if self.ffmpeg_process:
                try:
                    # Close stdin to signal FFmpeg to finish
                    if self.ffmpeg_process.stdin:
                        self.ffmpeg_process.stdin.close()
                    
                    # Wait for FFmpeg to finish (with timeout)
                    self.ffmpeg_process.wait(timeout=5)
                except Exception as e:
                    logger.warning(f"Error closing FFmpeg process: {str(e)}")
                    # Force kill if necessary
                    self.ffmpeg_process.kill()
                
                self.ffmpeg_process = None
            
            recorded_file = self.current_recording
            self.current_recording = None
            self.is_recording = False
            self.recording_status.value = 0
            self.recording_start_time = None
            
            logger.info(f"Recording stopped successfully: {recorded_file}")
            return True, f"Recording stopped: {recorded_file}"
            
        except Exception as e:
            logger.error(f"Failed to stop recording: {str(e)}")
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

    def _init_network_stream(self, source):
        """Initialize network stream with FFmpeg"""
        ffmpeg_cmd = (
            f'ffmpeg -rtsp_transport tcp -i {source} '
            f'-vsync 0 -copyts '
            f'-thread_queue_size 4096 '
            f'-f rawvideo -pix_fmt bgr24 -'
        )
        self.cap = cv2.VideoCapture(ffmpeg_cmd)
        if not self.cap.isOpened():
            raise CameraInitError(f"Failed to open network stream: {source}")

    def _init_local_camera(self, source):
        """Initialize local camera with proper checks"""
        # First try direct OpenCV capture
        self.cap = cv2.VideoCapture(source)
        if self.cap.isOpened():
            logger.info(f"Successfully opened camera with OpenCV: {source}")
            return

        # If OpenCV fails, try CamGear
        try:
            from vidgear.gears import CamGear
            options = {
                "THREADED_QUEUE_MODE": False,  # Set to False as it's causing issues
                "THREAD_TIMEOUT": 2000,
            }
            self.cap = CamGear(source=source, logging=True, **options).start()
            
            # Verify CamGear stream
            test_frame = self.cap.read()
            if test_frame is None:
                raise CameraInitError("CamGear failed to read test frame")
                
            logger.info(f"Successfully opened camera with CamGear: {source}")
        except Exception as e:
            raise CameraInitError(f"Failed to initialize camera with both OpenCV and CamGear: {str(e)}")

    async def get_latest_frame(self):
        """Get the latest frame from buffer"""
        try:
            if len(self.frame_buffer) > 0:
                return self.frame_buffer[-1]
            return None
        except Exception as e:
            logger.error(f"Error accessing frame buffer: {str(e)}")
            return None

    async def enable_detection(self):
        """Enable object detection"""
        try:
            if self.object_detector is None:
                from object_detection import ObjectDetector
                self.object_detector = ObjectDetector()
                logger.info("Object detection initialized successfully")
            self.detection_enabled = True
            logger.info("Object detection enabled")
            return True, "Object detection enabled"
        except Exception as e:
            logger.error(f"Failed to enable detection: {str(e)}")
            self.detection_enabled = False
            return False, str(e)
            
    async def disable_detection(self):
        """Disable object detection"""
        try:
            self.detection_enabled = False
            logger.info("Object detection disabled")
            return True, "Object detection disabled"
        except Exception as e:
            logger.error(f"Error disabling detection: {str(e)}")
            return False, str(e)
        
    def _process_frame(self, frame):
        """Process frame with object detection if enabled"""
        try:
            if self.detection_enabled and self.object_detector:
                frame, detections = self.object_detector(frame)
                self.last_detections = detections
            return frame
        except Exception as e:
            logger.error(f"Error in frame processing: {str(e)}")
            return frame

@app.websocket('/stream/<int:camera_id>')
@handle_camera_errors
async def stream(camera_id):
    camera = camera_registry.get_camera(camera_id)
    if not camera:
        logger.warning(f"Attempt to stream non-existent camera ID: {camera_id}")
        raise CameraError(f'Camera {camera_id} not found', 404)
    
    try:
        logger.info(f"Starting stream for camera {camera_id}")
        while True:  # Changed from camera.status_array[2] > 0
            try:
                # Get frame from shared memory
                frame = np.array(camera.frame_array, copy=True)
                
                if frame is not None and frame.size > 0:
                    # Process frame if needed (object detection)
                    if camera.detection_enabled and camera.object_detector:
                        frame, _ = camera.object_detector(frame)
                    
                    # Encode and send frame
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    await websocket.send(buffer.tobytes())
                
                # Control frame rate
                await asyncio.sleep(1/30)  # 30 FPS max
                
            except Exception as e:
                logger.error(f"Frame processing error: {str(e)}")
                await asyncio.sleep(0.1)  # Brief pause on error
                
    except Exception as e:
        logger.error(f"Streaming error: {str(e)}")
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
        max_fps = data.get('max_fps', 30)
        
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
    data = await request.get_json()
    camera_id = data.get('camera_id')
    
    if camera_id is None:
        raise CameraError('Camera ID is required', 400)
        
    camera = camera_registry.get_camera(camera_id)
    if not camera:
        raise CameraError(f'Camera {camera_id} not found', 404)
    
    success, message = await camera.start_recording()
    
    if not success:
        raise RecordingError(f"Failed to start recording: {message}")
        
    return jsonify({
        'status': 'success',
        'message': message
    })

@app.route('/recording_status/<int:camera_id>', methods=['GET'])
@handle_camera_errors
async def get_recording_status(camera_id):
    """Get recording status for a camera"""
    camera = camera_registry.get_camera(camera_id)
    if not camera:
        raise CameraError(f'Camera {camera_id} not found', 404)
        
    status = {
        'is_recording': camera.is_recording,
        'current_file': camera.current_recording if camera.is_recording else None,
        'duration': time.time() - (camera.recording_start_time or time.time()) if camera.is_recording else 0,
        'fps': camera.adaptive_fps
    }
    
    return jsonify({
        'status': 'success',
        'recording_status': status
    })

@app.route('/stop_recording', methods=['POST'])
@handle_camera_errors
async def stop_recording_route():
    data = await request.get_json()
    camera_id = data.get('camera_id')
    
    if camera_id is None:
        raise CameraError('Camera ID is required', 400)
        
    camera = camera_registry.get_camera(camera_id)
    if not camera:
        raise CameraError(f'Camera {camera_id} not found', 404)
    
    success, message = await camera.stop_recording()
    
    if not success:
        raise RecordingError(f"Failed to stop recording: {message}")
        
    return jsonify({
        'status': 'success',
        'message': message
    })

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

@app.route('/enable_detection/<int:camera_id>', methods=['POST'])
@handle_camera_errors
async def enable_detection(camera_id):
    """Enable object detection for a camera"""
    camera = camera_registry.get_camera(camera_id)
    if not camera:
        raise CameraError(f'Camera {camera_id} not found', 404)
        
    success, message = await camera.enable_detection()
    
    if not success:
        raise CameraError(f"Failed to enable detection: {message}")
        
    return jsonify({
        'status': 'success',
        'message': message
    })

@app.route('/disable_detection/<int:camera_id>', methods=['POST'])
@handle_camera_errors
async def disable_detection(camera_id):
    """Disable object detection for a camera"""
    camera = camera_registry.get_camera(camera_id)
    if not camera:
        raise CameraError(f'Camera {camera_id} not found', 404)
        
    success, message = await camera.disable_detection()
    
    return jsonify({
        'status': 'success',
        'message': message
    })

@app.route('/detections/<int:camera_id>', methods=['GET'])
@handle_camera_errors
async def get_detections(camera_id):
    """Get latest detections for a camera"""
    camera = camera_registry.get_camera(camera_id)
    if not camera:
        raise CameraError(f'Camera {camera_id} not found', 404)
        
    return jsonify({
        'status': 'success',
        'detections': camera.last_detections
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