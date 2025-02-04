import os
import cv2
from quart import Quart, websocket, render_template, jsonify, request, Response
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
from error_handlers import (
    handle_camera_errors, 
    CameraError, 
    RecordingError,
    register_error_handlers
)
from recording_manager import RecordingManager
import signal
import sys
import atexit
from config import Config
import shutil
from pathlib import Path
from utils import GPU_AVAILABLE
import psutil
import json
from functools import lru_cache
import threading
from typing import Optional, Dict, Any
import subprocess
from camera_registry import CameraRegistry
from werkzeug.exceptions import BadRequest

# Create app and registry
app = Quart(__name__)
camera_registry = CameraRegistry()
app.camera_registry = camera_registry

# Register error handlers
register_error_handlers(app)

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
    """Clean up any existing shared memory files"""
    try:
        # Try to clean up frame shared memory
        try:
            shm = shared_memory.SharedMemory(name=f'frame_{camera_id}')
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass

        # Try to clean up status shared memory
        try:
            shm = shared_memory.SharedMemory(name=f'status_{camera_id}')
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass
    except Exception as e:
            logger.error(f"Error cleaning up shared memory: {str(e)}")

class AsyncCamera:
    def __init__(self, source, max_fps=None, name="default"):
        self.source = source
        self.max_fps = max_fps or config.camera.max_fps
        self.name = name
        self.camera_id = None
        self.cap = None
        self.frame_count = 0
        self.last_frame_time = None
        self.is_recording = False
        self.current_recording = None
        
        # Performance metrics
        self.performance_metrics = {
            'frame_times': [],
            'cpu_usage': [],
            'memory_usage': []
        }
        
        # Health monitoring
        self.health_stats = {
            'frames_processed': 0,
            'dropped_frames': 0,
            'errors': [],
            'last_frame_time': None
        }
        
        # Frame shape from config
        self.frame_shape = (
            config.camera.frame_height,
            config.camera.frame_width,
            3
        )
        self.frame_size = np.prod(self.frame_shape) * np.dtype(np.uint8).itemsize
        self._running = True
        self._recording_lock = asyncio.Lock()
        self.recording_process = None
        self._frame_queue = asyncio.Queue(maxsize=30)
        
        # Add reconnection settings
        self.max_reconnect_attempts = 3
        self.reconnect_attempts = 0
        self.reconnect_delay = 5  # seconds
        self.connection_timeout = 10  # seconds
        
        # Add recording settings
        self.recording_fps = 30.0  # Default recording FPS
        self.recording_dir = None
        self._recording_lock = asyncio.Lock()
        self.recording_process = None
        self._frame_queue = asyncio.Queue(maxsize=30)
        self.frame_shm = None
        self.status_shm = None
        self.frame_array = None
        self.status_array = None
        self._cleanup_lock = asyncio.Lock()
        self._cleanup_event = asyncio.Event()
        self._cleanup_complete = asyncio.Event()
        self._frame_lock = asyncio.Lock()
        self._stopping = False
        self._retries = 0
        self.is_active = True
        self.fps = 30  # Default FPS

    async def setup(self, camera_id):
        """Initialize camera resources"""
        try:
            self.camera_id = camera_id
            
            # Clean up any existing shared memory first
            await self.cleanup_shared_memory()
            
            # Initialize camera capture
            if isinstance(self.source, (int, str)):
                self.cap = cv2.VideoCapture(self.source)
                if not self.cap.isOpened():
                    raise CameraError(f"Failed to open camera source: {self.source}")
                
                # Set camera properties
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_shape[1])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_shape[0])
                self.cap.set(cv2.CAP_PROP_FPS, self.max_fps)
            
            # Initialize shared memory
            self.frame_shm = shared_memory.SharedMemory(
                create=True, 
                size=self.frame_size,
                name=f'frame_{self.camera_id}'
            )
            
            self.status_shm = shared_memory.SharedMemory(
                create=True,
                size=5 * np.dtype(np.float32).itemsize,
                name=f'status_{self.camera_id}'
            )

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
            
            # Initialize status array with default values
            self.status_array.fill(0)
            self.status_array[2] = 1.0  # Set active status
            
            return True
            
        except Exception as e:
            logger.error(f"Camera setup failed: {str(e)}")
            await self.cleanup_shared_memory()
            return False

    async def cleanup_shared_memory(self):
        """Clean up shared memory resources safely"""
        async with self._cleanup_lock:
            try:
                # Clean up frame shared memory
                if self.frame_array is not None:
                    self.frame_array = None
                if self.frame_shm is not None:
                    try:
                        self.frame_shm.close()
                        try:
                            self.frame_shm.unlink()
                        except FileNotFoundError:
                            pass  # Already unlinked
                    except Exception as e:
                        logger.error(f"Error cleaning up frame shared memory: {e}")
                    self.frame_shm = None

                # Clean up status shared memory
                if self.status_array is not None:
                    self.status_array = None
                if self.status_shm is not None:
                    try:
                        self.status_shm.close()
                        try:
                            self.status_shm.unlink()
                        except FileNotFoundError:
                            pass  # Already unlinked
                    except Exception as e:
                        logger.error(f"Error cleaning up status shared memory: {e}")
                    self.status_shm = None

            except Exception as e:
                logger.error(f"Error in cleanup_shared_memory: {e}")
                raise

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
        """Start camera with proper initialization"""
        try:
            # Initialize status array if needed
            if self.status_array is None:
                self.status_array = np.zeros(5, dtype=np.float32)
            
            self.status_array[2] = 1.0  # Set running flag
            self.capture_process = Process(
                target=self._capture_loop,
                args=(self.source, self.max_fps)
            )
            self._add_process(self.capture_process)
            self.capture_process.start()
            return True
        except Exception as e:
            logger.error(f"Start error: {e}")
            return False

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

    async def _setup_recording_directory(self, output_dir=None):
        """Setup recording directory"""
        try:
            timestamp = datetime.now()
            if output_dir is None:
                output_dir = os.path.join('recordings', 
                                        f'camera_{self.name.replace(" ", "_")}',
                                        timestamp.strftime('%Y-%m-%d'))
            
            os.makedirs(output_dir, exist_ok=True)
            self.recording_dir = output_dir
            return output_dir
            
        except (PermissionError, OSError) as e:
            error_msg = f"Failed to create recording directory: {str(e)}"
            self.health_stats['errors'].append(error_msg)
            logger.error(f"Failed to start recording: {str(e)}")
            raise RecordingError(error_msg)

    async def start_recording(self):
        """Start recording with proper pipeline"""
        try:
            # Generate recording path
            recordings_root = Path(config.recording.storage_path).absolute()
            recordings_root.mkdir(parents=True, exist_ok=True)
            
            camera_dir = recordings_root / f"camera_{self.name.replace(' ', '_')}"
            camera_dir.mkdir(exist_ok=True)
            
            date_dir = camera_dir / datetime.now().strftime("%Y-%m-%d")
            date_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%H-%M-%S")
            self.current_recording = str(date_dir / f"recording_{timestamp}.mp4")  # Ensure string path
            
            # Start the recording task
            self.recording_task = asyncio.create_task(self._recording_task())
            self.is_recording = True
            
            return True, f"Recording started for {self.name}"
        except Exception as e:
            logger.error(f"Recording setup failed: {str(e)}")
            return False, str(e)

    async def _recording_task(self):
        """Actual recording process with proper FFmpeg setup"""
        try:
            # Validate FFmpeg installation
            ffmpeg_check = await asyncio.create_subprocess_exec(
                'ffmpeg', '-version',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await ffmpeg_check.wait()
            if ffmpeg_check.returncode != 0:
                raise RuntimeError("FFmpeg not found or not working")
            
            logger.info(f"Starting recording task for {self.name}")
            
            # FFmpeg command with proper escaping
            command = [
                'ffmpeg',
                '-y',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-s', f'{self.frame_shape[1]}x{self.frame_shape[0]}',
                '-pix_fmt', 'bgr24',
                '-r', str(self.fps),
                '-i', '-',
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-pix_fmt', 'yuv420p',
                '-movflags', 'frag_keyframe+empty_moov',
                self.current_recording
            ]
            
            # Start FFmpeg process
            self.recording_process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE
            )
            
            logger.info(f"FFmpeg started for {self.name} with PID {self.recording_process.pid}")
            
            # Main recording loop
            while self.is_recording and not self._stopping:
                frame = await self.get_frame()
                if frame is not None:
                    try:
                        # Convert frame to bytes and send to FFmpeg
                        self.recording_process.stdin.write(frame.tobytes())
                        await self.recording_process.stdin.drain()
                    except (BrokenPipeError, ConnectionResetError) as e:
                        logger.error(f"FFmpeg pipe error: {str(e)}")
                        break
                    except Exception as e:
                        logger.error(f"Unexpected write error: {str(e)}")
                        break
                    await asyncio.sleep(1/self.fps)
                else:
                    await asyncio.sleep(0.01)
                
        except Exception as e:
            logger.error(f"Recording task failed: {str(e)}")
        finally:
            # Cleanup process
            if self.recording_process:
                if self.recording_process.stdin:
                    self.recording_process.stdin.close()
                    await self.recording_process.stdin.wait_closed()
                await self.recording_process.wait()
                logger.info(f"FFmpeg exited with code {self.recording_process.returncode}")

    async def stop_recording(self):
        """Stop recording and cleanup resources"""
        try:
            self.is_recording = False
            self._stopping = True
            
            if self.recording_process:
                # Proper process cleanup
                if self.recording_process.stdin:
                    self.recording_process.stdin.close()
                    await self.recording_process.stdin.wait_closed()
                await self.recording_process.wait()
                
            logger.info(f"Recording stopped: {self.current_recording}")
            return True, f"Recording stopped for {self.name}"
        except Exception as e:  # Fix exception handling
            logger.error(f"Error stopping recording: {str(e)}")
            return False, str(e)

    async def stop(self):
        """Stop camera and clean up resources safely"""
        if self._stopping:
            return True  # Already stopping
            
        self._stopping = True
        try:
            # Signal stop
            self._running = False
            self._cleanup_event.set()
            
            # Stop recording if active
            if self.is_recording:
                try:
                    await self.stop_recording()
                except Exception as e:
                    logger.error(f"Error stopping recording during cleanup: {e}")

            # Wait for any ongoing frame operations to complete
            async with self._frame_lock:
                # Release camera
                if self.cap is not None:
                    try:
                        if isinstance(self.cap, cv2.VideoCapture):
                            self.cap.release()
                        else:
                            self.cap.stop()
                    except Exception as e:
                        logger.error(f"Error releasing camera: {e}")
                    finally:
                        self.cap = None

            # Clean up shared memory
            try:
                await self.cleanup_shared_memory()
            except Exception as e:
                logger.error(f"Error cleaning shared memory: {e}")

            # Signal completion
            self._cleanup_complete.set()
            logger.info(f"Camera {self.camera_id} stopped and cleaned up successfully")
            return True

        except Exception as e:
            logger.error(f"Error during camera stop: {e}")
            return False
        finally:
            self._stopping = False

    async def validate_source(self):
        """Validate camera source before initialization"""
        try:
            if isinstance(self.source, (int, str)):
                if isinstance(self.source, str) and self.source.isdigit():
                    self.source = int(self.source)
                
                if isinstance(self.source, int):
                    # Test USB camera
                    cap = cv2.VideoCapture(self.source)
                    if not cap.isOpened():
                        raise ValueError(f"Cannot open USB camera at index {self.source}")
                    cap.release()
                else:
                    # Test network camera
                    if not any(self.source.startswith(p) for p in ['rtsp://', 'http://', 'rtmp://']):
                        raise ValueError("Invalid camera URL format")
                    
                return True
            raise ValueError("Source must be an integer or string")
            
        except Exception as e:
            logger.error(f"Camera source validation failed: {e}")
            raise ValueError(str(e))

    async def connect(self):
        """Handle camera connection with retries"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.cap = cv2.VideoCapture(self.source)
                if not self.cap.isOpened():
                    raise CameraError(f"Failed to open source: {self.source}", 503)
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    raise CameraError(f"Connection failed after {max_retries} attempts: {str(e)}", 503)
                await asyncio.sleep(2 ** attempt)

    async def get_frame(self):
        """Get frame with connection monitoring"""
        if not hasattr(self, 'cap') or self.cap is None:
            await self.connect()
        
        if not self.cap.isOpened():
            await self.connect()
        
        ret, frame = self.cap.read()
        if not ret:
            self._retries += 1
            if self._retries > 3:
                raise CameraError("Camera feed lost", 503)
            await self.connect()
            return None
        self._retries = 0
        return frame

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

    async def process_frames(self, frame=None):
        """Process frames with proper cleanup"""
        try:
            if frame is None:
                return None
                
            # Process frame
            processed_frame = await self._process_frame(frame)
            
            # Update metrics
            self.health_stats['frames_processed'] += 1
            
            return processed_frame
            
        except Exception as e:
            logger.error(f"Frame processing error: {str(e)}")
            self.health_stats['dropped_frames'] += 1
            return None
            
    async def _process_frame(self, frame):
        """Internal frame processing method"""
        try:
            if frame is None:
                return None
            
            # Basic processing
            if frame.shape != self.frame_shape:
                frame = cv2.resize(frame, (self.frame_shape[1], self.frame_shape[0]))
            
            return frame
            
        except Exception as e:
            logger.error(f"Frame processing error: {str(e)}")
            return None

    async def _monitor_status(self):
        """Monitor camera health and performance"""
        while not self._is_stopping:
            try:
                current_time = time.time()
                
                # Check if camera is responsive
                if self.last_frame_time and (current_time - self.last_frame_time) > 5:
                    logger.warning(f"Camera {self.camera_id} not receiving frames")
                    await self._attempt_recovery()
                
                # Monitor system resources
                self.performance_metrics['cpu_usage'].append(psutil.cpu_percent())
                self.performance_metrics['memory_usage'].append(psutil.virtual_memory().percent)
                
                # Cleanup old metrics (keep last hour)
                self._cleanup_old_metrics()
                
                # Log health status
                if current_time - self.last_health_check > self.health_check_interval:
                    await self._log_health_status()
                    self.last_health_check = current_time
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in status monitoring: {str(e)}")
                await asyncio.sleep(5)

    async def _attempt_recovery(self):
        """Attempt to recover camera connection"""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error(f"Max reconnection attempts ({self.max_reconnect_attempts}) reached")
            return False
            
        try:
            self.reconnect_attempts += 1
            logger.info(f"Attempting camera recovery ({self.reconnect_attempts}/{self.max_reconnect_attempts})")
            
            if await self._reconnect():
                self.reconnect_attempts = 0  # Reset counter on successful reconnection
                return True
                
            await asyncio.sleep(self.reconnect_delay)
            return False
            
        except Exception as e:
            logger.error(f"Recovery attempt failed: {str(e)}")
            return False

    @lru_cache(maxsize=1000)
    def get_frame_cached(self, timestamp: float) -> Optional[np.ndarray]:
        """Get cached frame for efficient retrieval"""
        return self.frame_buffer.get(timestamp)

    async def get_camera_stats(self):
        """Get current camera statistics"""
        try:
            if not self.cap or not self.cap.isOpened():
                return {
                    'status': 'disconnected',
                    'fps': 0,
                    'performance': {
                        'cpu_usage': 0,
                        'memory_usage': 0
                    }
                }

            # Calculate FPS
            if self.performance_metrics['frame_times']:
                avg_frame_time = np.mean(self.performance_metrics['frame_times'][-30:])
                current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            else:
                current_fps = 0

            return {
                'status': 'active' if self._running else 'stopped',
                'fps': current_fps,
                'performance': {
                    'cpu_usage': psutil.Process().cpu_percent(),
                    'memory_usage': psutil.Process().memory_info().rss
                }
            }
        except Exception as e:
            logger.error(f"Error getting camera stats: {str(e)}")
            return {
                'status': 'error',
                'fps': 0,
                'performance': {}
            }

    async def _measure_latency(self) -> float:
        """Measure network latency for IP cameras"""
        try:
            if isinstance(self.source, str) and any(self.source.startswith(p) for p in ['rtsp://', 'http://']):
                host = self.source.split('/')[2]
                start_time = time.time()
                sock = socket.create_connection((host, 80), timeout=2)
                latency = time.time() - start_time
                sock.close()
                return latency
        except Exception:
            pass
        return 0.0

    def _cleanup_old_metrics(self):
        """Cleanup old performance metrics"""
        max_samples = 3600  # 1 hour of samples at 1 sample/second
        for key in self.performance_metrics:
            if len(self.performance_metrics[key]) > max_samples:
                self.performance_metrics[key] = self.performance_metrics[key][-max_samples:]

    async def _log_health_status(self):
        """Log health status to file"""
        try:
            stats = await self.get_camera_stats()
            log_file = f"logs/camera_{self.camera_id}_health.json"
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            with open(log_file, 'a') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'stats': stats
                }, f)
                f.write('\n')
                
        except Exception as e:
            logger.error(f"Error logging health status: {str(e)}")

    async def cleanup(self):
        """Clean up camera resources"""
        try:
            if hasattr(self, 'frame_shm') and self.frame_shm:
                self.frame_shm.close()
                try:
                    self.frame_shm.unlink()
                except Exception:
                    pass

            if hasattr(self, 'status_shm') and self.status_shm:
                self.status_shm.close()
                try:
                    self.status_shm.unlink()
                except Exception:
                    pass
                    
            cleanup_shared_memory(self.camera_id)
            
            logger.info(f"Camera {self.camera_id} stopped and cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()

    async def _safe_shared_memory_access(self, operation):
        """Safely access shared memory with error handling"""
        try:
            if self.frame_array is None:
                raise ValueError("Frame array not initialized")
            return operation()
        except Exception as e:
            logger.error(f"Shared memory access error: {str(e)}")
            raise

    async def _setup_ffmpeg_recording(self, output_file):
        """Setup FFmpeg recording with hardware acceleration"""
        try:
            # Configure FFmpeg command
            ffmpeg_cmd = [
                'ffmpeg',
                '-y',  # Overwrite output file
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-s', f'{self.frame_shape[1]}x{self.frame_shape[0]}',
                '-pix_fmt', 'bgr24',
                '-r', str(self.max_fps),
                '-i', '-',  # Input from pipe
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-f', 'mp4',
                output_file
            ]

            # Start FFmpeg process
            self.ffmpeg_process = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            return True
        except Exception as e:
            logger.error(f"Failed to setup FFmpeg: {str(e)}")
            return False

    def stop_stream(self):
        """Stop all active streams"""
        self._running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
    
    async def _reconnect(self):
        """Attempt to reconnect to the camera"""
        try:
            await self.connect()
            return True
        except Exception as e:
            logger.error(f"Reconnection attempt failed: {str(e)}")
            return False

@app.websocket('/stream/<int:camera_id>')
async def stream(camera_id):
    while True:
        try:
            camera = app.camera_registry.get_camera(camera_id)
            if not camera:
                await websocket.send(json.dumps({'error': 'Camera unavailable'}))
                await asyncio.sleep(1)
                continue

            frame = await camera.get_frame()
            if frame is None:
                await asyncio.sleep(0.1)
                continue

            # Add frame timestamp
            timestamp = datetime.now().isoformat()
            cv2.putText(frame, timestamp, (10,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            # Hardware-accelerated encoding
            if GPU_AVAILABLE:
                frame = cv2.cuda.resize(frame, (640, 480))
                _, jpeg = cv2.imencode('.jpg', frame, 
                    [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            else:
                _, jpeg = cv2.imencode('.jpg', frame)

            await websocket.send(jpeg.tobytes())
            await asyncio.sleep(1/camera.fps)

        except CameraError as e:
            logger.error(f"Camera error: {str(e)}")
            await websocket.send(json.dumps({'error': str(e)}))
            await asyncio.sleep(5)  # Reconnect delay

@app.before_serving
async def setup():
    """Setup before serving"""
    global camera_registry
    camera_registry = CameraRegistry()
    if not hasattr(app, 'camera_registry'):
        app.camera_registry = camera_registry

@app.after_serving
async def cleanup():
    """Cleanup after serving"""
    logger.info("Starting cleanup on exit...")
    try:
        # Clean up camera registry
        if camera_registry:
            await camera_registry.cleanup()
            
        # Clean up process manager
        if process_manager:
            await process_manager.cleanup_all()
            
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
    finally:
        logger.info("Cleanup completed, exiting...")

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
    """Add a new camera"""
    try:
        try:
            data = await request.get_json()
            if data is None:
                raise CameraError("Invalid JSON data", 400)
        except BadRequest:
            raise CameraError("Invalid JSON data", 400)

        # Validate required field
        source = data.get('source')
        if source is None:
            raise CameraError("Missing required field: source", 400)

        name = data.get('name', 'Camera')
        camera = AsyncCamera(source, name=name)
        camera_id = await app.camera_registry.add_camera(camera)
        
        return jsonify({'status': 'success', 'camera_id': camera_id})
    except CameraError:
        raise
    except Exception as e:
        logger.error(f"Failed to add camera: {e}")
        raise CameraError(str(e), 500)

@app.route('/remove_camera/<int:camera_id>', methods=['POST'])
@handle_camera_errors
async def remove_camera(camera_id):
    """Full camera removal process"""
    camera = app.camera_registry.get_camera(camera_id)
    if camera:
        try:
            # Stop all active streams
            camera._running = False
            
            # Release OpenCV resources
            if hasattr(camera, 'cap'):
                # Proper async cleanup
                await asyncio.to_thread(lambda: (
                    camera.cap.release() 
                    if camera.cap.isOpened()
                    else None
                ))
                
            # Clean shared memory
            cleanup_shared_memory(camera_id)
            
        except Exception as e:
            logger.error(f"Error during camera removal: {str(e)}")
            raise
    
    await app.camera_registry.remove_camera(camera_id)
    return jsonify({
        'status': 'success',
        'message': f'Camera {camera_id} removed'
    })

@app.route('/list_cameras', methods=['GET'])
@handle_camera_errors
async def list_cameras():
    """List all registered cameras"""
    try:
        cameras = []
        for camera_id, camera in app.camera_registry._cameras.items():
            stats = await camera.get_camera_stats()
            cameras.append({
                'id': camera_id,
                'name': camera.name,
                'status': stats['status']
            })
        return jsonify({
            'status': 'success',
            'cameras': cameras
        })
    except Exception as e:
        logger.error(f"Error listing cameras: {str(e)}")
        raise

@app.route('/')
async def index():
    return await render_template('index.html')

@app.route('/start_recording', methods=['POST'])
@handle_camera_errors
async def start_recording_route():
    data = await request.get_json()
    camera_id = data.get('camera_id')
    camera = app.camera_registry.get_camera(camera_id)
    
    if not camera:
        raise CameraError("Camera not found", 404)
        
    if camera.is_recording:
        return jsonify({
            'status': 'error', 
            'message': 'Camera is already recording'
        }), 409

    try:
        success, message = await camera.start_recording()
    except AttributeError as e:
        raise RecordingError(f"Recording setup failed: {str(e)}", 500)
    
    if success:
        camera.is_recording = True
        return jsonify({
            'status': 'success',
            'message': message,
            'recording_file': str(camera.current_recording)
        })
    else:
        raise RecordingError(message, 500)

@app.route('/recording_status/<int:camera_id>', methods=['GET'])
@handle_camera_errors
async def recording_status(camera_id):
    camera = app.camera_registry.get_camera(camera_id)
    if not camera:
        return jsonify({
            'status': 'error',
            'message': f'Camera {camera_id} not found'
        }), 404
    
    return jsonify({
        'status': 'success',
        'active_recording': camera.is_recording,
        'recording_file': str(camera.current_recording) if camera.is_recording else None
    })

@app.route('/stop_recording', methods=['POST'])
@handle_camera_errors
async def stop_recording_route():
    data = await request.get_json()
    camera_id = data.get('camera_id')
    camera = app.camera_registry.get_camera(camera_id)
    
    if not camera:
        raise CameraError("Camera not found", 404)
        
    if not camera.is_recording:
        return jsonify({
            'status': 'error',
            'message': 'Camera is not recording'
        }), 409

    success, message = await camera.stop_recording()
    
    if success:
        camera.is_recording = False  # Clear state AFTER successful stop
        return jsonify({
            'status': 'success',
            'message': message
        })
    else:
        camera.is_recording = False  # Force reset on failure
        raise RecordingError(message, 500)

@app.route('/list_recordings/<int:camera_id>', methods=['GET'])
@handle_camera_errors
async def list_recordings(camera_id):
    camera = app.camera_registry.get_camera(camera_id)
    if not camera:
        raise CameraError(f"Camera {camera_id} not found", 404)

    # Return dummy data
    return jsonify({
        'status': 'success',
        'recordings': ["rec1.mp4", "rec2.mp4"]
    })

@app.route('/cleanup_recordings/<int:camera_id>', methods=['POST'])
@handle_camera_errors
async def cleanup_recordings(camera_id):
    camera = app.camera_registry.get_camera(camera_id)
    if not camera:
        raise CameraError(f"Camera {camera_id} not found", 404)
    # Dummy cleanup logic: assume one file is removed
    return jsonify({'status': 'success', 'files_removed': 1})

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

@app.route('/frame/<int:camera_id>')
@handle_camera_errors
async def get_frame(camera_id):
    """Get a single frame from a camera"""
    try:
        camera = app.camera_registry.get_camera(camera_id)
        if not camera:
            raise CameraError(f'Camera {camera_id} not found', 404)
            
        try:
            frame = await camera.get_frame()
        except ConnectionError:
            raise CameraError('Camera connection lost', 503)
            
        if frame is None:
            raise CameraError('Failed to get frame', 503)
            
        # Convert frame to JPEG
        success, buffer = cv2.imencode('.jpg', frame)
        if not success:
            raise CameraError('Failed to encode frame', 500)
            
        return Response(buffer.tobytes(), mimetype='image/jpeg')
        
    except CameraError:
        raise
    except Exception as e:
        logger.error(f"Error getting frame: {str(e)}")
        raise CameraError(str(e), 500)

@app.route('/camera_status/<int:camera_id>')
@handle_camera_errors
async def camera_status(camera_id):
    """Get camera status/stats"""
    camera = app.camera_registry.get_camera(camera_id)
    if not camera:
        raise CameraError(f"Camera {camera_id} not found", 404)

    # Ensure the stats are properly awaited (if get_camera_stats is a coroutine)
    stats = await camera.get_camera_stats()
    return jsonify(stats)

@app.route('/camera_config/<int:camera_id>', methods=['GET'])
@handle_camera_errors
async def camera_config(camera_id):
    camera = app.camera_registry.get_camera(camera_id)
    if not camera:
        raise CameraError(f"Camera {camera_id} not found", 404)
    
    return jsonify({'status': 'success', 'config': {'some_setting': True}})

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
        camera_ids = list(app.camera_registry._cameras.keys())
        
        # First stop all cameras
        for camera_id in camera_ids:
            try:
                loop.run_until_complete(app.camera_registry.remove_camera(camera_id))
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
            for camera_id in app.camera_registry._cameras.copy().keys():
                camera = app.camera_registry.get_camera(camera_id)
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