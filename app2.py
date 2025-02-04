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
import json
from functools import lru_cache
import threading
from typing import Optional, Dict, Any
import subprocess

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
        self._running = True  # Add this flag
        self._recording_lock = asyncio.Lock()  # Add lock for recording
        self.recording_process = None
        self._frame_queue = asyncio.Queue(maxsize=30)  # Buffer for recording
        
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
        self.recording_start_time = None
        self.recording_status = {
            'is_recording': False,
            'duration': 0,
            'message': '',
            'error': None
        }
        self.recording_task = None  # Add this to track the recording task
        
        # Add FFmpeg settings
        self.ffmpeg_process = None
        self.encoding_format = 'bgr24'  # Default OpenCV format
        self.recording_codec = 'libx264'
        self.recording_preset = 'ultrafast'

    async def setup(self, camera_id):
        """Initialize camera resources"""
        try:
            self.camera_id = camera_id
        
        # Clean up any existing shared memory first
            cleanup_shared_memory(camera_id)
        
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
                                        f'camera_{self.camera_id}',
                                        timestamp.strftime('%Y-%m-%d'))
            
            os.makedirs(output_dir, exist_ok=True)
            self.recording_dir = output_dir
            return output_dir
            
        except (PermissionError, OSError) as e:
            error_msg = f"Failed to create recording directory: {str(e)}"
            self.health_stats['errors'].append(error_msg)
            logger.error(f"Failed to start recording: {str(e)}")
            raise RecordingError(error_msg)

    async def start_recording(self, output_dir=None):
        """Start recording camera feed"""
        async with self._recording_lock:
            if self.is_recording:
                return False, "Already recording"
            
            try:
                # Verify FFmpeg installation
                try:
                    process = await asyncio.create_subprocess_exec(
                        'ffmpeg', '-version',
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    await process.communicate()
                    if process.returncode != 0:
                        raise RecordingError("FFmpeg is not properly installed")
                except FileNotFoundError:
                    raise RecordingError("FFmpeg is not installed")

                # Get a test frame to verify camera and frame format
                test_frame = await self.get_frame()
                if test_frame is None:
                    raise RecordingError("Unable to capture frames from camera")
                
                height, width = test_frame.shape[:2]
                
                # Setup recording directory
                output_dir = await self._setup_recording_directory(output_dir)
                timestamp = datetime.now()
                output_file = os.path.join(
                    output_dir,
                    f'recording_{timestamp.strftime("%H-%M-%S")}.mp4'
                )
                
                # Construct FFmpeg command with explicit pixel format
                command = [
                    'ffmpeg',
                    '-y',  # Overwrite output file
                    '-f', 'rawvideo',
                    '-vcodec', 'rawvideo',
                    '-s', f'{width}x{height}',
                    '-pix_fmt', self.encoding_format,
                    '-r', str(self.recording_fps),
                    '-i', 'pipe:0',
                    '-c:v', self.recording_codec,
                    '-preset', self.recording_preset,
                    '-pix_fmt', 'yuv420p',  # Ensure compatibility
                    '-crf', '23',
                    '-movflags', '+faststart',
                    output_file
                ]
                
                # Start FFmpeg process
                self.recording_process = await asyncio.create_subprocess_exec(
                    *command,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.PIPE
                )
                
                if not self.recording_process:
                    raise RecordingError("Failed to start FFmpeg process")
                
                # Initialize recording state
                self.current_recording = output_file
                self.recording_fps = float(self.recording_fps)
                self.recording_start_time = time.time()
                self.is_recording = True
                
                # Start recording task
                self.recording_task = asyncio.create_task(self._recording_task())
                
                logger.info(f"Recording started: {output_file} at {self.recording_fps} FPS")
                return True, f"Recording started: {output_file}"
                
            except Exception as e:
                logger.error(f"Failed to start recording: {str(e)}")
                self.is_recording = False
                self.current_recording = None
                if hasattr(self, 'recording_process') and self.recording_process:
                    self.recording_process.kill()
                    self.recording_process = None
                raise RecordingError(str(e))

    async def _recording_task(self):
        """Background task to handle recording"""
        frame_count = 0
        start_time = time.time()
        frame_interval = 1.0 / self.recording_fps
        next_frame_time = start_time
        
        try:
            while self.is_recording and self.recording_process:
                current_time = time.time()
                
                if current_time >= next_frame_time:
                    frame = await self.get_frame()
                    if frame is not None:
                        try:
                            # Ensure frame is in correct format
                            if not isinstance(frame, np.ndarray):
                                logger.error("Invalid frame format")
                                continue
                                
                            # Check if process is still alive
                            if self.recording_process.returncode is not None:
                                logger.error("FFmpeg process has terminated")
                                break
                                
                            # Write frame
                            try:
                                frame_bytes = frame.tobytes()
                                await self.recording_process.stdin.write(frame_bytes)
                                await self.recording_process.stdin.drain()
                                frame_count += 1
                                
                                # Log progress
                                if frame_count % 30 == 0:
                                    elapsed = time.time() - start_time
                                    fps = frame_count / elapsed
                                    logger.debug(f"Recording at {fps:.1f} FPS")
                                    
                            except (BrokenPipeError, ConnectionError) as e:
                                logger.error(f"FFmpeg pipe error: {e}")
                                break
                                
                        except Exception as e:
                            logger.error(f"Frame processing error: {e}")
                            continue
                            
                    next_frame_time += frame_interval
                    
                await asyncio.sleep(frame_interval * 0.1)  # Short sleep to prevent CPU overload
                
        except Exception as e:
            logger.error(f"Recording task error: {e}")
        finally:
            # Get FFmpeg output for debugging
            if self.recording_process:
                try:
                    stderr_data = await self.recording_process.stderr.read()
                    if stderr_data:
                        logger.debug(f"FFmpeg output: {stderr_data.decode()}")
                except Exception as e:
                    logger.error(f"Error reading FFmpeg output: {e}")
            
            await self._safe_stop_recording()

    def _update_recording_status(self, message):
        """Update recording status with message"""
        self.recording_status = {
            'is_recording': self.is_recording,
            'duration': time.time() - self.recording_start_time if self.recording_start_time else 0,
            'message': message,
            'error': message if 'error' in message.lower() else None
        }
        logger.info(f"Camera {self.camera_id}: {message}")

    async def _safe_stop_recording(self):
        """Safely stop recording and cleanup resources"""
        try:
            if self.recording_process:
                try:
                    if (hasattr(self.recording_process, 'stdin') and 
                        self.recording_process.stdin and 
                        not self.recording_process.stdin.is_closing()):
                        await self.recording_process.stdin.write_eof()
                        await self.recording_process.stdin.drain()
                        
                    # Wait for process to finish
                    try:
                        await asyncio.wait_for(self.recording_process.wait(), timeout=5.0)
                    except asyncio.TimeoutError:
                        logger.warning("FFmpeg process did not exit gracefully, forcing termination")
                        self.recording_process.terminate()
                        await asyncio.sleep(0.5)
                        if self.recording_process.returncode is None:
                            self.recording_process.kill()
                        
                except Exception as e:
                    logger.error(f"Error during FFmpeg cleanup: {e}")
                    if self.recording_process:
                        self.recording_process.kill()
                
                self.recording_process = None
            
            self.is_recording = False
            logger.info(f"Recording stopped: {self.current_recording}")
            self.current_recording = None
            
        except Exception as e:
            logger.error(f"Error in safe stop recording: {e}")
            self.is_recording = False
            self.recording_process = None
            self.current_recording = None

    async def stop_recording(self):
        """Stop the current recording"""
        async with self._recording_lock:
            if not self.is_recording:
                return False, "Not recording"
            
        try:
                await self._safe_stop_recording()
                return True, "Recording stopped successfully"
            
        except Exception as e:
                logger.error(f"Error stopping recording: {str(e)}")
                return False, str(e)

    async def stop(self):
        """Stop the camera and clean up resources"""
        try:
            self._running = False  # Set flag to stop frame capture
            
            # Stop recording if active
            if self.is_recording:
                    await self.stop_recording()

            # Release camera
            if self.cap and self.cap.isOpened():
                self.cap.release()
                self.cap = None
            
            # Clean up shared memory
            if hasattr(self, 'frame_shm') and self.frame_shm:
                try:
                    self.frame_shm.close()
                    self.frame_shm.unlink()
                except Exception as e:
                    logger.error(f"Error cleaning up frame shared memory: {e}")
                    
            if hasattr(self, 'status_shm') and self.status_shm:
                try:
                    self.status_shm.close()
                    self.status_shm.unlink()
                except Exception as e:
                    logger.error(f"Error cleaning up status shared memory: {e}")

            cleanup_shared_memory(self.camera_id)
            
            logger.info(f"Camera {self.camera_id} stopped and cleaned up successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during camera cleanup: {str(e)}")
            return False

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

    async def get_frame(self):
        """Get the latest frame with proper error handling"""
        try:
            if not self._running or not self.cap or not self.cap.isOpened():
                return None

            # Read frame with timeout
            ret, frame = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, self.cap.read),
                timeout=1.0
            )
            
            if not ret or frame is None:
                logger.warning("Failed to read frame")
                return None

            # Update frame metrics
            current_time = time.time()
            if self.last_frame_time:
                frame_time = current_time - self.last_frame_time
                self.performance_metrics['frame_times'].append(frame_time)
                if len(self.performance_metrics['frame_times']) > 30:
                    self.performance_metrics['frame_times'] = self.performance_metrics['frame_times'][-30:]
            
            self.last_frame_time = current_time
            self.frame_count += 1
            
            # Ensure frame is in correct format
            if frame.shape != self.frame_shape:
                frame = cv2.resize(frame, (self.frame_shape[1], self.frame_shape[0]))
            
            # Update shared memory
            self.frame_array[:] = frame.copy()
            
            return frame

        except asyncio.TimeoutError:
            logger.error("Frame capture timeout")
            return None
        except Exception as e:
            logger.error(f"Error capturing frame: {str(e)}")
            return None

    async def _reconnect(self):
        """Attempt to reconnect to the camera"""
        try:
            if self.cap:
                self.cap.release()
            
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                logger.error(f"Failed to reconnect to camera {self.camera_id}")
                return False
                
            # Reset camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_shape[1])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_shape[0])
            self.cap.set(cv2.CAP_PROP_FPS, self.max_fps)
            
            # Reset metrics
            self.reconnect_attempts = 0
            self.health_stats['errors'] = []
            
            return True
        except Exception as e:
            logger.error(f"Error during reconnection: {str(e)}")
            return False

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

@app.websocket('/stream/<int:camera_id>')
async def stream(camera_id):
    """Stream camera frames over websocket"""
    try:
        camera = camera_registry.get_camera(camera_id)
        if not camera:
                await websocket.send(json.dumps({
                    'error': f'Camera {camera_id} not found'
                }))
                return logger.info(f"Starting stream for camera {camera_id}")
        
        while camera._running:  # Check running flag
            try:
                frame = await camera.get_frame()
                if frame is not None:
                    # Convert frame to JPEG
                    success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if success:
                        # Send frame as binary
                        await websocket.send(buffer.tobytes())
                        if logger.getEffectiveLevel() <= logging.DEBUG:
                                    logger.debug(f"Frame sent for camera {camera_id}")
                        else:
                                    logger.error("Failed to encode frame")
                else:
                    if camera._running:  # Only log if camera should be running
                        logger.warning(f"No frame received from camera {camera_id}")
                        await websocket.send(json.dumps({'status': 'no_frame'}))
                    
                await asyncio.sleep(1/30)  # Limit to ~30 FPS
                
            except Exception as e:
                if camera._running:  # Only log if camera should be running
                    logger.error(f"Error streaming frame: {str(e)}")
                    await websocket.send(json.dumps({'error': str(e)}))
                break
                
    except Exception as e:
        logger.error(f"Stream error: {str(e)}")
        await websocket.send(json.dumps({'error': str(e)}))

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
async def add_camera():
    """Add a new camera"""
    try:
        data = await request.get_json()
        source = data.get('source')
        name = data.get('name', f'Camera_{len(camera_registry._cameras)}')
        
        camera = AsyncCamera(source=source, name=name)
        camera_id = len(camera_registry._cameras)
        
        # Make setup async and await it
        success = await camera.setup(camera_id)
        if not success:
            raise CameraError("Failed to setup camera")
            
        # Add to registry
        await camera_registry.add_camera(camera)
        
        logger.info(f"Camera {name} (ID: {camera_id}) added successfully")
        return jsonify({
            'status': 'success',
            'message': f'Camera {name} added successfully',
            'camera_id': camera_id
        })
        
    except Exception as e:
        logger.error(f"Failed to add camera: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

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
    """List all registered cameras"""
    try:
        cameras = []
        for camera_id, camera in camera_registry._cameras.items():
            # Get camera stats asynchronously
            stats = await camera.get_camera_stats()
            
            # Convert camera object to dict for JSON serialization
            camera_info = {
                'id': camera_id,
                'name': getattr(camera, 'name', 'Unknown'),
                'status': stats['status'],
                'fps': stats['fps'],
                'is_recording': getattr(camera, 'is_recording', False),
                'resolution': f"{camera.frame_shape[1]}x{camera.frame_shape[0]}" if hasattr(camera, 'frame_shape') else 'unknown',
                'performance': stats.get('performance', {})
            }
            cameras.append(camera_info)
            
        return jsonify({
            'status': 'success',
                'cameras': cameras
            })
    except Exception as e:
        logger.error(f"Error listing cameras: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/')
async def index():
    return await render_template('index.html')

@app.route('/start_recording', methods=['POST'])
@handle_camera_errors
async def start_recording():
    """Start recording for a camera"""
    try:
        data = await request.get_json()
        camera_id = data.get('camera_id')
            
        camera = camera_registry.get_camera(camera_id)
        if not camera:
            raise CameraError(f'Camera {camera_id} not found', 404)
        
        logger.info(f"Queued recording for camera {camera_id}")
        success, message = await camera.start_recording()
            
        if success:
                return jsonify({
                    'status': 'success',
                            'message': message,
                            'recording_file': camera.current_recording
                        })
        else:
                        return jsonify({
                            'status': 'error',
                            'message': message
                        }), 400
            
    except Exception as e:
        logger.error(f"Error starting recording: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/recording_status/<int:camera_id>', methods=['GET'])
@handle_camera_errors
async def get_recording_status(camera_id):
    """Get current recording status"""
    camera = camera_registry.get_camera(camera_id)
    if not camera:
        return jsonify({'error': 'Camera not found'})
        
    return jsonify({
        'is_recording': camera.is_recording,
        'duration': time.time() - camera.recording_start_time if camera.recording_start_time else 0,
        'status': camera.recording_status.get('message', ''),
        'error': camera.recording_status.get('error')
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