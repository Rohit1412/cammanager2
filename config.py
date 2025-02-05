import os
from dataclasses import dataclass, field
from typing import Dict, Any
import yaml
from utils import check_gpu_available

@dataclass
class CameraConfig:
    max_fps: int = 30
    frame_width: int = 640
    frame_height: int = 480
    retry_attempts: int = 3
    retry_delay: float = 1.0
    connection_timeout: float = 5.0

@dataclass
class RecordingConfig:
    output_dir: str = 'recordings'
    max_duration: int = 5 # 5 seconds
    default_codec: str = 'mp4v'
    quality_presets: Dict[str, Dict[str, Any]] = None
    max_storage_gb: float = 50.0
    hardware_acceleration: str = "auto"  # auto/cuda/vaapi/cpu
    storage_path: str = 'recordings'
    retention_days: int = 7
    temp_dir: str = 'temp'
    auto_cleanup: bool = True
    cleanup_interval: int = 3600  # 1 hour
    codec: str = 'libx264'
    bitrate: str = '4M'

    def __post_init__(self):
        if self.quality_presets is None:
            self.quality_presets = {
                'high': {'width': 1920, 'height': 1080, 'fps': 30},
                'medium': {'width': 1280, 'height': 720, 'fps': 25},
                'low': {'width': 640, 'height': 480, 'fps': 20}
            }
        if self.hardware_acceleration == "auto":
            self.hardware_acceleration = "cuda" if check_gpu_available() else "cpu"

@dataclass
class ServerConfig:
    host: str = '0.0.0.0'
    port: int = 5000
    debug: bool = False
    log_level: str = 'INFO'
    max_cameras: int = 10

@dataclass
class ErrorEmailsConfig:
    enabled: bool = False
    smtp_server: str = "smtp.example.com"
    smtp_port: int = 587
    sender: str = "noreply@example.com"
    recipients: list = field(default_factory=lambda: ["admin@example.com"])
    username: str = ""
    password: str = ""

@dataclass
class Config:
    camera: CameraConfig
    recording: RecordingConfig
    server: ServerConfig
    error_emails: ErrorEmailsConfig

    @classmethod
    def load_from_file(cls, config_path: str = 'config.yml') -> 'Config':
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            config_dict = {}

        # Load configuration with defaults
        camera_config = CameraConfig(**config_dict.get('camera', {}))
        recording_config = RecordingConfig(**config_dict.get('recording', {}))
        server_config = ServerConfig(**config_dict.get('server', {}))
        error_emails_config = ErrorEmailsConfig(**config_dict.get('error_emails', {}))

        return cls(
            camera=camera_config,
            recording=recording_config,
            server=server_config,
            error_emails=error_emails_config
        )

    def save_to_file(self, config_path: str = 'config.yml'):
        config_dict = {
            'camera': {
                'max_fps': self.camera.max_fps,
                'frame_width': self.camera.frame_width,
                'frame_height': self.camera.frame_height,
                'retry_attempts': self.camera.retry_attempts,
                'retry_delay': self.camera.retry_delay,
                'connection_timeout': self.camera.connection_timeout
            },
            'recording': {
                'output_dir': self.recording.output_dir,
                'max_duration': self.recording.max_duration,
                'default_codec': self.recording.default_codec,
                'quality_presets': self.recording.quality_presets,
                'max_storage_gb': self.recording.max_storage_gb,
                'hardware_acceleration': self.recording.hardware_acceleration,
                'storage_path': self.recording.storage_path,
                'retention_days': self.recording.retention_days,
                'temp_dir': self.recording.temp_dir,
                'auto_cleanup': self.recording.auto_cleanup,
                'cleanup_interval': self.recording.cleanup_interval,
                'codec': self.recording.codec,
                'bitrate': self.recording.bitrate,
                'fps': self.recording.fps
            },
            'server': {
                'host': self.server.host,
                'port': self.server.port,
                'debug': self.server.debug,
                'log_level': self.server.log_level,
                'max_cameras': self.server.max_cameras
            },
            'error_emails': {
                'enabled': self.error_emails.enabled,
                'smtp_server': self.error_emails.smtp_server,
                'smtp_port': self.error_emails.smtp_port,
                'sender': self.error_emails.sender,
                'recipients': self.error_emails.recipients,
                'username': self.error_emails.username,
                'password': self.error_emails.password
            }
        }

        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False) 