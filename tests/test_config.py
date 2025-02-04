import pytest
import os
from config import Config, CameraConfig, RecordingConfig, ServerConfig

@pytest.fixture
def config():
    return Config(
        camera=CameraConfig(),
        recording=RecordingConfig(),
        server=ServerConfig(),
        error_emails=EmailConfig()
    )

class EmailConfig:
    def __init__(self):
        self.enabled = False
        self.username = ""
        self.password = ""
        self.recipients = []
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        self.use_tls = True
        self.sender = ""

def test_default_config_values(config):
    assert config.camera.max_fps == 20
    assert config.camera.frame_width == 640
    assert config.camera.frame_height == 480
    assert config.recording.output_dir == 'recordings'
    assert config.server.host == '0.0.0.0'
    assert config.server.port == 5000

def test_config_save_load(config, tmp_path):
    config_path = tmp_path / "test_config.yml"
    config.save_to_file(str(config_path))
    assert os.path.exists(config_path)
    
    loaded_config = Config.load_from_file(str(config_path))
    assert loaded_config.camera.max_fps == config.camera.max_fps
    assert loaded_config.recording.output_dir == config.recording.output_dir
    assert loaded_config.server.port == config.server.port

def test_quality_presets(config):
    presets = config.recording.quality_presets
    assert 'high' in presets
    assert 'medium' in presets
    assert 'low' in presets
    assert presets['high']['width'] == 1920
    assert presets['medium']['width'] == 1280
    assert presets['low']['width'] == 640 