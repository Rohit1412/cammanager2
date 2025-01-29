import subprocess
import time
import os
import signal
import sys
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StreamGenerator:
    def __init__(self):
        self.processes: List[subprocess.Popen] = []
        
        # Test pattern video file
        self.test_input = "testsrc=size=640x480:rate=30"
        
        # Stream configurations
        self.rtsp_streams = {
            "stream1": "rtsp://localhost:8554/stream1",
            "stream2": "rtsp://localhost:8554/stream2"
        }
        
        self.rtmp_streams = {
            "stream1": "rtmp://localhost:1935/live/stream1",
            "stream2": "rtmp://localhost:1935/live/stream2"
        }

    def start_rtsp_server(self):
        """Start RTSP simple server"""
        try:
            cmd = ["rtsp-simple-server"]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.processes.append(process)
            logger.info("Started RTSP server")
            time.sleep(2)  # Give server time to start
        except FileNotFoundError:
            logger.error("rtsp-simple-server not found. Please install it first.")
            sys.exit(1)

    def create_rtsp_streams(self):
        """Create RTSP test streams"""
        for name, url in self.rtsp_streams.items():
            cmd = [
                "ffmpeg",
                "-re",  # Read input at native frame rate
                "-f", "lavfi",  # Use lavfi input
                "-i", self.test_input,  # Test input pattern
                "-c:v", "libx264",  # Video codec
                "-preset", "ultrafast",
                "-tune", "zerolatency",
                "-f", "rtsp",  # Output format
                url
            ]
            
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.processes.append(process)
            logger.info(f"Started RTSP stream: {url}")
            time.sleep(1)

    def create_rtmp_streams(self):
        """Create RTMP test streams"""
        for name, url in self.rtmp_streams.items():
            cmd = [
                "ffmpeg",
                "-re",
                "-f", "lavfi",
                "-i", self.test_input,
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-tune", "zerolatency",
                "-f", "flv",  # RTMP uses FLV format
                url
            ]
            
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.processes.append(process)
            logger.info(f"Started RTMP stream: {url}")
            time.sleep(1)

    def cleanup(self, signum=None, frame=None):
        """Cleanup all processes"""
        logger.info("Cleaning up streams...")
        for process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            except Exception as e:
                logger.error(f"Error cleaning up process: {e}")
        
        logger.info("All streams stopped")
        if signum is not None:
            sys.exit(0)

    def get_stream_urls(self) -> Dict[str, List[str]]:
        """Return all stream URLs"""
        return {
            "rtsp": list(self.rtsp_streams.values()),
            "rtmp": list(self.rtmp_streams.values())
        }

def main():
    generator = StreamGenerator()
    
    # Register signal handlers for cleanup
    signal.signal(signal.SIGINT, generator.cleanup)
    signal.signal(signal.SIGTERM, generator.cleanup)
    
    try:
        # Start RTSP server
        generator.start_rtsp_server()
        
        # Create streams
        generator.create_rtsp_streams()
        generator.create_rtmp_streams()
        
        # Print stream URLs
        streams = generator.get_stream_urls()
        print("\nAvailable test streams:")
        print("RTSP Streams:")
        for url in streams["rtsp"]:
            print(f"  - {url}")
        print("\nRTMP Streams:")
        for url in streams["rtmp"]:
            print(f"  - {url}")
        
        print("\nPress Ctrl+C to stop all streams")
        
        # Keep the script running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        generator.cleanup()

if __name__ == "__main__":
    main() 