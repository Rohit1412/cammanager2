import subprocess
import time
import os
import sys
from threading import Thread
import socket

def get_local_ip():
    """Get the local IP address of the machine"""
    try:
        # Create a socket to get local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "localhost"

def create_color_video(output_file, duration=100, size="640x480", color="red"):
    """Create a color video file using ffmpeg"""
    cmd = [
        'ffmpeg', '-y',
        '-f', 'lavfi',
        '-i', f'color=c={color}:s={size}:d={duration}',
        '-c:v', 'libx264',
        '-tune', 'stillimage',
        '-pix_fmt', 'yuv420p',
        output_file
    ]
    subprocess.run(cmd, check=True)

def start_rtsp_stream(input_file, rtsp_url):
    """Start an RTSP stream using ffmpeg"""
    cmd = [
        'ffmpeg', '-re',
        '-stream_loop', '-1',  # Loop indefinitely
        '-i', input_file,
        '-c', 'copy',
        '-f', 'rtsp',
        '-rtsp_transport', 'tcp',
        rtsp_url
    ]
    process = subprocess.Popen(cmd)
    return process

def start_rtmp_stream(input_file, rtmp_url):
    """Start an RTMP stream using ffmpeg"""
    cmd = [
        'ffmpeg', '-re',
        '-stream_loop', '-1',  # Loop indefinitely
        '-i', input_file,
        '-c', 'copy',
        '-f', 'flv',
        rtmp_url
    ]
    process = subprocess.Popen(cmd)
    return process

def main():
    local_ip = get_local_ip()
    print(f"Local IP address: {local_ip}")
    
    # Create temporary video files
    print("Creating temporary video files...")
    os.makedirs("temp", exist_ok=True)
    
    # Define video files
    videos = {
        'red': "temp/red.mp4",
        'blue': "temp/blue.mp4",
        'pink': "temp/pink.mp4",
        'yellow': "temp/yellow.mp4"
    }
    
    # Create color videos
    for color, filepath in videos.items():
        print(f"Creating {color} video...")
        create_color_video(filepath, color=color)
    
    # Stream settings
    rtsp_streams = {
        'red': {'port': 8554, 'path': '/red'},
        'yellow': {'port': 8556, 'path': '/yellow'}
    }
    
    rtmp_streams = {
        'blue': {'port': 1935, 'path': '/live/blue'},
        'pink': {'port': 1935, 'path': '/live/pink'}
    }
    
    # Start RTMP server (using nginx-rtmp)
    nginx_conf = f"""
worker_processes  1;
events {{
    worker_connections  1024;
}}
rtmp {{
    server {{
        listen {rtmp_streams['blue']['port']};
        chunk_size 4096;
        application live {{
            live on;
            allow publish all;
            allow play all;
        }}
    }}
}}
"""
    
    with open("temp/nginx.conf", "w") as f:
        f.write(nginx_conf)
    
    # Kill any existing nginx processes
    subprocess.run(['pkill', 'nginx'], stderr=subprocess.DEVNULL)
    
    # Start nginx-rtmp server
    nginx_process = subprocess.Popen(
        ['nginx', '-c', os.path.abspath('temp/nginx.conf')],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Kill any existing rtsp-simple-server processes
    subprocess.run(['pkill', 'rtsp-simple-server'], stderr=subprocess.DEVNULL)
    
    # Start RTSP server
    rtsp_server_process = subprocess.Popen(
        ['rtsp-simple-server'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    time.sleep(2)  # Wait for servers to start
    
    # Start streams
    print("\nStarting streams...")
    processes = []
    
    # Start RTSP streams
    for color, config in rtsp_streams.items():
        url = f"rtsp://{local_ip}:{config['port']}{config['path']}"
        process = start_rtsp_stream(videos[color], url)
        processes.append(process)
        print(f"RTSP stream ({color}): {url}")
    
    # Start RTMP streams
    for color, config in rtmp_streams.items():
        url = f"rtmp://{local_ip}:{config['port']}{config['path']}"
        process = start_rtmp_stream(videos[color], url)
        processes.append(process)
        print(f"RTMP stream ({color}): {url}")
    
    print("\nPress Ctrl+C to stop the streams")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping streams...")
        # Stop all streaming processes
        for process in processes:
            process.terminate()
        
        # Stop servers
        nginx_process.terminate()
        rtsp_server_process.terminate()
        
        # Clean up temporary files
        for filepath in videos.values():
            if os.path.exists(filepath):
                os.remove(filepath)
        
        if os.path.exists("temp/nginx.conf"):
            os.remove("temp/nginx.conf")
            
        try:
            os.rmdir("temp")
        except:
            pass

if __name__ == "__main__":
    main() 