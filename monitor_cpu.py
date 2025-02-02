import psutil
import time
import json
from datetime import datetime
import os

def monitor_cpu_usage(duration=60, interval=1):
    """
    Monitor CPU usage for a specified duration
    duration: monitoring period in seconds
    interval: sampling interval in seconds
    """
    stats = {
        'start_time': datetime.now().isoformat(),
        'samples': [],
        'summary': {}
    }
    
    try:
        print(f"Starting CPU monitoring for {duration} seconds...")
        end_time = time.time() + duration
        
        while time.time() < end_time:
            # Get CPU usage per core
            cpu_percent = psutil.cpu_percent(interval=interval, percpu=True)
            
            # Get process-specific CPU usage
            python_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
                try:
                    if 'python' in proc.info['name'].lower():
                        proc_cpu = proc.cpu_percent(interval=0.1)
                        python_processes.append({
                            'pid': proc.info['pid'],
                            'cpu_percent': proc_cpu
                        })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            # Get memory usage
            memory = psutil.virtual_memory()
            
            sample = {
                'timestamp': datetime.now().isoformat(),
                'cpu_percent_per_core': cpu_percent,
                'cpu_average': sum(cpu_percent) / len(cpu_percent),
                'python_processes': python_processes,
                'memory_percent': memory.percent,
                'camera_processes': [p for p in python_processes 
                                   if any(x in p.get('cmdline', []) 
                                   for x in ['ffmpeg', 'opencv', 'vidgear'])]
            }
            
            stats['samples'].append(sample)
            
            # Print real-time stats
            print(f"\rCPU Usage: {sample['cpu_average']:.1f}% | Memory: {memory.percent}% | "
                  f"Camera Processes: {len(sample['camera_processes'])}", end='')
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    
    # Calculate summary statistics
    if stats['samples']:
        cpu_averages = [s['cpu_average'] for s in stats['samples']]
        memory_averages = [s['memory_percent'] for s in stats['samples']]
        
        stats['summary'] = {
            'cpu_min': min(cpu_averages),
            'cpu_max': max(cpu_averages),
            'cpu_avg': sum(cpu_averages) / len(cpu_averages),
            'memory_min': min(memory_averages),
            'memory_max': max(memory_averages),
            'memory_avg': sum(memory_averages) / len(memory_averages)
        }
        
        # Save results
        filename = f"cpu_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print("\n\nSummary:")
        print(f"Average CPU Usage: {stats['summary']['cpu_avg']:.1f}%")
        print(f"Peak CPU Usage: {stats['summary']['cpu_max']:.1f}%")
        print(f"Average Memory Usage: {stats['summary']['memory_avg']:.1f}%")
        print(f"Results saved to {filename}")

if __name__ == "__main__":
    # Monitor for 5 minutes by default
    monitor_cpu_usage(duration=300, interval=1) 