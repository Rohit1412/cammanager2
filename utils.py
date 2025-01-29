# Shared utilities used across multiple modules
import cv2

def check_gpu_available():
    """Check if GPU acceleration is available"""
    try:
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            return True
        if cv2.ocl.haveOpenCL():
            return True
        return False
    except:
        return False

GPU_AVAILABLE = check_gpu_available() 