import time
from collections import deque
import logging
from datetime import datetime, timedelta

logger = logging.getLogger('camera_manager')

class ErrorMonitor:
    def __init__(self, error_threshold=10, time_window=300):  # 5 minutes window
        self.error_threshold = error_threshold
        self.time_window = time_window
        self.error_logs = deque(maxlen=1000)  # Store up to 1000 errors
        self.alert_handlers = []

    def log_error(self, error_type, message, camera_id=None):
        timestamp = time.time()
        error_entry = {
            'timestamp': timestamp,
            'type': error_type,
            'message': message,
            'camera_id': camera_id
        }
        self.error_logs.append(error_entry)
        
        # Check for error threshold
        self._check_error_threshold()

    def add_alert_handler(self, handler):
        """Add a handler function to be called when error threshold is exceeded"""
        self.alert_handlers.append(handler)

    def _check_error_threshold(self):
        """Check if error threshold has been exceeded in the time window"""
        current_time = time.time()
        window_start = current_time - self.time_window
        
        # Count errors in the time window
        recent_errors = sum(1 for error in self.error_logs 
                          if error['timestamp'] > window_start)
        
        if recent_errors >= self.error_threshold:
            self._trigger_alert(recent_errors)

    def _trigger_alert(self, error_count):
        """Trigger alert handlers when threshold is exceeded"""
        alert_message = f"Error threshold exceeded: {error_count} errors in the last {self.time_window} seconds"
        logger.warning(alert_message)
        
        for handler in self.alert_handlers:
            try:
                handler(alert_message, self.error_logs)
            except Exception as e:
                logger.error(f"Error in alert handler: {str(e)}")

    def get_error_summary(self):
        """Get a summary of recent errors"""
        if not self.error_logs:
            return "No errors recorded"
            
        current_time = time.time()
        window_start = current_time - self.time_window
        recent_errors = [e for e in self.error_logs if e['timestamp'] > window_start]
        
        return {
            'total_errors': len(recent_errors),
            'time_window': self.time_window,
            'errors_by_type': self._group_errors_by_type(recent_errors)
        }

    def _group_errors_by_type(self, errors):
        error_groups = {}
        for error in errors:
            error_type = error['type']
            if error_type not in error_groups:
                error_groups[error_type] = 0
            error_groups[error_type] += 1
        return error_groups 