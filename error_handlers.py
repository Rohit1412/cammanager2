from quart import jsonify
from functools import wraps
import logging
import smtplib
from email.message import EmailMessage
from config import Config

logger = logging.getLogger('camera_manager')
config = Config.load_from_file()

class CameraError(Exception):
    """Base exception class for camera-related errors"""
    def __init__(self, message, status_code=500):
        super().__init__(message)
        self.status_code = status_code

class CameraInitError(CameraError):
    """Raised when camera initialization fails"""
    pass

class StreamError(CameraError):
    """Raised when streaming operations fail"""
    pass

class RecordingError(CameraError):
    """Raised when recording operations fail"""
    pass

def handle_camera_errors(f):
    """Decorator to handle camera-related errors"""
    @wraps(f)
    async def decorated_function(*args, **kwargs):
        try:
            return await f(*args, **kwargs)
        except CameraError as e:
            logger.error(f"Camera operation failed: {str(e)}", exc_info=True)
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), e.status_code
        except Exception as e:
            logger.critical(f"Unexpected error: {str(e)}", exc_info=True)
            return jsonify({
                'status': 'error',
                'message': 'An unexpected error occurred'
            }), 500
    return decorated_function

def send_error_email(subject, body):
    """Send error notification email"""
    try:
        if not config.error_emails.enabled:
            return

        msg = EmailMessage()
        msg.set_content(body)
        msg['Subject'] = f"[Camera System] {subject}"
        msg['From'] = config.error_emails.sender
        msg['To'] = config.error_emails.recipients

        with smtplib.SMTP(config.error_emails.smtp_server,
                         config.error_emails.smtp_port) as server:
            server.starttls()
            server.login(config.error_emails.username,
                        config.error_emails.password)
            server.send_message(msg)
            
    except Exception as e:
        logger.error(f"Failed to send error email: {str(e)}")

def register_error_handlers(app):
    """Register error handlers with the app"""
    @app.errorhandler(400)
    async def bad_request_error(e):
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

    @app.errorhandler(404)
    async def not_found_error(e):
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 404

    @app.errorhandler(503)
    async def service_unavailable_error(e):
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 503 