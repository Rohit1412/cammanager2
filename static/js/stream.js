const ws = new WebSocket(`ws://${location.host}/stream/${cameraId}`);
const img = document.getElementById('stream-frame');

ws.onmessage = function(event) {
    if(event.data.size > 0) {
        const blob = new Blob([event.data], {type: 'image/jpeg'});
        img.src = URL.createObjectURL(blob);
    } else {
        img.src = '/static/images/placeholder.jpg';
    }
};

ws.onerror = function(error) {
    img.src = '/static/images/error.jpg';
    console.error('WebSocket Error:', error);
};

async function updateRecordingStatus(cameraId) {
    try {
        const response = await fetch(`/recording_status/${cameraId}`);
        const data = await response.json();
        
        // Handle API errors
        if (data.status === 'error') {
            throw new Error(data.message);
        }
        
        // Safe status check with defaults
        const isRecording = data.active_recording || false;
        const recordingFile = data.recording_file || 'unknown';
        
        const statusElement = document.getElementById(`recording-status-${cameraId}`);
        if (statusElement) {
            statusElement.textContent = isRecording ? 
                `Recording: ${recordingFile}` : 
                'Not Recording';
            statusElement.className = isRecording ? 
                'recording-active' : 'recording-inactive';
        }
    } catch (error) {
        console.error('Error updating recording status:', error);
        const statusElement = document.getElementById(`recording-status-${cameraId}`);
        if (statusElement) {
            statusElement.textContent = 'Status Error';
            statusElement.className = 'recording-error';
        }
    }
} 