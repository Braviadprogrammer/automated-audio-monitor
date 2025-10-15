import flask
import sounddevice as sd
import numpy as np
import webrtcvad
import collections
import queue
import threading
import time
import logging
import sys
from datetime import datetime

# --- Configuration ---
SAMPLE_RATE = 16000       # Sample rate in Hz (VAD supports 8k, 16k, 32k)
BLOCK_DURATION_MS = 30    # Duration of audio chunks for VAD (10, 20, or 30 ms)
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DURATION_MS / 1000) # Samples per chunk
CHANNELS = 1              # Mono audio
DTYPE = 'int16'           # Data type for audio samples (VAD requires 16-bit PCM)
VAD_AGGRESSIVENESS = 1    # VAD aggressiveness (0=least, 3=most aggressive)

# Thresholds (adjust based on your microphone and environment)
RMS_SILENCE_THRESHOLD = 50   # RMS value below which is considered near silence
RMS_NOISE_THRESHOLD = 500    # RMS value above which is considered significant noise
VOICE_CONFIDENCE_FRAMES = 3  # How many consecutive VAD=True frames needed to confirm voice
SILENCE_HANGOVER_FRAMES = 10 # How many consecutive VAD=False frames needed to confirm silence

LOG_FILE = 'audio_monitor.log'

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout) # Also print logs to console
    ]
)

# --- Flask App Setup ---
app = flask.Flask(__name__)
app.secret_key = 'super secret key' # Change this for production

# --- Shared State (Thread-Safe) ---
# Queue to send updates to the web interface (Server-Sent Events)
sse_queue = queue.Queue()

# Current status dictionary
current_status = {
    "timestamp": datetime.now().isoformat(),
    "voice_detected": False,
    "noise_detected": False,
    "sound_detected": False,
    "conducive": True,
    "message": "Monitoring started..."
}
status_lock = threading.Lock() # To protect access to current_status

# --- Audio Processing Logic ---
class AudioProcessor:
    def __init__(self, aggressiveness, rms_silence_threshold, rms_noise_threshold):
        self.vad = webrtcvad.Vad(aggressiveness)
        self.rms_silence_threshold = rms_silence_threshold
        self.rms_noise_threshold = rms_noise_threshold
        self.triggered = False
        self.ring_buffer = collections.deque(maxlen=VOICE_CONFIDENCE_FRAMES)
        self.silence_buffer = collections.deque(maxlen=SILENCE_HANGOVER_FRAMES)

    def process_chunk(self, audio_chunk):
        """Analyzes a single audio chunk."""
        is_speech = False
        is_noise = False
        is_sound = False

        # 1. Calculate RMS for noise/sound detection
        rms = np.sqrt(np.mean(audio_chunk.astype(np.float32)**2))

        if rms > self.rms_noise_threshold:
            is_noise = True
            is_sound = True
        elif rms > self.rms_silence_threshold:
            is_sound = True # Any sound above silence threshold

        # 2. Perform VAD
        try:
            # VAD requires bytes
            is_speech_frame = self.vad.is_speech(audio_chunk.tobytes(), SAMPLE_RATE)
        except Exception as e:
            # Handle potential errors like incorrect chunk size
            # logging.warning(f"VAD error: {e}")
            is_speech_frame = False # Assume not speech on error

        # 3. Use ring buffers for stable voice detection
        self.ring_buffer.append(is_speech_frame)
        self.silence_buffer.append(is_speech_frame)

        num_voiced = sum(self.ring_buffer)
        num_unvoiced = len(self.silence_buffer) - sum(self.silence_buffer)

        if not self.triggered and num_voiced >= VOICE_CONFIDENCE_FRAMES:
             self.triggered = True
             is_speech = True
        elif self.triggered and num_unvoiced >= SILENCE_HANGOVER_FRAMES:
             self.triggered = False
             is_speech = False # End of speech segment
        elif self.triggered:
             is_speech = True # Still in speech segment

        return is_speech, is_noise, is_sound, rms

# --- Background Audio Monitoring Thread ---
def audio_monitor_thread():
    """
    Continuously captures and processes audio in a background thread.
    Updates the shared status and pushes messages to the SSE queue.
    """
    processor = AudioProcessor(VAD_AGGRESSIVENESS, RMS_SILENCE_THRESHOLD, RMS_NOISE_THRESHOLD)
    last_log_time = time.time()
    log_interval = 2 # Log status every X seconds even if no change

    logging.info("Audio monitoring thread started.")

    try:
        # Use sounddevice's InputStream for continuous audio capture
        with sd.InputStream(samplerate=SAMPLE_RATE,
                            blocksize=BLOCK_SIZE,
                            channels=CHANNELS,
                            dtype=DTYPE) as stream:
            logging.info(f"Audio stream opened successfully (Samplerate: {SAMPLE_RATE}, Blocksize: {BLOCK_SIZE})")
            while True:
                # Read a block of audio data
                audio_chunk, overflowed = stream.read(BLOCK_SIZE)
                if overflowed:
                    logging.warning("Audio buffer overflowed!")

                # Process the chunk
                is_speech, is_noise, is_sound, rms = processor.process_chunk(audio_chunk)

                # Determine overall status
                now = datetime.now()
                env_conducive = not (is_speech or is_noise or is_sound)
                status_changed = False
                new_message = ""

                with status_lock:
                    if (current_status["voice_detected"] != is_speech or
                        current_status["noise_detected"] != is_noise or
                        current_status["sound_detected"] != is_sound or
                        current_status["conducive"] != env_conducive):
                        status_changed = True

                        current_status["timestamp"] = now.isoformat()
                        current_status["voice_detected"] = is_speech
                        current_status["noise_detected"] = is_noise
                        current_status["sound_detected"] = is_sound
                        current_status["conducive"] = env_conducive

                        if is_speech: new_message = "ALERT: Voice detected!"
                        elif is_noise: new_message = "ALERT: Significant noise detected!"
                        elif is_sound: new_message = "ALERT: Background sound detected!"
                        else: new_message = "Environment clear."

                        current_status["message"] = f"{new_message} (RMS: {rms:.2f})"

                        # Log the change
                        log_level = logging.WARNING if not env_conducive else logging.INFO
                        logging.log(log_level, f"Status Change: Voice={is_speech}, Noise={is_noise}, Sound={is_sound}, Conducive={env_conducive}, RMS={rms:.2f}")

                    # Create a copy to send via SSE
                    status_update = current_status.copy()

                # Push update to SSE queue if status changed
                if status_changed:
                    sse_queue.put(status_update)

                # Optional: Log status periodically even if no change
                current_time = time.time()
                if current_time - last_log_time > log_interval and not status_changed:
                     logging.debug(f"Periodic Status: Voice={is_speech}, Noise={is_noise}, Sound={is_sound}, Conducive={env_conducive}, RMS={rms:.2f}")
                     last_log_time = current_time
                     # Also push periodic updates to SSE if desired
                     # sse_queue.put(status_update)


    except sd.PortAudioError as e:
        logging.error(f"PortAudioError: {e}. Is a microphone connected and configured?")
        # Attempt to inform the frontend about the error
        with status_lock:
            current_status["message"] = f"ERROR: Could not open audio stream. Check microphone. {e}"
            current_status["conducive"] = False # Mark as non-conducive due to error
        sse_queue.put(current_status.copy())
    except Exception as e:
        logging.exception("An unexpected error occurred in the audio thread.")
        with status_lock:
            current_status["message"] = f"FATAL ERROR in audio thread: {e}"
            current_status["conducive"] = False
        sse_queue.put(current_status.copy())
    finally:
        logging.info("Audio monitoring thread finished.")


# --- Flask Routes ---
@app.route('/real_time_background_activities_detection')
def ireal_time_background_activities_detection():
    """Serves the main HTML page."""
    return flask.render_template('real_time_background_activities_detection.html')

@app.route('/stream')
def stream():
    """Server-Sent Events endpoint to push status updates."""
    def event_stream():
        while True:
            # Wait for a new status update from the queue
            try:
                status = sse_queue.get(timeout=60) # Timeout helps prevent deadlocks if queue empty
                # Format as SSE message: data: <json_string>\n\n
                yield f"data: {flask.json.dumps(status)}\n\n"
            except queue.Empty:
                # Send a keep-alive comment if no data for a while
                yield ": keep-alive\n\n"
            except Exception as e:
                logging.error(f"Error in SSE stream: {e}")
                # Optionally break or handle differently
                yield f"event: error\ndata: {flask.json.dumps({'error': str(e)})}\n\n"
                # Maybe break here depending on desired behavior
                # break


    # Response object for SSE
    return flask.Response(event_stream(), mimetype="text/event-stream")

@app.route('/log')
def view_log():
    """Optional: Endpoint to view the full log file content."""
    try:
        with open(LOG_FILE, 'r') as f:
            log_content = f.read()
        # Simple text response, could be formatted better
        return flask.Response(log_content, mimetype='text/plain')
    except FileNotFoundError:
        return "Log file not found yet.", 404
    except Exception as e:
        logging.error(f"Error reading log file: {e}")
        return f"Error reading log file: {e}", 500


# --- Main Execution ---
if __name__ == '__main__':
    # Start the background audio monitoring thread
    monitor = threading.Thread(target=audio_monitor_thread, daemon=True)
    monitor.start()

    # Run the Flask web server
    logging.info("Starting Flask web server on http://127.0.0.1:5000")
    # Use host='0.0.0.0' to make it accessible on your network
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    # Note: debug=True can cause issues with threading sometimes. Keep it False for stability.
    # threaded=True is important for handling SSE and audio thread concurrently.