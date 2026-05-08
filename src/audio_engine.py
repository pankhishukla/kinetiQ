import threading
import queue
import pyttsx3

class AudioEngine:
    """
    Background worker thread for Text-to-Speech (TTS).
    WHY a background thread?
        pyttsx3.say() blocks execution until the voice finishes speaking (1-3 seconds).
        If called in the main loop, it would freeze the webcam feed.
    """
    def __init__(self):
        self.q = queue.Queue()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        # pyttsx3 initialization MUST happen in the thread where it is used.
        try:
            engine = pyttsx3.init()
            # Set speech rate slightly faster for crisp real-time feedback
            rate = engine.getProperty('rate')
            engine.setProperty('rate', rate + 25)
        except Exception as e:
            print(f"[AudioEngine] Failed to initialize TTS: {e}")
            return
            
        while not self.stop_event.is_set():
            try:
                # Block until text is received (timeout allows checking stop_event)
                text = self.q.get(timeout=0.5)
                if text is None:  # Sentinel value to exit
                    break
                
                engine.say(text)
                engine.runAndWait()
                self.q.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[AudioEngine Error] {e}")

    def speak(self, text: str):
        """
        Queue text to be spoken. Clears any pending backlog so that the 
        audio always represents the LATEST state.
        """
        # Clear backlog
        while not self.q.empty():
            try:
                self.q.get_nowait()
                self.q.task_done()
            except queue.Empty:
                break
                
        self.q.put(text)

    def stop(self):
        """Gracefully shut down the background thread."""
        self.stop_event.set()
        self.q.put(None)
        self.thread.join(timeout=2.0)
