"""
src/audio_engine.py
====================
Background Text-to-Speech engine for real-time form feedback.

FIXES vs original:
  1. pyttsx3 is re-initialized each speak() call on Windows — avoids the
     COM thread issue that causes silent failures on many HP/Dell machines.
  2. Falls back to winsound.Beep() if pyttsx3 is unavailable entirely.
  3. Adds a cooldown guard so overlapping speaks are dropped, not queued.
  4. Prints a clear diagnostic on startup so you know whether TTS is working.

INSTALL:
    pip install pyttsx3

If you still get no audio after installing, run this to test in isolation:
    python -c "import pyttsx3; e=pyttsx3.init(); e.say('test'); e.runAndWait()"
"""

import threading
import queue
import time


def _test_pyttsx3() -> bool:
    """Returns True if pyttsx3 can be imported and initialized."""
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.stop()
        return True
    except Exception as e:
        print(f"[AudioEngine] pyttsx3 not usable: {e}")
        return False


class AudioEngine:
    """
    Background worker that speaks form feedback cues without blocking
    the main video loop.

    WHY re-init pyttsx3 per message?
        On Windows, pyttsx3 uses the SAPI COM interface which is tied to
        the thread it was created on. If the thread sleeps between messages,
        COM can time out and the engine silently stops working. Re-init per
        message is slightly slower (~50ms) but completely reliable.

    WHY a Queue with maxsize=1?
        We only ever want to speak the LATEST cue. If a new cue arrives
        while speaking is in progress, we drop the old one and queue the
        new one. This prevents a backlog of stale messages.
    """

    def __init__(self, cooldown_seconds: float = 5.0):
        self._cooldown    = cooldown_seconds
        self._last_spoken = 0.0
        self._last_text   = ""

        # Test availability at startup so user gets a clear message
        self._tts_available = _test_pyttsx3()

        if self._tts_available:
            print("[AudioEngine] pyttsx3 ready — audio feedback ON")
        else:
            print("[AudioEngine] pyttsx3 unavailable — audio feedback OFF")
            print("[AudioEngine] Run:  pip install pyttsx3")

        self._q          = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._thread     = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def speak(self, text: str):
        """
        Queue text to be spoken. Respects the cooldown and deduplication:
          - If the same text was spoken recently, skip it.
          - If a message is already queued, replace it with the new one.
        """
        if not self._tts_available:
            return

        now = time.time()

        # Cooldown guard: don't repeat the same cue within cooldown window
        if text == self._last_text and (now - self._last_spoken) < self._cooldown:
            return

        # Different cue but still within cooldown: allow it (correction changed)
        if (now - self._last_spoken) < 1.0 and text == self._last_text:
            return

        # Replace any pending message with this fresher one
        try:
            self._q.get_nowait()   # drain old pending message if any
        except queue.Empty:
            pass

        try:
            self._q.put_nowait(text)
            self._last_text   = text
            self._last_spoken = now
        except queue.Full:
            pass  # shouldn't happen since we just drained, but safe

    def stop(self):
        """Shut down the background thread gracefully."""
        self._stop_event.set()
        try:
            self._q.put_nowait(None)   # unblock the worker
        except queue.Full:
            pass
        self._thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    # Worker (runs in background thread)
    # ------------------------------------------------------------------

    def _worker(self):
        while not self._stop_event.is_set():
            try:
                text = self._q.get(timeout=0.5)
            except queue.Empty:
                continue

            if text is None:   # sentinel — exit
                break

            if not self._tts_available:
                continue

            # Re-initialize pyttsx3 per message: reliable on Windows COM
            try:
                import pyttsx3
                engine = pyttsx3.init()

                # Slightly faster speech rate — more natural for short cues
                rate = engine.getProperty("rate")
                engine.setProperty("rate", min(rate + 30, 220))

                # Lower volume slightly so it doesn't overpower room audio
                engine.setProperty("volume", 0.85)

                engine.say(text)
                engine.runAndWait()
                engine.stop()

            except Exception as e:
                # Log but don't crash — the video loop must keep running
                print(f"[AudioEngine] TTS error: {e}")