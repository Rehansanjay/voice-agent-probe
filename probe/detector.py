"""Find where the other side of a phone call is speaking, from 8 kHz audio as it arrives.

Every incoming chunk is stamped with the time it arrived. Silero scores 32 ms
windows (256 samples at 8 kHz); a run of speech windows marks an onset, and a
long enough run of silence marks the end of that speech. Times are arrival
times on this machine, so a gap measured here is what a caller would hear.
"""

from dataclasses import dataclass, field

import numpy as np

from pipecat.audio.vad.silero import SileroVADAnalyzer

RATE = 8000
WINDOW = 256  # samples Silero scores at 8 kHz
WINDOW_S = WINDOW / RATE  # 0.032 s


@dataclass
class Span:
    start: float
    end: float | None = None


@dataclass
class SpeechDetector:
    """Turns a stream of 16-bit PCM chunks into speech spans with timestamps.

    threshold: Silero confidence above which a window counts as speech.
    onset_windows: consecutive speech windows needed to start a span (~96 ms).
    end_silence_s: silence needed to end a span. Long enough to ride over the
        pauses inside one reply, short enough to close it before our next line.
    """

    threshold: float = 0.5
    onset_windows: int = 3
    end_silence_s: float = 0.8
    spans: list[Span] = field(default_factory=list)

    def __post_init__(self):
        self._vad = SileroVADAnalyzer(sample_rate=RATE)
        # The constructor only records the rate; scoring needs it applied,
        # which a Pipecat pipeline would otherwise do at start.
        self._vad.set_sample_rate(RATE)
        self._buf = b""
        self._buf_t = 0.0  # arrival time of the first sample in _buf
        self._run = 0  # consecutive speech windows
        self._run_start = 0.0
        self._last_speech_end = 0.0
        self.confidences: list[tuple[float, float]] = []

    @property
    def speaking(self) -> bool:
        return bool(self.spans) and self.spans[-1].end is None

    def feed(self, pcm: bytes, arrived_at: float) -> None:
        """Add a chunk that arrived at ``arrived_at`` (seconds, monotonic)."""
        self._buf += pcm
        # Anchor to the latest arrival: the newest sample is "now", and the
        # buffer start is that many samples earlier. Counting samples alone
        # drifts whenever the stream arrives unevenly.
        self._buf_t = arrived_at - len(self._buf) / 2 / RATE
        while len(self._buf) >= WINDOW * 2:
            window, self._buf = self._buf[: WINDOW * 2], self._buf[WINDOW * 2 :]
            t0 = self._buf_t
            self._buf_t += WINDOW_S
            self._score(window, t0)

    def _score(self, window: bytes, t0: float) -> None:
        conf = float(np.asarray(self._vad.voice_confidence(window)).reshape(-1)[0])
        self.confidences.append((t0, conf))
        t1 = t0 + WINDOW_S
        if conf >= self.threshold:
            if self._run == 0:
                self._run_start = t0
            self._run += 1
            self._last_speech_end = t1
            if not self.speaking and self._run >= self.onset_windows:
                self.spans.append(Span(start=self._run_start))
        else:
            self._run = 0
            if self.speaking and t1 - self._last_speech_end >= self.end_silence_s:
                self.spans[-1].end = self._last_speech_end

    def close(self, at: float) -> None:
        """End an open span at the last speech seen, e.g. when the call drops."""
        if self.speaking:
            self.spans[-1].end = self._last_speech_end or at

    def first_onset_after(self, t: float) -> float | None:
        for s in self.spans:
            if s.start >= t:
                return s.start
        return None

    def speech_end_after(self, t: float) -> float | None:
        """When the speech that was going on at (or started after) ``t`` ended."""
        for s in self.spans:
            if s.end is not None and s.end >= t:
                return s.end
        return None


def pcm_rms(pcm: bytes) -> float:
    a = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
    return float(np.sqrt(np.mean(a * a))) if a.size else 0.0
