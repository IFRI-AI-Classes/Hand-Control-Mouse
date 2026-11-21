"""Détecteur factice pour tester sans MediaPipe."""
from extraction.data_models import HandObservation


class FakeExtractor:
    def __init__(self, missing_every: int = 0, side: str = "Right"):
        self.missing_every = missing_every
        self.side = side
        self.timestamps = []
        self.closed = False

    def extract(self, rgb_frame, timestamp_ms):
        self.timestamps.append(timestamp_ms)
        if self.missing_every and len(self.timestamps) % self.missing_every == 0:
            return None
        points = tuple((0.1 * (i % 10), 0.05 * (i % 20), 0.0) for i in range(21))
        return HandObservation(0.9, self.side, points)

    def close(self):
        self.closed = True


def factory(**kwargs):
    return lambda: FakeExtractor(**kwargs)
