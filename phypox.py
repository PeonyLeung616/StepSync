import requests
import numpy as np

class PhyphoxClient:
    def __init__(self, base_url, frequency_unit="hz"):
        if frequency_unit not in {"per_minute", "hz"}:
            raise ValueError("frequency_unit must be 'per_minute' or 'hz'")
        self.base_url = base_url.rstrip("/")
        self.frequency_unit = frequency_unit
        self.spm_buffer = []
        self.prev_smoothed_spm = 120.0  # Default fallback baseline

    def fetch_current_spm(self):
        """Polls the Acceleration Spectrum peak-frequency buffer and returns SPM."""
        try:
            response = requests.get(f"{self.base_url}/get?f0", timeout=1.0)
            if response.status_code == 200:
                data = response.json()
                values = data.get("buffer", {}).get("f0", {}).get("buffer", [])
                if values:
                    raw_frequency = values[-1]
                    return raw_frequency * 60 if self.frequency_unit == "hz" else raw_frequency
        except Exception as e:
            print(f"Connection warning: {e}")
        return None

    def get_smoothed_spm(self, raw_spm, alpha=0.2):
        """Applies Exponential Moving Average (EMA) to prevent sudden jumps."""
        if raw_spm is None:
            return self.prev_smoothed_spm
        
        smoothed = alpha * raw_spm + (1 - alpha) * self.prev_smoothed_spm
        self.prev_smoothed_spm = smoothed
        return smoothed