import librosa
import soundfile as sf

class StepSyncAudioEngine:
    def __init__(self, uploaded_file_path, base_track_bpm):
        self.base_bpm = base_track_bpm
        # Load audio file into a floating-point time series
        self.y, self.sr = librosa.load(uploaded_file_path, sr=None)
        
    def calculate_speed_ratio(self, current_spm):
        """Computes the scaling multiplier bounded within a safe listening range."""
        ratio = current_spm / self.base_bpm
        # Clamp speed between 0.75x and 1.40x to prevent artifacts
        return float(np.clip(ratio, 0.75, 1.40))

    def stretch_audio_segment(self, speed_ratio, output_path="temp_stream.wav"):
        """
        Applies a Phase Vocoder time-stretch algorithm.
        Changing tempo via time_stretch preserves original pitch.
        """
        y_stretched = librosa.effects.time_stretch(self.y, rate=speed_ratio)
        sf.write(output_path, y_stretched, self.sr)
        return output_path