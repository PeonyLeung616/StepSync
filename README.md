pip install streamlit requests numpy

streamlit run app.py

Upload an MP3, WAV, or M4A track, set its original tempo (defaults to 112 BPM),
and press play in the built-in player. The large speed readout shows the current
playback multiplier and resulting BPM. Choose **Phone (Phyphox)** to adjust
playback speed from counted SPM, within 0.75×–1.40×, or choose **Manual** to
adjust the same range with the slider without polling Phyphox. These modes are
mutually exclusive. Pitch preservation is enabled where supported by the
browser. The browser requires the user to start playback manually.

The app reads raw X/Y/Z acceleration samples and timestamps from the Phyphox
Acceleration Spectrum buffers. It estimates acceleration magnitude, removes a
slow gravity baseline, then counts local peaks above a 1.0 m/s² threshold with
a 0.3-second minimum interval. Playback speed uses the median of three to five
valid intervals between counted step events (0.8–3.0 steps/s), so it waits for
a few steps before changing and is less sensitive to a single false movement.
The live panel shows a calibration state while collecting those initial steps.
Cadence is not calculated from the FFT peak. If no step is detected for fifteen
seconds, cadence resets to rest and music speed returns to 1.0×. The live panel
shows the step count, counted cadence, raw Phyphox FFT peak, and diagnostic
sensor values; its rolling chart compares the FFT peak with cadence from
detected steps. This threshold-based prototype needs validation for the phone's
carrying position and the user's gait.