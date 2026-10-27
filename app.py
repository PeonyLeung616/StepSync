import base64
import hashlib
import streamlit as st
import requests
import numpy as np
import math
import statistics
import time

st.set_page_config(page_title="StepSync Demo", page_icon="🎵", layout="centered")

_AUDIO_PLAYER = st.components.v2.component(
    "stepsync_audio_player",
    html="<audio controls preload='metadata' style='width: 100%;'></audio>",
    js="""
export default function(component) {
    const { data, parentElement } = component;
    const audio = parentElement.querySelector("audio");
    if (!audio) return;

    if (audio.dataset.sourceId !== data.sourceId) {
        audio.pause();
        audio.src = data.source;
        audio.dataset.sourceId = data.sourceId;
        audio.load();
    }

    audio.preservesPitch = true;
    if ("webkitPreservesPitch" in audio) audio.webkitPreservesPitch = true;
    audio.playbackRate = data.playbackRate;

    if (!audio._stepsyncRateListener) {
        audio._stepsyncRateListener = (event) => {
            const rate = Number(event.detail?.rate);
            if (Number.isFinite(rate) && rate >= 0.75 && rate <= 1.4) {
                audio.playbackRate = rate;
            }
        };
        window.addEventListener("stepsync-playback-rate", audio._stepsyncRateListener);
    }

    return () => {
        if (audio._stepsyncRateListener) {
            window.removeEventListener(
                "stepsync-playback-rate",
                audio._stepsyncRateListener
            );
            delete audio._stepsyncRateListener;
        }
    };
}
""",
)

_PLAYBACK_RATE_BRIDGE = st.components.v2.component(
    "stepsync_playback_rate_bridge",
    js="""
export default function({ data }) {
    window.dispatchEvent(
        new CustomEvent("stepsync-playback-rate", {
            detail: { rate: data.rate }
        })
    );
}
""",
)

st.title("StepSync")
st.markdown("*Music that follows your stride*")

# --- Sidebar / State Initialization ---
if "spm" not in st.session_state:
    st.session_state.spm = 0.0
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "cadence_plot_history" not in st.session_state:
    st.session_state.cadence_plot_history = []
if "step_count" not in st.session_state:
    st.session_state.step_count = 0
if "step_event_times" not in st.session_state:
    st.session_state.step_event_times = []
if "step_detector_state" not in st.session_state:
    st.session_state.step_detector_state = None
if "last_accel_sample_time" not in st.session_state:
    st.session_state.last_accel_sample_time = None
if "step_interval_history" not in st.session_state:
    st.session_state.step_interval_history = []
if "last_step_at_monotonic" not in st.session_state:
    st.session_state.last_step_at_monotonic = None

MIN_CADENCE_HZ = 0.8
MAX_CADENCE_HZ = 3.0
MIN_CADENCE_INTERVALS = 3
CADENCE_HISTORY_SIZE = 5
CADENCE_PLOT_HISTORY_SIZE = 300
INACTIVITY_TIMEOUT_SECONDS = 15.0
STEP_PEAK_THRESHOLD_MPS2 = 1.0
STEP_MIN_INTERVAL_SECONDS = 0.3
GRAVITY_BASELINE_TIME_CONSTANT_SECONDS = 0.6


def cadence_from_step_times(step_times):
    intervals = [
        current_time - previous_time
        for previous_time, current_time in zip(step_times, step_times[1:])
    ]
    valid_intervals = [
        interval
        for interval in intervals
        if 1 / MAX_CADENCE_HZ <= interval <= 1 / MIN_CADENCE_HZ
    ]
    if len(valid_intervals) < MIN_CADENCE_INTERVALS:
        return None

    return 1 / statistics.median(valid_intervals)


def latest_buffer_value(buffers, buffer_id):
    buffer_data = buffers.get(buffer_id)
    if not isinstance(buffer_data, dict):
        return None

    values = buffer_data.get("buffer")
    if not isinstance(values, list) or not values:
        return None

    try:
        return float(values[-1])
    except (TypeError, ValueError):
        return None


def full_buffer_values(buffers, buffer_id):
    buffer_data = buffers.get(buffer_id)
    if not isinstance(buffer_data, dict):
        return []

    values = buffer_data.get("buffer")
    if not isinstance(values, list):
        return []

    try:
        return [float(value) for value in values]
    except (TypeError, ValueError):
        return []


def process_acceleration_samples(times, x_values, y_values, z_values, detector_state):
    sample_count = min(len(times), len(x_values), len(y_values), len(z_values))
    samples = []
    for index in range(sample_count):
        sample = (
            times[index],
            x_values[index],
            y_values[index],
            z_values[index],
        )
        if all(math.isfinite(value) for value in sample):
            samples.append(sample)

    if not samples:
        return detector_state, [], None

    magnitudes = [
        math.sqrt(x_value**2 + y_value**2 + z_value**2)
        for _, x_value, y_value, z_value in samples
    ]
    if detector_state is None:
        baseline = statistics.mean(magnitudes[-min(len(magnitudes), 100):])
        previous_samples = samples[-2:]
        previous_dynamics = [
            magnitude - baseline
            for magnitude in magnitudes[-len(previous_samples):]
        ]
        detector_state = {
            "baseline": baseline,
            "sample_time": samples[-1][0],
            "previous_dynamic": previous_dynamics[-1],
            "previous_dynamic_time": previous_samples[-1][0],
            "previous_previous_dynamic": (
                previous_dynamics[-2] if len(previous_dynamics) == 2 else None
            ),
            "previous_previous_time": (
                previous_samples[-2][0] if len(previous_samples) == 2 else None
            ),
            "last_peak_time": None,
            "last_step_time": None,
            "step_intervals": [],
        }
        return detector_state, [], samples[-1][0]

    detected_step_times = []
    for (sample_time, x_value, y_value, z_value), magnitude in zip(samples, magnitudes):
        if sample_time <= detector_state["sample_time"]:
            continue

        delta_time = sample_time - detector_state["sample_time"]
        baseline_alpha = delta_time / (
            GRAVITY_BASELINE_TIME_CONSTANT_SECONDS + delta_time
        )
        detector_state["baseline"] += baseline_alpha * (
            magnitude - detector_state["baseline"]
        )
        dynamic_acceleration = magnitude - detector_state["baseline"]

        previous_dynamic = detector_state["previous_dynamic"]
        previous_time = detector_state["previous_dynamic_time"]
        previous_previous_dynamic = detector_state["previous_previous_dynamic"]
        previous_previous_time = detector_state["previous_previous_time"]
        is_peak = (
            previous_previous_dynamic is not None
            and previous_dynamic > previous_previous_dynamic
            and previous_dynamic >= dynamic_acceleration
            and previous_dynamic >= STEP_PEAK_THRESHOLD_MPS2
        )
        if is_peak:
            time_since_peak = (
                previous_time - detector_state["last_peak_time"]
                if detector_state["last_peak_time"] is not None
                else float("inf")
            )
            if time_since_peak >= STEP_MIN_INTERVAL_SECONDS:
                detector_state["last_peak_time"] = previous_time
                previous_step_time = detector_state["last_step_time"]
                if previous_step_time is None:
                    detected_step_times.append(previous_time)
                    detector_state["last_step_time"] = previous_time
                else:
                    step_interval = previous_time - previous_step_time
                    if step_interval >= STEP_MIN_INTERVAL_SECONDS:
                        detected_step_times.append(previous_time)
                        detector_state["last_step_time"] = previous_time
                        step_frequency = 1 / step_interval
                        step_intervals = detector_state["step_intervals"]
                        if MIN_CADENCE_HZ <= step_frequency <= MAX_CADENCE_HZ:
                            step_intervals.append(step_interval)
                            detector_state["step_intervals"] = step_intervals[
                                -CADENCE_HISTORY_SIZE:
                            ]
                        else:
                            detector_state["step_intervals"] = []

        detector_state["previous_previous_dynamic"] = previous_dynamic
        detector_state["previous_previous_time"] = previous_time
        detector_state["previous_dynamic"] = dynamic_acceleration
        detector_state["previous_dynamic_time"] = sample_time
        detector_state["sample_time"] = sample_time

    return detector_state, detected_step_times, samples[-1][0]


# --- Music Source Section ---
st.subheader("Music source")
uploaded_file = st.file_uploader("Upload Music...", type=["mp3", "wav", "m4a"])
base_bpm = st.number_input(
    "Track's original tempo (BPM)",
    min_value=40,
    max_value=240,
    value=112,
    step=1,
)
if uploaded_file is not None:
    st.success(f"Loaded: {uploaded_file.name} ({base_bpm} BPM)")
    audio_bytes = uploaded_file.getvalue()
    source_id = hashlib.sha256(audio_bytes).hexdigest()
    extension = uploaded_file.name.rsplit(".", 1)[-1].lower()
    mime_type = {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "m4a": "audio/mp4",
    }.get(extension)
    if mime_type is None:
        st.error("Unsupported audio format. Upload an MP3, WAV, or M4A file.")
    else:
        audio_source = (
            f"data:{mime_type};base64,"
            f"{base64.b64encode(audio_bytes).decode('ascii')}"
        )
        _AUDIO_PLAYER(
            data={
                "source": audio_source,
                "sourceId": source_id,
                "playbackRate": 1.0,
            },
            key="stepsync_audio_player",
            height=80,
        )
        st.caption(
            "Press play to start the track. Playback speed follows your counted "
            "cadence while pitch is preserved; audio playback must be started "
            "manually in the browser."
        )

# --- Cadence Input Section ---
st.subheader("Cadence input")
cadence_source = st.radio(
    "Playback control",
    ["Phone (Phyphox)", "Manual"],
    horizontal=True,
    key="cadence_source",
)
if "last_cadence_source" not in st.session_state:
    st.session_state.last_cadence_source = cadence_source
elif st.session_state.last_cadence_source != cadence_source:
    st.session_state.is_running = False
    st.session_state.spm = 0.0
    st.session_state.step_count = 0
    st.session_state.step_event_times = []
    st.session_state.step_detector_state = None
    st.session_state.last_accel_sample_time = None
    st.session_state.step_interval_history = []
    st.session_state.last_step_at_monotonic = None
    st.session_state.filtered_cadence_hz = None
    st.session_state.last_phone_measurement = None
    st.session_state.cadence_plot_history = []
    st.session_state.last_cadence_source = cadence_source

if cadence_source == "Phone (Phyphox)":
    phyphox_code = st.text_input(
        "Phyphox Remote Access URL",
        value="",
        placeholder="Paste the code xxx.xxx.xxx.xxx shown in Phyphox Remote Access",
        help=(
            "In the Phyphox app, open the experiment menu (⋮) → Remote Access, "
            "then paste the displayed URL here."
        ),
    )
    phyphox_url = f"http://{phyphox_code}"
else:
    manual_speed_ratio = st.slider(
        "Manual playback speed (×)",
        min_value=0.75,
        max_value=1.40,
        value=1.0,
        step=0.01,
        help="Adjust the music speed directly to test playback without Phyphox.",
    )

# --- Display Metrics ---
st.markdown("---")

# --- Control Button ---
if cadence_source == "Phone (Phyphox)":
    if st.button("Start walking" if not st.session_state.is_running else "Stop"):
        st.session_state.is_running = not st.session_state.is_running
        if st.session_state.is_running:
            st.session_state.step_count = 0
            st.session_state.step_event_times = []
            st.session_state.step_detector_state = None
            st.session_state.last_accel_sample_time = None
            st.session_state.step_interval_history = []
            st.session_state.last_step_at_monotonic = None
            st.session_state.filtered_cadence_hz = None
            st.session_state.last_phone_measurement = None
            st.session_state.step_source_url = phyphox_url.rstrip("/")
            st.session_state.cadence_plot_history = []
        st.rerun()


def display_playback_speed(speed_ratio, base_bpm):
    target_bpm = round(base_bpm * speed_ratio)
    _PLAYBACK_RATE_BRIDGE(
        data={"rate": float(speed_ratio)},
        key="stepsync_playback_rate_bridge",
        height=1,
    )
    st.markdown(
        f"<h1 style='text-align: center; font-size: clamp(2rem, 7vw, 3.5rem);'>"
        f"Speed x{speed_ratio:.2f} → {target_bpm} BPM</h1>",
        unsafe_allow_html=True,
    )
    st.caption("Pitch preserved where supported by your browser.")

# --- Live Fetch & Calculation Logic ---
@st.fragment(run_every="0.5s")
def live_metrics(phyphox_url, base_bpm):
    normalized_url = phyphox_url.rstrip("/")
    if st.session_state.get("step_source_url") != normalized_url:
        st.session_state.step_source_url = normalized_url
        st.session_state.step_count = 0
        st.session_state.step_event_times = []
        st.session_state.step_detector_state = None
        st.session_state.last_accel_sample_time = None
        st.session_state.step_interval_history = []
        st.session_state.last_step_at_monotonic = None
        st.session_state.filtered_cadence_hz = None
        st.session_state.last_phone_measurement = None
        st.session_state.cadence_plot_history = []

    sample_cursor = st.session_state.last_accel_sample_time
    requested_buffers = {
        "f0": "",
        "t": "",
        "actual_samples": "",
        "period": "",
        "df": "",
        "arate": "",
        "maxf": "",
        "halfFrequency": "full",
        "fft": "full",
        "time": "full" if sample_cursor is None else str(sample_cursor),
        "accX": "full" if sample_cursor is None else f"{sample_cursor}|time",
        "accY": "full" if sample_cursor is None else f"{sample_cursor}|time",
        "accZ": "full" if sample_cursor is None else f"{sample_cursor}|time",
    }
    request_url = f"{normalized_url}/get"
    latest_frequency_text = ""
    phone_values = {}
    spectrum_frequencies = []
    spectrum_magnitudes = []
    measurement_is_fresh = None
    inactivity_timed_out = False
    phone_measuring = None
    try:
        response = requests.get(
            request_url,
            params=requested_buffers,
            timeout=2.0,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        st.error(f"Could not reach Phyphox at {request_url}: {error}")
    else:
        try:
            data = response.json()
        except ValueError as error:
            st.error(f"Phyphox returned invalid JSON: {error}")
        else:
            buffers = data.get("buffer") if isinstance(data, dict) else None
            status = data.get("status") if isinstance(data, dict) else None
            if isinstance(status, dict):
                phone_measuring = status.get("measuring")
            if not isinstance(buffers, dict):
                st.error("Phyphox response has no buffer object. Check the Remote Access URL and experiment.")
            elif "f0" not in buffers:
                st.error(
                    "Phyphox did not return the requested 'f0' buffer. Check that "
                    "the Acceleration Spectrum experiment is running and that the "
                    "Remote Access URL points to that phone."
                )
            else:
                phone_values = {
                    buffer_id: latest_buffer_value(buffers, buffer_id)
                    for buffer_id in (
                        "f0",
                        "t",
                        "actual_samples",
                        "period",
                        "df",
                        "arate",
                        "maxf",
                    )
                }
                spectrum_frequencies = full_buffer_values(buffers, "halfFrequency")
                spectrum_magnitudes = full_buffer_values(buffers, "fft")
                sample_times = full_buffer_values(buffers, "time")
                acceleration_x = full_buffer_values(buffers, "accX")
                acceleration_y = full_buffer_values(buffers, "accY")
                acceleration_z = full_buffer_values(buffers, "accZ")

                raw_frequency = phone_values["f0"]
                measurement_time = phone_values["t"]
                if (
                    raw_frequency is None
                    or measurement_time is None
                    or not math.isfinite(raw_frequency)
                    or not math.isfinite(measurement_time)
                ):
                    st.warning(
                        "Phyphox did not return a valid peak frequency or measurement "
                        "timestamp. Make sure the Acceleration Spectrum experiment is running."
                    )
                else:
                    last_measurement = st.session_state.get("last_phone_measurement")
                    same_source = (
                        last_measurement is not None
                        and last_measurement["url"] == phyphox_url.rstrip("/")
                    )
                    measurement_is_fresh = (
                        not same_source or measurement_time != last_measurement["time"]
                    )
                    timer_reset = (
                        same_source
                        and measurement_time < last_measurement["time"]
                    )
                    if timer_reset:
                        st.session_state.step_count = 0
                        st.session_state.step_event_times = []
                        st.session_state.step_detector_state = None
                        st.session_state.last_accel_sample_time = None
                        st.session_state.step_interval_history = []
                        st.session_state.last_step_at_monotonic = None
                        st.session_state.filtered_cadence_hz = None
                        st.session_state.cadence_plot_history = []

                    if (
                        sample_times
                        and acceleration_x
                        and acceleration_y
                        and acceleration_z
                        and not timer_reset
                    ):
                        detector_state = st.session_state.step_detector_state
                        was_initializing = detector_state is None
                        (
                            detector_state,
                            detected_step_times,
                            latest_sample_time,
                        ) = process_acceleration_samples(
                            sample_times,
                            acceleration_x,
                            acceleration_y,
                            acceleration_z,
                            detector_state,
                        )
                        st.session_state.step_detector_state = detector_state
                        st.session_state.last_accel_sample_time = latest_sample_time

                        if not was_initializing and detected_step_times:
                            st.session_state.step_count += len(detected_step_times)
                            st.session_state.last_step_at_monotonic = time.monotonic()
                            step_event_times = st.session_state.step_event_times
                            step_event_times.extend(detected_step_times)
                            st.session_state.step_event_times = step_event_times[-(
                                CADENCE_HISTORY_SIZE + 1
                            ):]
                            filtered_cadence_hz = cadence_from_step_times(
                                st.session_state.step_event_times
                            )
                            st.session_state.filtered_cadence_hz = filtered_cadence_hz
                            st.session_state.spm = (
                                filtered_cadence_hz * 60
                                if filtered_cadence_hz is not None
                                else 0.0
                            )
                        elif was_initializing:
                            st.caption(
                                "Loading recent accelerometer samples; step counting "
                                "starts with new samples."
                            )
                    elif not timer_reset:
                        st.warning(
                            "Raw accelerometer buffers (time, accX, accY, accZ) were "
                            "not returned. Step counting needs these buffers from the "
                            "Acceleration Spectrum experiment."
                        )

                    if measurement_is_fresh:
                        st.session_state.cadence_plot_history.append(
                            {
                                "measurement_time_s": measurement_time,
                                "raw_peak_hz": raw_frequency,
                                "filtered_cadence_hz": st.session_state.get(
                                    "filtered_cadence_hz"
                                ),
                            }
                        )
                        st.session_state.cadence_plot_history = (
                            st.session_state.cadence_plot_history[
                                -CADENCE_PLOT_HISTORY_SIZE:
                            ]
                        )
                        st.session_state.last_phone_measurement = {
                            "url": phyphox_url.rstrip("/"),
                            "time": measurement_time,
                        }
                    latest_frequency_text = (
                        f"<br>Raw Phyphox peak: {raw_frequency:.2f} Hz "
                        f"({raw_frequency * 60:.2f} 1/min)"
                    )
                    if not measurement_is_fresh:
                        st.warning(
                            "The phone's measurement timestamp has not advanced. "
                            "Waiting for a new Phyphox measurement."
                        )
                missing_buffers = [
                    buffer_id
                    for buffer_id, value in phone_values.items()
                    if value is None
                ]
                if missing_buffers:
                    st.warning(
                        "Phyphox did not return values for: "
                        + ", ".join(missing_buffers)
                    )

    last_step_at = st.session_state.get("last_step_at_monotonic")
    if (
        last_step_at is not None
        and time.monotonic() - last_step_at >= INACTIVITY_TIMEOUT_SECONDS
    ):
        st.session_state.filtered_cadence_hz = None
        st.session_state.spm = 0.0
        st.session_state.step_event_times = []
        st.session_state.step_interval_history = []
        st.session_state.last_step_at_monotonic = None
        last_step_at = None
        inactivity_timed_out = True

    current_spm = st.session_state.spm
    filtered_cadence_hz = st.session_state.get("filtered_cadence_hz")
    if inactivity_timed_out:
        st.info(
            f"No steps detected for {INACTIVITY_TIMEOUT_SECONDS:.0f} seconds. "
            "Cadence reset to rest and music speed returned to 1.0×."
        )
    has_recent_valid_cadence = (
        st.session_state.get("last_step_at_monotonic") is not None
        and filtered_cadence_hz is not None
    )
    speed_ratio = (
        np.clip(current_spm / base_bpm, 0.75, 1.40)
        if has_recent_valid_cadence
        else 1.0
    )

    # Update UI metrics
    step_count_column, cadence_column = st.columns(2)
    step_count_column.metric("Steps detected", st.session_state.step_count)
    if filtered_cadence_hz is not None:
        cadence_display = f"{current_spm:.0f} SPM"
    elif last_step_at is not None:
        cadence_display = "Calibrating cadence"
    else:
        cadence_display = "Rest / waiting"
    cadence_column.metric(
        "Current cadence",
        cadence_display,
    )
    display_playback_speed(speed_ratio, base_bpm)
    if latest_frequency_text:
        st.markdown(
            f"<p style='text-align: center;'>{latest_frequency_text.removeprefix('<br>')}</p>",
            unsafe_allow_html=True,
        )
    with st.expander("Live values from the Phyphox phone", expanded=False):
        st.caption(
            "Step count comes from local peaks in the time-domain accelerometer "
            "magnitude (X/Y/Z), with a gravity baseline removed and a minimum "
            f"{STEP_MIN_INTERVAL_SECONDS:.1f}s interval to prevent double-counts. "
            f"Playback follows the median of {MIN_CADENCE_INTERVALS} to "
            f"{CADENCE_HISTORY_SIZE} valid intervals between detected steps, "
            "so it waits for a few steps before changing speed. Cadence is shown "
            "as calibrating during this initial period. The FFT peak is "
            "shown separately for comparison."
        )
        if phone_values:
            frequency_hz = phone_values.get("f0")
            value_columns = st.columns(2)
            value_columns[0].metric(
                "Phone peak (Hz)",
                f"{frequency_hz:.2f} Hz" if frequency_hz is not None else "—",
            )
            value_columns[1].metric(
                "Phone peak (1/min)",
                f"{frequency_hz * 60:.2f} 1/min" if frequency_hz is not None else "—",
            )
            value_columns = st.columns(2)
            value_columns[0].metric(
                "Accelerometer samples received",
                len(sample_times),
            )
            value_columns[1].metric(
                "Counted cadence",
                f"{filtered_cadence_hz:.2f} steps/s"
                if filtered_cadence_hz is not None
                else "Waiting for consecutive steps",
            )
            if filtered_cadence_hz is not None:
                st.caption(
                    f"Filtered estimate: {filtered_cadence_hz * 60:.1f} steps/min"
                )

            if st.session_state.cadence_plot_history:
                chart_history = st.session_state.cadence_plot_history
                st.line_chart(
                    {
                        "Measurement time (s)": [
                            point["measurement_time_s"] for point in chart_history
                        ],
                        "Raw phone peak (Hz)": [
                            point["raw_peak_hz"] for point in chart_history
                        ],
                        "Filtered cadence (Hz)": [
                            point["filtered_cadence_hz"]
                            for point in chart_history
                        ],
                    },
                    x="Measurement time (s)",
                    y=["Raw phone peak (Hz)", "Filtered cadence (Hz)"],
                    height=280,
                )
                st.caption(
                    f"Latest {len(chart_history)} fresh readings. The raw series "
                    "is the phone's FFT peak; the filtered series is derived from "
                    "detected step intervals."
                )

            detail_columns = st.columns(3)
            phone_readings = (
                ("Measurement time", "t", "s"),
                ("Samples used", "actual_samples", ""),
                ("Period used", "period", "s"),
                ("Resolution", "df", "Hz"),
                ("Acquisition rate", "arate", "Hz"),
                ("Nyquist frequency", "maxf", "Hz"),
            )
            for index, (label, buffer_id, unit) in enumerate(phone_readings):
                value = phone_values.get(buffer_id)
                formatted_value = "—" if value is None else f"{value:.2f} {unit}".strip()
                detail_columns[index % len(detail_columns)].metric(label, formatted_value)

            if measurement_is_fresh is not None:
                st.caption(
                    "Phyphox measurement status: "
                    + ("new data received" if measurement_is_fresh else "timestamp unchanged")
                )

            if spectrum_frequencies and spectrum_magnitudes:
                point_count = min(len(spectrum_frequencies), len(spectrum_magnitudes))
                if len(spectrum_frequencies) != len(spectrum_magnitudes):
                    st.warning(
                        "The frequency and spectrum buffers have different lengths; "
                        f"showing their first {point_count} matching points."
                    )
                st.line_chart(
                    {
                        "Frequency (Hz)": spectrum_frequencies[:point_count],
                        "FFT magnitude": spectrum_magnitudes[:point_count],
                    },
                    x="Frequency (Hz)",
                    y="FFT magnitude",
                    height=280,
                )
                st.caption(
                    f"Fourier Transform from phone buffers "
                    f"`halfFrequency` and `fft` ({point_count} points)."
                )
            else:
                st.warning("The phone returned no spectrum points to plot.")
        else:
            st.info("Waiting for the first Phyphox response.")

        if phone_measuring is not None:
            st.caption(
                "Phyphox measurement: "
                + ("running" if phone_measuring else "paused")
            )

    if not has_recent_valid_cadence and not inactivity_timed_out:
        if last_step_at is not None:
            st.caption(
                "Collecting steps for the cadence estimate; music stays at normal "
                "speed until it is ready."
            )
        else:
            st.caption("Waiting for steps; music speed is normal (1.0×).")


if cadence_source == "Manual":
    display_playback_speed(manual_speed_ratio, base_bpm)
    st.caption("Manual mode is active; Phyphox polling and step counting are off.")
elif st.session_state.is_running:
    live_metrics(phyphox_url, base_bpm)
else:
    display_playback_speed(1.0, base_bpm)
    st.metric("Steps detected", st.session_state.step_count)
    st.metric("Current cadence", "Rest / waiting")
    st.markdown(
        "<p style='text-align: center;'>Click 'Start walking' to begin streaming cadence.</p>",
        unsafe_allow_html=True,
    )