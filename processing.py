import time

def run_stepsync_loop(phyphox_ip, music_file_path, track_bpm=112):
    client = PhyphoxClient(base_url=phyphox_ip)
    engine = StepSyncAudioEngine(music_file_path, base_track_bpm=track_bpm)
    
    print("StepSync live sync loop initialized. Start walking...")
    
    try:
        while True:
            # 1. Fetch raw SPM from phone
            raw_spm = client.fetch_current_spm()
            
            # 2. Smooth out tracking noise
            current_spm = client.get_smoothed_spm(raw_spm)
            
            # 3. Calculate dynamic speed multiplier
            speed_multiplier = engine.calculate_speed_ratio(current_spm)
            
            print(f"Current SPM: {current_spm:.1f} | Speed Multiplier: x{speed_multiplier:.2f}")
            
            # 4. Stream / update audio buffer parameters dynamically
            # (In your Streamlit or web UI, this updates the playbackRate state)
            
            time.sleep(0.5) # Poll interval
            
    except KeyboardInterrupt:
        print("Session stopped.")