import streamlit as st
import numpy as np
import pandas as pd
import wfdb
import neurokit2 as nk
import pickle
import matplotlib.pyplot as plt
import tempfile
import os

# Load trained ML model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

import shutil  # For cleaning up the directory later


def load_signal(dat_file=None, hea_file=None, example_name=None):
    """Load and return the ECG signal and sampling rate."""
    if example_name:  # Use example file if example_name is provided
        record = wfdb.rdrecord(fr"Examples Files/{example_name}")
        sampling_rate = record.fs
        signal = record.p_signal.flatten()
        random_end = np.random.randint(len(signal) * 0.1, high=len(signal), size=1)[0]
        signal = signal[random_end - int(len(signal) * 0.1):random_end]
    else:
        temp_dir = tempfile.mkdtemp()

        temp_hea_path = os.path.join(temp_dir, "temp.hea")
        temp_dat_path = os.path.join(temp_dir, "temp.dat")

        with open(temp_hea_path, "wb") as f:
            f.write(hea_file.getvalue())
        with open(temp_dat_path, "wb") as f:
            f.write(dat_file.getvalue())

        with open(temp_hea_path, "r") as f:
            first_line = f.readline().strip().split()
            expected_base_name = first_line[0]

        final_hea_path = os.path.join(temp_dir, f"{expected_base_name}.hea")
        final_dat_path = os.path.join(temp_dir, f"{expected_base_name}.dat")
        os.rename(temp_hea_path, final_hea_path)
        os.rename(temp_dat_path, final_dat_path)

        record = wfdb.rdrecord(os.path.join(temp_dir, expected_base_name))
        sampling_rate = record.fs
        signal = record.p_signal.flatten()

    return signal, sampling_rate

@st.cache_data
def extract_features(signal, sampling_rate):
    """Extract features from 120-second overlapping windows with 60-second steps."""
    df = pd.DataFrame()
    valid_times = []  # Store the start time (in seconds) of each valid 120s window
    length_of_signal = len(signal) // sampling_rate

    # Sliding window: step size is 60 seconds, window size is 120 seconds
    for start_time in range(0, length_of_signal - 120, 60):
        try:
            start_idx = start_time * sampling_rate
            end_idx = (start_time + 120) * sampling_rate
            clean_segment = signal[start_idx:end_idx]

            # Check signal quality
            quality = np.mean(nk.ecg_quality(clean_segment, sampling_rate=sampling_rate))
            if quality >= 0.7:
                # Extract features from clean ECG segment
                peaks = nk.ecg_peaks(clean_segment, sampling_rate=sampling_rate)
                features = nk.hrv(peaks[0], sampling_rate=sampling_rate)
                df = pd.concat([df, features])
                valid_times.append(start_time + 60)  # Middle of the 120s window
        except Exception as e:
            print(f"Error processing window starting at {start_time}s: {e}")

    features_array = df.to_numpy() if not df.empty else np.array([])
    features_array[np.isinf(features_array)] = np.nan
    nan_cols = np.all(np.isnan(features_array), axis=0)
    clean_features = features_array[:, ~nan_cols]
    nan_rows = np.isnan(clean_features).any(axis=1)
    features_cleaned = clean_features[~nan_rows]

    return features_cleaned, valid_times

# Initialize session state for signal and sampling rate
if "signal" not in st.session_state:
    st.session_state.signal = None
    st.session_state.sampling_rate = None
if "example_used" not in st.session_state:
    st.session_state.example_used = None 
# Streamlit UI
st.title("ECG-Based Apnea Detection with AHI Calculation")
st.write("Upload both the **.dat** (ECG signal) and **.hea** (metadata) files or use an example file.")

# File upload
dat_file = st.file_uploader("Upload ECG signal file (.dat)", type=["dat"])
hea_file = st.file_uploader("Upload metadata header file (.hea)", type=["hea"])
st.write("You may also analyze small random parts of example ECG with different severites of Apnea:")
col1, col2, col3, col4 = st.columns(4, gap="large")
if st.button("No Apnea", use_container_width=True):
    st.session_state.signal, st.session_state.sampling_rate = load_signal(example_name="c06")
    st.session_state.example_used = True

if st.button("Mild Apnea", use_container_width=True):
    st.session_state.signal, st.session_state.sampling_rate = load_signal(example_name="x10")
    st.session_state.example_used = True

if st.button("Moderate Apnea", use_container_width=True):
    st.session_state.signal, st.session_state.sampling_rate = load_signal(example_name="a10")
    st.session_state.example_used = True

if st.button("Severe Apnea", use_container_width=True):
    st.session_state.signal, st.session_state.sampling_rate = load_signal(example_name="a01")
    st.session_state.example_used = True
if dat_file is not None and hea_file is not None:
    st.session_state.signal, st.session_state.sampling_rate = load_signal(dat_file=dat_file, hea_file=hea_file)

if st.session_state.signal is not None:
    signal = st.session_state.signal
    sampling_rate = st.session_state.sampling_rate

    # Extract features and make predictions
    features, valid_times = extract_features(signal, sampling_rate)
    if features.size == 0:
        st.error("No valid ECG features extracted.")
    else:
        predictions = model.predict(features)

        total_minutes = len(signal) / (sampling_rate * 60)

        # Each prediction corresponds to a 60-second period in the middle of each 120-second window
        apnea_minutes = np.sum(predictions) * 1  # Each prediction represents 1 minute of apnea

        # AHI = Apnea events per hour of total recording time
        if total_minutes > 0:
            AHI = (apnea_minutes / total_minutes) * 60
        else:
            AHI = 0

        # Classify severity based on AHI
        if AHI < 5:
            severity = "No Sleep Apnea"
        elif 5 <= AHI < 15:
            severity = "Mild Sleep Apnea"
        elif 15 <= AHI < 30:
            severity = "Moderate Sleep Apnea"
        else:
            severity = "Severe Sleep Apnea"

        st.subheader("Prediction Result")
        if st.session_state.example_used:  
            st.write(f"**{severity}** with an AHI of {AHI:.2f} events/hour.")
            st.write(
                "**Note:** This prediction is based on a random 10% of the signal from the selected example file. "
                "The prediction may not fully represent the overall severity of the example."
            )
        else:
            st.write(f"**{severity}** (AHI: {AHI:.2f} events/hour)")

        # Find indices for apnea and non-apnea windows
        apnea_indices = [valid_times[i] for i in np.where(predictions == 1)[0]]
        non_apnea_indices = [valid_times[i] for i in np.where(predictions == 0)[0]]
        # Plot one apnea window (if available)
        if apnea_indices:
            st.subheader("Apnea Event - 120s Window")
            time_apnea = int(np.random.choice(apnea_indices) *sampling_rate)
            end_idx = min(time_apnea + 120 * sampling_rate, len(signal))  # Ensure end index is within bounds

            if time_apnea < len(signal) and (end_idx - time_apnea) > 0:
                actual_duration = (end_idx - time_apnea) / sampling_rate
                time_window = np.arange(0, actual_duration, 1 / sampling_rate)

                fig1, ax1 = plt.subplots(figsize=(10, 4))
                ax1.plot(time_window, signal[time_apnea:end_idx], color="red", label="Apnea Segment")
                ax1.set_xlabel("Time (s)")
                ax1.set_ylabel("ECG Amplitude")
                ax1.set_title("120s Window with Apnea")
                ax1.legend()
                st.pyplot(fig1)
            else:
                st.warning("Apnea window is out of bounds or too short to display.")
        else:
            st.subheader("Apnea Event - 120s Window")
            st.info("No Apnea events were detected in the analyzed signal.")

        # Plot one non-apnea window (if available)
        if non_apnea_indices:
            st.subheader("Non-Apnea Event - 120s Window")
            time_non_apnea = int(np.random.choice(non_apnea_indices)* sampling_rate)
            time_window = np.arange(0, 120, 1 / sampling_rate)
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.plot(time_window, signal[time_non_apnea:time_non_apnea + 120 * sampling_rate], color="green", label="Non-Apnea Segment")
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("ECG Amplitude")
            ax2.set_title("120s Window without Apnea")
            ax2.legend()
            st.pyplot(fig2)
        else:
            st.subheader("Non-Apnea Event - 120s Window")
            st.info("No Non-Apnea events were detected in the analyzed signal.")
else:
    st.warning("Please upload both the .dat and .hea files or use the example file.")
