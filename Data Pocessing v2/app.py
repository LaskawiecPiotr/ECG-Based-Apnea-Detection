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

def extract_features(dat_file, hea_file):
    """Extract features from an uploaded ECG file (.dat + .hea) for apnea detection."""

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_hea_path = os.path.join(temp_dir, "temp.hea")

        # Save `.hea` file first to extract the expected filename
        with open(temp_hea_path, "wb") as f:
            f.write(hea_file.getvalue())

        # Read the expected filename from the first line of `.hea`
        with open(temp_hea_path, "r") as f:
            first_line = f.readline().strip().split()
            expected_filename = first_line[0]  # Expected base filename (e.g., 'a01')

        # Rename `.hea` file correctly
        hea_path = os.path.join(temp_dir, expected_filename + ".hea")
        os.rename(temp_hea_path, hea_path)

        # Save `.dat` file with the correct name
        dat_path = os.path.join(temp_dir, expected_filename + ".dat")
        with open(dat_path, "wb") as f:
            f.write(dat_file.getvalue())

        # Read WFDB record from saved files
        record = wfdb.rdrecord(os.path.join(temp_dir, expected_filename))

        # Extract sampling rate and signal
        sampling_rate = record.fs
        signal = record.p_signal.flatten()

        # Clean the ECG signal
        signal_clean = nk.ecg_clean(signal, sampling_rate=sampling_rate)

        # Determine signal length
        length_of_signal = len(signal) // sampling_rate

        df = pd.DataFrame()
        valid_indices = []  # To store indices of valid time windows

        # Process ECG in 120-second windows
        for i in range(length_of_signal // 120):
            try:
                start_idx = i * 120 * sampling_rate
                end_idx = (i + 1) * 120 * sampling_rate

                clean_segment = signal_clean[start_idx:end_idx]

                # Compute ECG quality
                quality = np.mean(nk.ecg_quality(clean_segment, sampling_rate=sampling_rate))

                if quality >= 0.5:
                    # Extract features
                    peaks = nk.ecg_peaks(clean_segment, sampling_rate=sampling_rate)
                    features = nk.hrv(peaks[0], sampling_rate=sampling_rate)

                    # Append features to DataFrame
                    df = pd.concat([df, features])

                    # Store the original index of this window
                    valid_indices.append(i)
                else:
                    print(f"Warning: Poor ECG quality in window {i}, skipping.")

            except Exception as e:
                print(f"Error processing window {i}: {e}")

        # Convert features to NumPy array
        features_array = df.to_numpy() if not df.empty else np.array([])

    features_array[np.isinf(features_array)] = np.nan
    nan_cols = np.all(np.isnan(features_array), axis=0)
    clean_features = features_array[:, ~nan_cols]
    nan_rows = np.isnan(clean_features).any(axis=1)
    features_cleaned = clean_features[~nan_rows]

    return features_cleaned, signal_clean, sampling_rate, valid_indices

# Streamlit UI
st.title("ECG-Based Apnea Detection")
st.write("Upload both the **.dat** (ECG signal) and **.hea** (metadata) files.")

# Upload `.dat` and `.hea` files
dat_file = st.file_uploader("Upload ECG signal file (.dat)", type=["dat"])
hea_file = st.file_uploader("Upload metadata header file (.hea)", type=["hea"])

if dat_file and hea_file:
    try:
        features, signal, sampling_rate, valid_indices = extract_features(dat_file, hea_file)

        if features.size == 0:
            st.error("No valid ECG features extracted. Check your input data.")
        else:
            predictions = model.predict(features)
            AHI = np.sum(predictions) / (len(predictions) / 60)
            label = ("You exhibit signs of mild Sleep Apnea" if 5 <= AHI <= 15 else
                     "You exhibit signs of moderate Sleep Apnea" if 15 < AHI <= 30 else
                     "You exhibit signs of severe Sleep Apnea" if AHI > 30 else
                     "You exhibit no signs of Sleep Apnea")

            st.subheader("Prediction Result")
            st.write(f"**{label}** (You have an AHI of {AHI:.2f})")

            # Map predictions back to time windows
            apnea_indices = [valid_indices[i] for i in np.where(predictions == 1)[0]]
            non_apnea_indices = [valid_indices[i] for i in np.where(predictions == 0)[0]]

            if apnea_indices:
                st.subheader("Apnea Event - 120s Window")
                time = 120 * sampling_rate * apnea_indices[0]
                time_window = np.arange(0, 120, 1 / sampling_rate)

                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(time_window, signal[time:time + 120 * sampling_rate], color="red", label="Apnea Segment")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("ECG Amplitude")
                ax.set_title("120s Window with Apnea")
                ax.legend()
                st.pyplot(fig)

            if non_apnea_indices:
                st.subheader("Non-Apnea Event - 120s Window")
                time = 120 * sampling_rate * non_apnea_indices[0]
                time_window = np.arange(0, 120, 1 / sampling_rate)

                fig2, ax2 = plt.subplots(figsize=(10, 4))
                ax2.plot(time_window, signal[time:time + 120 * sampling_rate], color="blue",
                         label="Non-Apnea Segment")
                ax2.set_xlabel("Time (s)")
                ax2.set_ylabel("ECG Amplitude")
                ax2.set_title("120s Window without Apnea")
                ax2.legend()
                st.pyplot(fig2)

    except Exception as e:
        st.error(f"Error processing file: {e}")
