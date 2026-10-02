import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy import signal
from tensorflow.keras.models import load_model

MODEL_PATH = "rf_model.keras"
CLASSES_PATH = "classes.json"
DEMO_PATH = "demo_samples.npz"
SEQ_LEN = 128

st.set_page_config(page_title="RF Signal Classifier", layout="wide")
st.title("AI-Based RF Signal Detection & Modulation Classification")


# ---------- Helpers ----------
def to_frame(arr):
    """Converts uploaded array to float32 (128, 2). Accepts complex (N,),
    real (N, 2) or (2, N). Crops or zero-pads to 128 samples."""
    arr = np.asarray(arr)
    if np.iscomplexobj(arr):
        arr = np.column_stack((arr.real.ravel(), arr.imag.ravel()))
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {arr.shape}.")
    if arr.shape[1] != 2 and arr.shape[0] == 2:
        arr = arr.T
    if arr.shape[1] != 2:
        raise ValueError(f"Expected shape (N, 2) or (2, N), got {arr.shape}.")
    arr = arr.astype(np.float32)
    if len(arr) >= SEQ_LEN:
        arr = arr[:SEQ_LEN]
    else:
        arr = np.vstack((arr, np.zeros((SEQ_LEN - len(arr), 2), np.float32)))
    return arr


def normalize(frame):
    """Same per-frame power normalization used during training."""
    return frame / (np.sqrt(np.mean(frame ** 2)) + 1e-8)


@st.cache_resource
def get_model(path):
    return load_model(path)


@st.cache_data
def get_classes(path):
    with open(path) as f:
        return json.load(f)


@st.cache_data
def get_demo(path):
    d = np.load(path)
    return d["X"], d["y"], d["snr"]


# ---------- Required files ----------
if not os.path.exists(CLASSES_PATH):
    st.error(f"'{CLASSES_PATH}' not found. Run rf_modulation_cnn.py first "
             "to train the model on the dataset.")
    st.stop()

CLASSES = get_classes(CLASSES_PATH)

# ---------- Sidebar ----------
st.sidebar.header("Signal Input")
source = st.sidebar.radio("Source", ["Dataset test frame", "Upload .npy"])

true_label, true_snr = None, None

if source == "Upload .npy":
    uploaded_file = st.sidebar.file_uploader("Upload I/Q Array (.npy)", type=["npy"])
    if uploaded_file is None:
        st.info("Upload a .npy file in the sidebar, or switch to a dataset test frame.")
        st.stop()
    try:
        iq_data = to_frame(np.load(uploaded_file))
    except Exception as e:
        st.sidebar.error(f"Could not read file: {e}")
        st.stop()
else:
    if not os.path.exists(DEMO_PATH):
        st.error(f"'{DEMO_PATH}' not found. Run rf_modulation_cnn.py first.")
        st.stop()
    Xd, yd, sd = get_demo(DEMO_PATH)
    mod_name = st.sidebar.selectbox("Modulation", CLASSES)
    snr_options = sorted({int(s) for s in sd})
    snr_sel = st.sidebar.select_slider("SNR (dB)", options=snr_options,
                                       value=snr_options[-1])
    cls_idx = CLASSES.index(mod_name)
    idxs = np.where((yd == cls_idx) & (sd == snr_sel))[0]
    if len(idxs) == 0:
        st.warning("No demo frames for this combination.")
        st.stop()
    frame_no = st.sidebar.slider("Frame", 1, len(idxs), 1) if len(idxs) > 1 else 1
    iq_data = Xd[idxs[frame_no - 1]].astype(np.float32)
    true_label, true_snr = mod_name, snr_sel

# ---------- Plots ----------
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Time Domain (I/Q)")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(iq_data[:, 0], label="I", color="blue", alpha=0.7)
    ax.plot(iq_data[:, 1], label="Q", color="orange", alpha=0.7)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

with col2:
    st.subheader("Constellation")
    fig3, ax3 = plt.subplots(figsize=(5, 4))
    ax3.scatter(iq_data[:, 0], iq_data[:, 1], s=10, alpha=0.6)
    ax3.set_xlabel("I")
    ax3.set_ylabel("Q")
    ax3.set_aspect("equal", adjustable="datalim")
    ax3.grid(True, alpha=0.3)
    st.pyplot(fig3)
    plt.close(fig3)

with col3:
    st.subheader("Spectrogram")
    complex_sig = iq_data[:, 0] + 1j * iq_data[:, 1]
    f, t_spec, Sxx = signal.spectrogram(
        complex_sig, fs=1.0, nperseg=32, noverlap=24, return_onesided=False
    )
    Sxx_db = 10 * np.log10(np.fft.fftshift(Sxx, axes=0) + 1e-12)
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    cax = ax2.pcolormesh(t_spec, np.fft.fftshift(f), Sxx_db,
                         shading="gouraud", cmap="viridis")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Normalized frequency")
    fig2.colorbar(cax, ax=ax2, label="Power (dB)")
    st.pyplot(fig2)
    plt.close(fig2)

# ---------- Inference ----------
st.divider()
st.subheader("Signal Classification")

if st.button("Run Inference", type="primary"):
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file '{MODEL_PATH}' not found. "
                 "Run rf_modulation_cnn.py first.")
    else:
        try:
            model = get_model(MODEL_PATH)
            x = normalize(iq_data)[np.newaxis, ...]
            probs = model.predict(x, verbose=0)[0]

            if len(probs) != len(CLASSES):
                st.error(f"Model outputs {len(probs)} classes but classes.json "
                         f"has {len(CLASSES)}. Retrain or fix classes.json.")
            else:
                idx = int(np.argmax(probs))
                conf = float(probs[idx])

                c1, c2 = st.columns(2)
                with c1:
                    st.success(f"Detected Modulation: **{CLASSES[idx]}**")
                    st.metric("Confidence", f"{conf * 100:.2f}%")
                    st.progress(conf)
                    if true_label is not None:
                        ok = CLASSES[idx] == true_label
                        st.caption(
                            f"Ground truth: {true_label} at {true_snr} dB "
                            f"({'correct' if ok else 'incorrect'})"
                        )
                    if conf < 0.5:
                        st.warning("Low confidence. The signal may be too noisy "
                                   "or outside the training distribution.")
                with c2:
                    st.bar_chart(pd.DataFrame({"Probability": probs}, index=CLASSES))
        except Exception as e:
            st.error(f"Inference failed: {e}")
