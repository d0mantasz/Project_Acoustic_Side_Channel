# app.py
# Streamlit GUI for acoustic keystroke demo (live-current model)
# Domantas Prialgauskas · C00285607
#
# Key features:
# - Loads your live-current SVM + scaler from kbd_live_pipeline.py paths
# - 3 modes: single-letter vote, fixed-length word, sentence
# - Mic selector (input device) + records at device SR then resamples to model SR
#   -> avoids common PortAudio "MME error 1" / sample-rate issues

import numpy as np
import streamlit as st
import joblib

import sounddevice as sd
import librosa

import kbd_live_pipeline as kp  # must be in same folder as app.py


st.set_page_config(page_title="Acoustic Keystroke Decoder", layout="wide")

st.title("Acoustic Keystroke Classification (Live Demo)")
st.caption("Live-current model + RMS segmentation + MFCC + SVM + word decoding")
st.caption("Domantas Prialgauskas · C00285607")


# -----------------------------
# Load model/scaler once
# -----------------------------
@st.cache_resource
def load_model():
    if not (kp.os.path.exists(kp.SVM_OUT) and kp.os.path.exists(kp.SCALER_OUT)):
        return None, None
    svm = joblib.load(kp.SVM_OUT)
    scaler = joblib.load(kp.SCALER_OUT)
    return svm, scaler


svm, scaler = load_model()

if svm is None or scaler is None:
    st.error("Model not found. Train first: `python kbd_live_pipeline.py train` or copy `models_live_current/` into this folder.")
    st.stop()


# -----------------------------
# Model SR (for resampling)
# -----------------------------
MODEL_SR = int(getattr(kp, "SR", 44100))
try:
    if kp.os.path.exists(kp.META_OUT):
        meta = joblib.load(kp.META_OUT)
        # your meta often stores sr_device
        MODEL_SR = int(meta.get("sr_device", MODEL_SR))
except Exception:
    pass


# -----------------------------
# Audio devices (input only)
# -----------------------------
def _list_input_devices():
    devs = sd.query_devices()
    out = []
    for i, d in enumerate(devs):
        try:
            if int(d.get("max_input_channels", 0)) > 0:
                out.append({
                    "index": i,
                    "name": d.get("name", f"Device {i}"),
                    "sr": int(round(float(d.get("default_samplerate", 44100))))
                })
        except Exception:
            continue
    return out


INPUT_DEVS = _list_input_devices()
if not INPUT_DEVS:
    st.error("No microphone input devices found (sounddevice/PortAudio).")
    st.stop()


# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Controls")

mode = st.sidebar.radio(
    "Mode",
    ["Single letter (vote)", "Fixed-length word", "Sentence"],
    index=0
)

# Suggested demo inputs (for lecturers)
SUGGESTED = {
    "Single letter (vote)": {
        "title": "Suggested single-letter tests",
        "items": ["a", "q", "t", "b", "<SPACE> (press spacebar)"]
    },
    "Fixed-length word": {
        "title": "Suggested word tests",
        "items": ["this", "vote", "quit", "devil", "line"]
    },
    "Sentence": {
        "title": "Suggested sentence tests",
        "items": ["this is a test", "vote in the quiz", "quit the app"]
    }
}

with st.expander("🎯 Suggested demo inputs (for lecturers)", expanded=True):
    st.markdown(f"**{SUGGESTED[mode]['title']}**")
    for x in SUGGESTED[mode]["items"]:
        st.markdown(f"- `{x}`")


# Recommended presets (auto-apply when switching modes)
MODE_PRESETS = {
    "Single letter (vote)": {
        "rec_seconds": 8.0,
        "vote_n": 3,
        "temp": 1.2,
        "beam_width": 50,
        "topk_per_pos": 8,
    },
    "Fixed-length word": {
        "rec_seconds": 6.0,
        "word_len": 4,
        "temp": 1.3,
        "beam_width": 60,
        "topk_per_pos": 9,
    },
    "Sentence": {
        "rec_seconds": 12.0,
        "temp": 1.6,
        "beam_width": 120,
        "topk_per_pos": 12,
        "sent_threshold_k": 1.4,
        "sent_debounce_ms": 60,
    },
}

# init last mode
if "last_mode" not in st.session_state:
    st.session_state.last_mode = mode

# seed defaults for current mode
for k, v in MODE_PRESETS.get(mode, {}).items():
    st.session_state.setdefault(k, v)

# mode switch -> apply presets
if st.session_state.last_mode != mode:
    for k, v in MODE_PRESETS.get(mode, {}).items():
        st.session_state[k] = v
    st.session_state.last_mode = mode


# ---- Microphone selector
st.sidebar.subheader("Microphone")

dev_labels = [f"[{d['index']}] {d['name']}  (default {d['sr']} Hz)" for d in INPUT_DEVS]

# pick a default based on kp.DEVICE if it’s set, otherwise first input device
default_choice = 0
try:
    if getattr(kp, "DEVICE", None) is not None:
        want = int(kp.DEVICE)
        for j, d in enumerate(INPUT_DEVS):
            if d["index"] == want:
                default_choice = j
                break
except Exception:
    pass

chosen_dev_label = st.sidebar.selectbox("Input device", dev_labels, index=default_choice)
chosen_dev = INPUT_DEVS[dev_labels.index(chosen_dev_label)]
DEVICE_INDEX = int(chosen_dev["index"])
DEVICE_SR = int(chosen_dev["sr"])

st.sidebar.caption(f"Record SR={DEVICE_SR} Hz → Resample to model SR={MODEL_SR} Hz")


# ---- Sliders
rec_seconds = st.sidebar.slider(
    "Recording length (seconds)", 3.0, 20.0,
    float(st.session_state.get("rec_seconds", 8.0)), 0.5,
    key="rec_seconds"
)

vote_n = None
word_len = None
sent_threshold_k = None
sent_debounce_ms = None

if mode == "Single letter (vote)":
    vote_n = st.sidebar.slider(
        "Vote presses (N)", 1, 7,
        int(st.session_state.get("vote_n", 3)), 1,
        key="vote_n"
    )

if mode == "Fixed-length word":
    word_len = st.sidebar.slider(
        "Word length (L)", 2, 10,
        int(st.session_state.get("word_len", 4)), 1,
        key="word_len"
    )

if mode == "Sentence":
    sent_threshold_k = st.sidebar.slider(
        "Sentence sensitivity (threshold K)", 0.8, 2.5,
        float(st.session_state.get("sent_threshold_k", 1.4)), 0.1,
        key="sent_threshold_k"
    )
    sent_debounce_ms = st.sidebar.slider(
        "Sentence debounce (ms)", 20, 150,
        int(st.session_state.get("sent_debounce_ms", 60)), 5,
        key="sent_debounce_ms"
    )

temp = st.sidebar.slider(
    "Decoder temperature", 0.8, 2.5,
    float(st.session_state.get("temp", 1.3)), 0.1,
    key="temp"
)
beam_width = st.sidebar.slider(
    "Beam width", 10, 200,
    int(st.session_state.get("beam_width", 50)), 10,
    key="beam_width"
)
topk_per_pos = st.sidebar.slider(
    "Top-K per position", 4, 15,
    int(st.session_state.get("topk_per_pos", 8)), 1,
    key="topk_per_pos"
)


# -----------------------------
# Utility: record audio (device SR -> model SR)
# -----------------------------
def record_audio(seconds: float, device_index: int, device_sr: int, target_sr: int):
    """
    Record from selected device at device_sr, then resample to target_sr (MODEL_SR).
    Returns (y, sr_used).
    """
    try:
        audio = sd.rec(
            int(seconds * device_sr),
            samplerate=device_sr,
            channels=1,
            dtype="float32",
            device=device_index
        )
        sd.wait()
    except Exception as e:
        raise RuntimeError(
            f"Failed to open microphone stream.\n"
            f"Try a different Input device in the sidebar, or close apps using the mic.\n\n"
            f"Error: {e}"
        )

    y = audio[:, 0].astype(np.float32)

    if int(device_sr) != int(target_sr):
        y = librosa.resample(y, orig_sr=int(device_sr), target_sr=int(target_sr)).astype(np.float32)

    return y, int(target_sr)


def score_vector_for_clip(clip: np.ndarray, sr_used: int) -> np.ndarray:
    feat = kp.extract_mfcc_features_array(clip, sr_used).reshape(1, -1)
    z = scaler.transform(feat)
    df = svm.decision_function(z)

    if df.ndim == 2 and df.shape[0] == 1:
        s = df[0].astype(np.float32)
    elif df.ndim == 1:
        s = df.astype(np.float32)
    else:
        s = None

    if (s is None) or (s.shape[0] != len(svm.classes_)):
        pred = str(svm.predict(z)[0])
        s = np.full((len(svm.classes_),), -10.0, dtype=np.float32)
        pred_i = int(np.where(svm.classes_ == pred)[0][0])
        s[pred_i] = 10.0

    return s


# -----------------------------
# Main action / layout
# -----------------------------
colA, colB = st.columns([1, 1])

with colA:
    st.subheader("Action")
    st.write("Press **Record & Decode** and type during the recording window.")
    do_record = st.button("🎙️ Record & Decode", type="primary")

with colB:
    st.subheader("Model info")
    if kp.os.path.exists(kp.META_OUT):
        try:
            meta = joblib.load(kp.META_OUT)
            st.json({
                "model_version": meta.get("model_version"),
                "trained_at": meta.get("trained_at"),
                "n_samples_total": meta.get("n_samples_total"),
                "eval_accuracy": meta.get("eval_accuracy"),
                "model_sr": MODEL_SR,
            })
        except Exception:
            st.info("Meta file exists but couldn't be read.")
    else:
        st.json({"model_sr": MODEL_SR})

st.divider()


# -----------------------------
# Record + decode
# -----------------------------
if do_record:
    try:
        with st.spinner("Recording..."):
            y, sr_live = record_audio(float(rec_seconds), DEVICE_INDEX, DEVICE_SR, MODEL_SR)
    except Exception as e:
        st.error(str(e))
        st.stop()

    # Segmentation
    if mode == "Sentence":
        try:
            ks_list, centers, thresh, n_peaks = kp.extract_keystrokes_from_signal(
                y, sr_live,
                threshold_k=float(sent_threshold_k),
                debounce_ms=float(sent_debounce_ms)
            )
        except TypeError:
            ks_list, centers, thresh, n_peaks = kp.extract_keystrokes_from_signal(y, sr_live)
    else:
        ks_list, centers, thresh, n_peaks = kp.extract_keystrokes_from_signal(y, sr_live)

    st.write(f"**Segmentation:** peaks_found={n_peaks} | extracted={len(ks_list)} | thresh={thresh:.6f}")

    if len(ks_list) == 0:
        st.error("No taps extracted. Try pressing more distinctly or increase recording length.")
        st.stop()

    # -----------------------------
    # Mode: Single letter (vote)
    # -----------------------------
    if mode == "Single letter (vote)":
        if len(ks_list) < int(vote_n):
            st.warning(f"Extracted {len(ks_list)} taps but need {vote_n}. Try longer recording or clearer presses.")
            st.stop()

        chosen = ks_list[:int(vote_n)]

        preds = []
        for clip in chosen:
            s = score_vector_for_clip(clip, sr_live)
            pred = str(svm.classes_[int(np.argmax(s))])
            preds.append("<SPACE>" if pred == "SPACE" else pred.lower())

        maj = max(set(preds), key=preds.count)
        conf = preds.count(maj) / len(preds)

        st.subheader("Result")
        st.write(f"**Preds:** {preds}")
        st.success(f"**FINAL:** {maj}  (vote_conf={conf:.2f})")

    # -----------------------------
    # Mode: Fixed-length word
    # -----------------------------
    elif mode == "Fixed-length word":
        if len(ks_list) < int(word_len):
            st.warning(f"Extracted {len(ks_list)} taps but need {word_len}. Try slower typing or longer recording.")
            st.stop()

        chosen = ks_list[:int(word_len)]

        per_pos_scores = []
        for clip in chosen:
            s = score_vector_for_clip(clip, sr_live)

            # exclude SPACE from word decode (if model includes it)
            if "SPACE" in svm.classes_:
                space_i = int(np.where(svm.classes_ == "SPACE")[0][0])
                s = s.copy()
                s[space_i] = -1e9

            per_pos_scores.append(s)

        raw, decoded, top_candidates, top_beams = kp.decode_word_beam_autocorrect(
            per_pos_scores, svm.classes_, int(word_len),
            beam_width=int(beam_width), topk_per_pos=int(topk_per_pos), temp=float(temp),
            max_edits=1.0, debug=False
        )

        st.subheader("Word decode")
        st.write(f"**Raw:** `{raw}`")
        if top_beams:
            st.write(f"**Top beam seqs:** {', '.join(top_beams)}")
        st.success(f"**Decoded:** {decoded}")

        if top_candidates:
            st.write("Top candidates:")
            st.write(", ".join([w for w, _ in top_candidates[:10]]))

    # -----------------------------
    # Mode: Sentence
    # -----------------------------
    else:
        per_tap_scores = []
        tap_labels = []

        for clip in ks_list:
            s = score_vector_for_clip(clip, sr_live)
            per_tap_scores.append(s)
            lab = str(svm.classes_[int(np.argmax(s))])
            tap_labels.append(lab)

        disp = []
        for lab in tap_labels:
            disp.append(" " if lab == "SPACE" else lab.lower())

        raw_stream = "".join(disp)
        label_stream = "".join(["<SPACE>" if x == "SPACE" else x for x in tap_labels])

        st.subheader("Sentence (raw)")
        st.code(label_stream)
        st.code(raw_stream)

        # Split into word chunks by SPACE and decode each chunk
        words_raw, words_decoded, words_top = [], [], []
        cur_scores = []

        def decode_chunk(scores):
            L = len(scores)
            if L <= 0:
                return "", "", []
            dyn_edits = min(3.0, max(1.0, 0.35 * L))
            r, d, topc, _ = kp.decode_word_chunk(
                scores, svm.classes_,
                temp=float(temp), beam_width=int(beam_width), topk_per_pos=int(topk_per_pos),
                max_edits=float(dyn_edits), debug=False
            )
            opts = [w for w, _ in topc[:5]] if topc else [d]
            return r, d, opts

        for s, lab in zip(per_tap_scores, tap_labels):
            if lab == "SPACE":
                if cur_scores:
                    r, d, opts = decode_chunk(cur_scores)
                    words_raw.append(r)
                    words_decoded.append(d)
                    words_top.append(opts)
                    cur_scores = []
            else:
                cur_scores.append(s)

        if cur_scores:
            r, d, opts = decode_chunk(cur_scores)
            words_raw.append(r)
            words_decoded.append(d)
            words_top.append(opts)

        decoded_sentence = " ".join([w for w in words_decoded if w])
        raw_words = " ".join([w for w in words_raw if w])

        st.subheader("Sentence (decoded)")
        st.write(f"**Raw words:** `{raw_words}`")
        st.success(decoded_sentence if decoded_sentence else "(empty)")

        if words_top:
            st.write("Top candidates per word:")
            for i, opts in enumerate(words_top, 1):
                st.write(f"word{i}: " + ", ".join(opts))
