# kbd_live_pipeline.py
# Usage:
#   python kbd_live_pipeline.py collect_raw
#   python kbd_live_pipeline.py train
#   python kbd_live_pipeline.py demo
#
# LIVE DEMO + LIVE ADAPTATION (does NOT touch your offline defensible dataset)
# - collect_raw: records a session, segments, saves keystroke windows to datasets_live_current/<LETTER>/keystrokes/
# - train: retrains on the ENTIRE live-current database and prints a stable holdout report
# - demo: predicts from a fresh recording; if wrong and user types the true letter:
#         saves the vote windows into the live-current database, and retrains whole DB every N corrected taps
#
# Key upgrades included:
# - Auto attenuation (keeps headroom if recording is too "hot")
# - Soft-penalty scoring for selecting taps (prefers clean taps, never hard-rejects loud taps)
# - Always-on debug print for chosen tap windows (flushes, unmissable)
# - Commands must be typed fully: "retry" and "quit" (so letters R/Q don’t collide)

import os, sys, time, glob
from pathlib import Path
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import joblib
from scipy.signal import butter, sosfilt, find_peaks
from collections import Counter

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================
# =========================
# PATHS (LIVE CURRENT) — PORTABLE
# =========================
# Default: use the folder containing this script.
# Optional override: set env var PROJECT_DIR to point somewhere else (e.g., your OneDrive folder).
PROJECT_ROOT = Path(os.environ.get("PROJECT_DIR", Path(__file__).resolve().parent)).resolve()

BASE_LIVE = str(PROJECT_ROOT / "datasets_live_current")
MODEL_OUT_DIR = str(PROJECT_ROOT / "models_live_current")
os.makedirs(BASE_LIVE, exist_ok=True)
os.makedirs(MODEL_OUT_DIR, exist_ok=True)

SVM_OUT = os.path.join(MODEL_OUT_DIR, "svm_model_live.joblib")
SCALER_OUT = os.path.join(MODEL_OUT_DIR, "scaler_live.joblib")
META_OUT = os.path.join(MODEL_OUT_DIR, "meta_live.joblib")
CM_OUT = os.path.join(MODEL_OUT_DIR, "confusion_matrix_live.npy")
VERSION_OUT = os.path.join(MODEL_OUT_DIR, "model_version.txt")
EVAL_SPLIT_OUT = os.path.join(MODEL_OUT_DIR, "eval_split_live.joblib")  # stable report split

print(f"[INFO] PROJECT_ROOT={PROJECT_ROOT}", flush=True)
print(f"[INFO] BASE_LIVE={BASE_LIVE}", flush=True)
print(f"[INFO] MODEL_OUT_DIR={MODEL_OUT_DIR}", flush=True)

# AUDIO DEVICE
# =========================
DEVICE = None
CHANNELS = 1

def get_input_sr(device=None):
    dev = sd.query_devices(device, "input")
    return int(round(dev["default_samplerate"]))

SR = get_input_sr(DEVICE)
print(f"[INFO] Using input SR={SR} Hz (device default)")
print(f"[DEBUG] Running: {__file__}", flush=True)

# =========================
# LABELS (A–Z + SPACE)
# =========================
LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["SPACE"]
LABEL_DISPLAY = {**{c: c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}, "SPACE": " "}
LABEL_DISPLAY_LETTER_UI = {**{c: c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}, "SPACE": "<SPACE>"}

# =========================
# LIVE ADAPTIVE TRAINING SETTINGS
# =========================
ADAPTIVE_TRAINING = True
RETRAIN_EVERY_N_CORRECTED_TAPS = 20  # change to 30 later if you want
MAX_RETRAIN_FILES = None            # None = use all files
MIN_TEST_PER_CLASS = 15             # stable holdout support per class (when available)

# =========================
# SEGMENTATION CONFIG (RMS)
# =========================
LOW_FREQ = 100
HIGH_FREQ = 8000
FILTER_ORDER = 4

FRAME_MS = 20
HOP_MS = 10
SMOOTH_WIN = 5

THRESHOLD_K = 2.0
DEBOUNCE_MS = 120

# =========================
# WINDOWING (ONSET-ANCHORED)
# =========================
WINDOW_MS = 420
PRE_MS = 180
POST_MS = WINDOW_MS - PRE_MS

ONSET_FRAC = 0.35
MAX_BACKTRACK_MS = 240
PEAK_SEARCH_AFTER_ONSET_MS = 90

# =========================
# FEATURES
# =========================
N_MFCC = 20
N_FFT = 1024
HOP_LENGTH = 256

# =========================
# DEMO QUALITY/ROBUSTNESS
# =========================
DEBUG_TAP_STATS = True

# auto attenuation (headroom)
TARGET_HEADROOM = 0.85  # only scales down if peak > this

# selection scoring
USE_SOFT_SCORING = True

# =========================
# VERSION HELPERS
# =========================
def read_model_version():
    try:
        with open(VERSION_OUT, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    except:
        return 0

def write_model_version(v: int):
    with open(VERSION_OUT, "w", encoding="utf-8") as f:
        f.write(str(int(v)))

# =========================
# SIGNAL UTILITIES
# =========================
def bandpass_sos(x, sr, low=100, high=8000, order=4):
    nyq = 0.5 * sr
    sos = butter(order, [low/nyq, high/nyq], btype="band", output="sos")
    return sosfilt(sos, x).astype(np.float32)

def moving_average(x, w):
    return np.convolve(x, np.ones(w) / w, mode="same")

def rms_energy(x):
    x = np.asarray(x, dtype=np.float32)
    return float(np.sqrt(np.mean(x * x) + 1e-12))

def crest_factor(x):
    x = np.asarray(x, dtype=np.float32)
    rms = float(np.sqrt(np.mean(x * x) + 1e-12))
    peak = float(np.max(np.abs(x)) + 1e-12)
    return peak / (rms + 1e-12), rms, peak

def peak_pos_ratio(x):
    x = np.asarray(x, dtype=np.float32)
    i = int(np.argmax(np.abs(x)))
    return i / max(1, (len(x) - 1))

def clipped_fraction(x, thr=0.98):
    x = np.asarray(x, dtype=np.float32)
    return float(np.mean(np.abs(x) >= thr))

def auto_rms_floor(rms_vals):
    rms_vals = np.asarray(rms_vals, dtype=np.float32)
    med = float(np.median(rms_vals))
    p40 = float(np.quantile(rms_vals, 0.40))
    min_floor = 1e-5
    floor = 0.60 * med
    floor = min(floor, p40)
    floor = max(floor, min_floor)
    return floor

def auto_attenuate(y, target=TARGET_HEADROOM):
    """
    Scale signal down if peak is too high, to keep headroom.
    NOTE: cannot "unclip" audio already clipped by hardware/OS.
    """
    y = np.asarray(y, dtype=np.float32)
    peak = float(np.max(np.abs(y)) + 1e-12)
    if peak > target:
        gain = target / peak
        y = (y * gain).astype(np.float32)
        return y, gain, peak
    return y, 1.0, peak

def soft_quality_score(k):
    """
    Prefer clean transient taps but never hard-reject loud ones.
    Score = (crest*rms) * (1 - soft_penalty)
    """
    cf, rms, peak = crest_factor(k)
    clip_pct = clipped_fraction(k, thr=0.98) * 100.0

    base = cf * rms

    p = 0.0
    if peak > 0.95:
        p += (peak - 0.95) / 0.05   # 0..1
    if clip_pct > 0.01:
        p += min(1.0, (clip_pct - 0.01) / 0.05)

    penalty = min(0.85, 0.6 * p)   # cap penalty
    return float(base * (1.0 - penalty))

def is_valid_keystroke_window(x, sr, rms_floor):
    """
    Demo-friendly: only reject obvious junk.
    (Do NOT over-reject; user experience matters.)
    """
    x = np.asarray(x, dtype=np.float32)
    cf, rms, peak = crest_factor(x)
    ppos = peak_pos_ratio(x)

    if rms < rms_floor:
        return False, "quiet"
    if ppos < 0.02 or ppos > 0.98:
        return False, "edge-peak"
    # allow hot peaks; we handle that via soft scoring + attenuation
    return True, "ok"

# =========================
# SEGMENT + ONSET-ANCHORED EXTRACT
# =========================
def extract_keystrokes_from_signal(y, sr, threshold_k=None, debounce_ms=None):
    y = np.asarray(y, dtype=np.float32)

    # filter for RMS segmentation only
    y_filt = bandpass_sos(y, sr, LOW_FREQ, HIGH_FREQ, FILTER_ORDER)

    frame_len = int(sr * FRAME_MS / 1000)
    hop_len = int(sr * HOP_MS / 1000)

    rms = librosa.feature.rms(y=y_filt, frame_length=frame_len, hop_length=hop_len)[0]
    rms_s = moving_average(rms, SMOOTH_WIN)

    k = THRESHOLD_K if threshold_k is None else float(threshold_k)
    db = DEBOUNCE_MS if debounce_ms is None else float(debounce_ms)

    thresh = float(np.mean(rms_s) + k * np.std(rms_s))
    min_dist_frames = int(db / HOP_MS)

    peaks, props = find_peaks(rms_s, height=thresh, distance=min_dist_frames)

    win_samp = int(sr * WINDOW_MS / 1000)
    pre_samp = int(sr * PRE_MS / 1000)
    post_samp = win_samp - pre_samp

    max_backtrack_frames = int(MAX_BACKTRACK_MS / HOP_MS)
    peak_search_after = int(sr * PEAK_SEARCH_AFTER_ONSET_MS / 1000)

    base_mu = float(np.median(rms_s))
    base_sd = float(np.std(rms_s))
    base_onset = base_mu + 1.5 * base_sd

    keystrokes = []
    centers = []

    for p_i, p in enumerate(peaks):
        peak_val = float(props["peak_heights"][p_i])
        onset_thr = max(base_onset, ONSET_FRAC * peak_val)

        j = p
        stop = max(0, p - max_backtrack_frames)
        while j > stop and rms_s[j] > onset_thr:
            j -= 1

        onset_sample = int(j * hop_len)

        lo = onset_sample
        hi = min(len(y), onset_sample + peak_search_after)

        # peak search on filtered helps find the actual impact peak
        if hi > lo + 2:
            peak_sample = lo + int(np.argmax(np.abs(y_filt[lo:hi])))
        else:
            peak_sample = onset_sample

        start = peak_sample - pre_samp
        end = peak_sample + post_samp
        if start < 0 or end > len(y):
            continue

        keystrokes.append(y[start:end].astype(np.float32))
        centers.append(peak_sample)

    return keystrokes, centers, thresh, len(peaks)

# =========================
# MFCC FEATURES
# =========================
def extract_mfcc_features_array(y, sr):
    y = np.asarray(y, dtype=np.float32)
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)
    feats = np.vstack([mfcc, d1, d2])
    return np.concatenate([feats.mean(axis=1), feats.std(axis=1)]).astype(np.float32)

# =========================
# DATASET HELPERS
# =========================
def ensure_letter_dir(letter):
    ks_dir = os.path.join(BASE_LIVE, letter, "keystrokes")
    os.makedirs(ks_dir, exist_ok=True)
    return ks_dir

def next_index(ks_dir, prefix):
    files = glob.glob(os.path.join(ks_dir, f"{prefix}_*.wav"))
    mx = 0
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        try:
            num = int(stem.split("_")[-1])
            mx = max(mx, num)
        except:
            pass
    return mx + 1

def label_from_path(wav_path):
    p = os.path.normpath(wav_path)
    parent = os.path.basename(os.path.dirname(p))
    if parent.lower() == 'keystrokes':
        return os.path.basename(os.path.dirname(os.path.dirname(p)))
    return parent

def parse_group_from_filename(wav_path):
    """
    Correct grouping for your patterns:
      LETTER_session_YYYYMMDD_HHMMSS_00001.wav -> LETTER_session_YYYYMMDD_HHMMSS
      LETTER_ADAPT_YYYYMMDD_HHMMSS_00001.wav   -> LETTER_ADAPT_YYYYMMDD_HHMMSS
    """
    base = os.path.splitext(os.path.basename(wav_path))[0]
    parts = base.split("_")
    if len(parts) >= 5 and parts[1].lower() == "session":
        return "_".join(parts[:4])
    if len(parts) >= 5 and parts[1].lower() == "adapt":
        return "_".join(parts[:4])
    if len(parts) >= 2:
        return "_".join(parts[:-1])
    return base

def save_demo_keystrokes(letter, clips, prefix):
    ks_dir = ensure_letter_dir(letter)
    idx = next_index(ks_dir, prefix)
    saved = 0
    for clip in clips:
        out = os.path.join(ks_dir, f"{prefix}_{idx:05d}.wav")
        sf.write(out, clip, SR, subtype="PCM_16")
        idx += 1
        saved += 1
    return saved

# =========================
# LOAD FULL LIVE DATASET
# =========================
def list_all_keystroke_files(max_files=None):
    """Robust scan:
    - preferred: datasets_live_current/<LABEL>/keystrokes/*.wav
    - fallback:  datasets_live_current/<LABEL>/*.wav
    """
    files = []
    per_letter = {}
    for L in LABELS:
        fL = []

        ks_dir = os.path.join(BASE_LIVE, L, "keystrokes")
        if os.path.isdir(ks_dir):
            fL.extend(glob.glob(os.path.join(ks_dir, "*.wav")))

        # fallback layout (user forgot 'keystrokes' folder)
        label_dir = os.path.join(BASE_LIVE, L)
        if os.path.isdir(label_dir):
            fL.extend(glob.glob(os.path.join(label_dir, "*.wav")))

        if not fL:
            continue

        per_letter[L] = len(fL)
        files.extend(fL)

    if not files:
        return [], per_letter

    if max_files is not None and len(files) > max_files:
        rng = np.random.RandomState(0)
        files = list(rng.choice(files, size=max_files, replace=False))
        # recompute per-letter counts after downsampling
        per_letter = {}
        for f in files:
            lab = label_from_path(f)
            per_letter[lab] = per_letter.get(lab, 0) + 1

    return files, per_letter

# =========================
# STABLE HOLDOUT SPLIT
# =========================
def build_or_load_eval_split(files, y, groups, min_test_per_class=15, seed=42):
    """Creates or loads a stable holdout split for reporting.
    If the saved split references paths that don't exist on this machine,
    it auto-rebuilds a fresh split.
    """
    if os.path.exists(EVAL_SPLIT_OUT):
        try:
            split = joblib.load(EVAL_SPLIT_OUT)
            test_files = [f for f in split.get("test_files", []) if os.path.exists(f)]
            train_files = [f for f in split.get("train_files", []) if os.path.exists(f)]

            # If split became invalid on a new machine/path, rebuild
            if len(train_files) == 0 or len(test_files) == 0:
                print("[WARN] Saved eval split is empty/invalid here. Rebuilding split...", flush=True)
                try:
                    os.remove(EVAL_SPLIT_OUT)
                except:
                    pass
            else:
                return {"train_files": train_files, "test_files": test_files, "created_at": split.get("created_at", "?")}
        except Exception as e:
            print(f"[WARN] Failed to load eval split ({e}). Rebuilding...", flush=True)
            try:
                os.remove(EVAL_SPLIT_OUT)
            except:
                pass

    rng = np.random.RandomState(seed)

    files = np.asarray(files, dtype=object)
    y = np.asarray(y, dtype=str)
    groups = np.asarray(groups, dtype=str)

    test_mask = np.zeros(len(files), dtype=bool)

    for L in LABELS:
        idx_L = np.where(y == L)[0]
        if len(idx_L) == 0:
            continue

        groups_L = groups[idx_L]
        uniq = np.unique(groups_L)
        rng.shuffle(uniq)

        picked = 0
        for g in uniq:
            if picked >= min_test_per_class:
                break
            g_idx = idx_L[groups_L == g]
            test_mask[g_idx] = True
            picked += len(g_idx)

    test_files = files[test_mask].tolist()
    train_files = files[~test_mask].tolist()

    # Last-resort safety: if test or train ended up empty (tiny dataset), do a simple split.
    if len(train_files) == 0 or len(test_files) == 0:
        uniq_groups = np.unique(groups)
        rng.shuffle(uniq_groups)
        n_test_groups = max(1, int(round(0.2 * len(uniq_groups))))
        test_groups = set(uniq_groups[:n_test_groups])
        test_mask = np.array([g in test_groups for g in groups], dtype=bool)
        test_files = files[test_mask].tolist()
        train_files = files[~test_mask].tolist()

    split = {
        "train_files": train_files,
        "test_files": test_files,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "min_test_per_class": min_test_per_class,
        "seed": seed,
    }
    joblib.dump(split, EVAL_SPLIT_OUT)
    return split
# RETRAIN ON FULL DATA + REPORT
# =========================
def retrain_full_and_report(reason="manual", verbose=True):
    files_all, per_letter = list_all_keystroke_files(max_files=MAX_RETRAIN_FILES)
    if not files_all:
        return None, None

    X_all, y_all, groups_all = load_features_labels_groups(files_all)
    total = int(len(y_all))
    uniq_groups = int(len(np.unique(groups_all)))

    split = build_or_load_eval_split(files_all, y_all, groups_all,
                                     min_test_per_class=MIN_TEST_PER_CLASS, seed=42)
    train_files = split["train_files"]
    test_files = split["test_files"]

    Xtr, ytr, _ = load_features_labels_groups(train_files)
    Xte, yte, _ = load_features_labels_groups(test_files)


    if Xtr.shape[0] == 0 or Xte.shape[0] == 0:
        # Usually happens if an old eval split referenced paths from another machine.
        # Rebuild split once and retry.
        print("[WARN] Empty train/test split detected. Rebuilding eval split and retrying...", flush=True)
        try:
            os.remove(EVAL_SPLIT_OUT)
        except:
            pass
        split = build_or_load_eval_split(files_all, y_all, groups_all,
                                         min_test_per_class=max(1, MIN_TEST_PER_CLASS), seed=42)
        train_files = split["train_files"]
        test_files = split["test_files"]
        Xtr, ytr, _ = load_features_labels_groups(train_files)
        Xte, yte, _ = load_features_labels_groups(test_files)
        if Xtr.shape[0] == 0 or Xte.shape[0] == 0:
            raise SystemExit("Training failed: no usable train/test samples found in datasets_live_current/.")

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    svm = SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced",
          decision_function_shape="ovr")
    svm.fit(Xtr_s, ytr)

    yp = svm.predict(Xte_s)
    acc = float(accuracy_score(yte, yp))
    rep = classification_report(yte, yp, labels=LABELS, zero_division=0)
    cm = confusion_matrix(yte, yp, labels=LABELS)

    v = read_model_version() + 1
    write_model_version(v)

    meta = {
        "model_version": v,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "reason": reason,
        "n_samples_total": total,
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "unique_groups_total": uniq_groups,
        "per_letter_counts_total": {L: int(per_letter.get(L, 0)) for L in LABELS},
        "eval_split_created_at": split.get("created_at", "?"),
        "eval_min_test_per_class": MIN_TEST_PER_CLASS,
        "eval_accuracy": acc,
    }

    joblib.dump(svm, SVM_OUT)
    joblib.dump(scaler, SCALER_OUT)
    joblib.dump(meta, META_OUT)
    np.save(CM_OUT, cm)

    if verbose:
        print("\n" + "=" * 74)
        print(f"[RETRAIN] Updated live-current model -> v{v} (reason: {reason})")
        print(f"  Total samples: {total} | Unique groups: {uniq_groups}")
        nz = [(L, per_letter.get(L, 0)) for L in LABELS if per_letter.get(L, 0)]
        if nz:
            print("  Per-letter counts:", " ".join([f"{L}:{c}" for L, c in nz]))
        print(f"  Report set: stable holdout (created_at={split.get('created_at')}, min_test_per_class={MIN_TEST_PER_CLASS})")
        print(f"\n  Held-out accuracy: {acc:.4f}")
        print(rep)
        print("=" * 74 + "\n")

    return svm, scaler

# =========================
# CONFUSION-AWARE SUBSTITUTIONS (derived from your runs)
# =========================
CONFUSION_COST = {
    # Tier 1: dominant confusions (very frequent)
    ("h", "n"): 0.20, ("n", "h"): 0.20,
    ("x", "z"): 0.20, ("z", "x"): 0.20,
    ("g", "v"): 0.20, ("v", "g"): 0.20,
    ("c", "z"): 0.20, ("z", "c"): 0.20,
    ("l", "f"): 0.20, ("f", "l"): 0.40,   # f->l less evidenced but allow

    # M cluster (dominant)
    ("m", "h"): 0.20, ("h", "m"): 0.40,
    ("m", "n"): 0.30, ("n", "m"): 0.50,

    # Tier 2: observed but weaker (from alphabet sweep)
    ("b", "k"): 0.40, ("k", "b"): 0.60,
    ("f", "k"): 0.40, ("k", "f"): 0.60,
    ("i", "p"): 0.40, ("p", "i"): 0.50,
    ("k", "i"): 0.40, ("i", "k"): 0.60,
    ("p", "j"): 0.40, ("j", "p"): 0.60,
    ("r", "i"): 0.40, ("i", "r"): 0.60,
    ("t", "g"): 0.40, ("g", "t"): 0.50,
    ("u", "r"): 0.40, ("r", "u"): 0.60,
    ("u", "p"): 0.40, ("p", "u"): 0.60,
    ("w", "a"): 0.40, ("a", "w"): 0.60,
    ("w", "i"): 0.40, ("i", "w"): 0.60,
    ("a", "z"): 0.50, ("z", "a"): 0.70,
    ("s", "f"): 0.40, ("f", "s"): 0.60,
    ("s", "d"): 0.40, ("d", "s"): 0.60,
    ("e", "t"): 0.40, ("t", "e"): 0.60,
    ("e", "w"): 0.40, ("w", "e"): 0.60,
    ("y", "g"): 0.40, ("g", "y"): 0.60,
    ("y", "t"): 0.40, ("t", "y"): 0.60,
}

def weighted_substitution_cost(a: str, b: str) -> float:
    """Cost to substitute observed letter a -> true letter b."""
    if a == b:
        return 0.0
    return float(CONFUSION_COST.get((a, b), 1.0))

def seq_word_cost(seq: str, word: str):
    """
    seq: beam hypothesis (observed)
    word: dictionary candidate (true)
    Returns: (total_cost, n_exact, n_confused, n_other)
    """
    total = 0.0
    n_exact = n_conf = n_other = 0
    for a, b in zip(seq, word):
        if a == b:
            n_exact += 1
            continue
        c = weighted_substitution_cost(a, b)
        total += c
        if c < 1.0:
            n_conf += 1
        else:
            n_other += 1
    return total, n_exact, n_conf, n_other

# =========================
# MODE: collect_raw
# =========================
def mode_collect_raw():
    print("\nCOLLECT_RAW mode (record -> segment -> save)  [LIVE CURRENT]")
    rec_seconds = float(input("Record duration seconds per session (suggest 12-20): ").strip() or "15")
    target_min = int(input("Minimum keystrokes expected (suggest 15-25): ").strip() or "15")

    raw_dir = os.path.join(BASE_LIVE, "_raw_sessions")
    os.makedirs(raw_dir, exist_ok=True)

    while True:

        letter = input("\nLabel to collect (A-Z or SPACE) or 'done': ").strip().upper()
        if letter in ("DONE", "QUIT", "EXIT"):
            break

        # normalize common ways user might type it
        if letter in (" ", "_"):
            letter = "SPACE"

        if letter not in LABELS:
            print("Enter A-Z or SPACE.")
            continue

        session = time.strftime("session_%Y%m%d_%H%M%S")
        raw_path = os.path.join(raw_dir, f"{letter}_{session}.wav")

        print(f"\nRecording {rec_seconds:.1f}s for {letter}. Start pressing NOW...")
        audio = sd.rec(int(rec_seconds * SR), samplerate=SR, channels=1, dtype="float32", device=DEVICE)
        sd.wait()

        ysig = audio[:, 0].astype(np.float32)

        # Auto attenuation (keeps collection consistent with demo)
        ysig, gain, raw_peak = auto_attenuate(ysig, target=TARGET_HEADROOM)
        if gain != 1.0:
            print(f"[INFO] Applied attenuation: x{gain:.3f} (raw_peak={raw_peak:.3f})")

        sf.write(raw_path, ysig, SR, subtype="PCM_16")
        print("Saved raw:", raw_path)

        ks, _, thresh, n_peaks = extract_keystrokes_from_signal(ysig, SR)
        print(f"Segmentation: peaks_found={n_peaks}, extracted={len(ks)}, thresh={thresh:.6f}")

        if len(ks) < target_min:
            print("⚠️ Too few detected. If needed: THRESHOLD_K 2.0 -> 1.6.\n")

        ks_dir = ensure_letter_dir(letter)
        prefix = f"{letter}_{session}"
        idx = next_index(ks_dir, prefix)

        for clip in ks:
            out = os.path.join(ks_dir, f"{prefix}_{idx:05d}.wav")
            sf.write(out, clip, SR, subtype="PCM_16")
            idx += 1

        print(f"✅ Saved {len(ks)} keystrokes to: {ks_dir}")

    print("\n✅ Finished collect_raw.")

# =========================
# MODE: train
# =========================
def mode_train():
    print("\nTRAIN mode  [LIVE CURRENT] (full retrain + stable holdout report)")
    svm, scaler = retrain_full_and_report(reason="manual train", verbose=True)
    if svm is None:
        raise SystemExit("No live-current training data found.")

# =========================
# WORD DECODING (fixed length)
# =========================
WORDLIST_PATH = str(PROJECT_ROOT / "wordlist_common_clean.txt")

FALLBACK_WORDS = {
    3: ["the", "and", "you", "not", "for", "are", "was", "but", "all"],
    4: ["that", "have", "with", "this", "from", "they", "will", "what", "when"],
    5: ["there", "which", "about", "would", "these", "their", "other", "after", "first", "never"],
    6: ["before", "should", "people", "little", "system", "public", "during", "within"],
}

_WORDS_BY_LEN = None  # cache


def _load_wordlist():
    """
    Returns dict: length -> list of lowercase words.
    Caches result in _WORDS_BY_LEN.
    """
    global _WORDS_BY_LEN
    if _WORDS_BY_LEN is not None:
        return _WORDS_BY_LEN

    words_by_len = {}

    if WORDLIST_PATH and os.path.exists(WORDLIST_PATH):
        try:
            with open(WORDLIST_PATH, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    w = line.strip().lower()
                    if not w.isalpha():
                        continue
                    words_by_len.setdefault(len(w), []).append(w)
        except Exception as e:
            print(f"[WARN] Failed reading wordlist: {e}. Using fallback list.")
            words_by_len = {}

    if not words_by_len:
        for L, ws in FALLBACK_WORDS.items():
            words_by_len[L] = [w.lower() for w in ws if w.isalpha() and len(w) == L]

    _WORDS_BY_LEN = words_by_len
    return _WORDS_BY_LEN


def _softmax(x, temp=1.0):
    x = np.asarray(x, dtype=np.float64)
    t = max(1e-9, float(temp))
    x = (x / t) - np.max(x)
    ex = np.exp(x)
    return ex / (np.sum(ex) + 1e-12)


def probs_from_scores(scores, temp=1.3):
    return _softmax(scores, temp=temp).astype(np.float32)


def beam_search_letters(per_pos_scores, classes, beam_width=50, topk_per_pos=8, temp=1.3):
    """
    Beam search over letter sequences.
    NOTE: For word mode, we expect classes to be A-Z only (SPACE excluded upstream).
    Returns list of (seq_str, logp) best-first.
    """
    cls = [str(c).lower() for c in classes]  # e.g. ['a','b',...,'z']

    beams = [("", 0.0)]
    for scores in per_pos_scores:
        p = probs_from_scores(scores, temp=temp)
        topi = np.argsort(p)[-topk_per_pos:][::-1]

        new_beams = []
        for seq, lp in beams:
            for j in topi:
                new_seq = seq + cls[j]
                new_lp = lp + float(np.log(p[j] + 1e-12))
                new_beams.append((new_seq, new_lp))

        new_beams.sort(key=lambda t: t[1], reverse=True)
        beams = new_beams[:beam_width]

    return beams


def decode_word_beam_autocorrect(
    per_pos_scores,
    classes,
    word_len,
    beam_width=50,
    topk_per_pos=8,
    temp=1.3,
    max_edits=1.0,
    edit_penalty=0.15,
    max_candidates=10,
    debug=True,
):
    """
    Beam -> dictionary snap using weighted confusion costs (seq_word_cost).
    Returns: raw, decoded, top_candidates, top_beams
    """
    words_by_len = _load_wordlist()
    cand_words = words_by_len.get(int(word_len), [])
    cand_set = set(cand_words)

    raw = "".join([str(classes[int(np.argmax(s))]).lower() for s in per_pos_scores])

    beams = beam_search_letters(
        per_pos_scores, classes,
        beam_width=beam_width,
        topk_per_pos=topk_per_pos,
        temp=temp
    )
    top_beams = [seq for seq, _ in beams[:5]]

    if raw in cand_set:
        return raw, raw, [(raw, 0.0)], top_beams

    cand_words = [
        w.strip().lower()
        for w in cand_words
        if w and w.strip().isalpha() and len(w.strip()) == word_len
    ]

    best_word = raw
    best_score = -1e18
    scored_words = {}

    for seq, lp in beams:
        for w in cand_words:
            cost, n_exact, n_conf, n_other = seq_word_cost(seq, w)

            if cost <= float(max_edits):
                sc = lp - edit_penalty * cost
                if (w not in scored_words) or (sc > scored_words[w]):
                    scored_words[w] = sc
                if sc > best_score:
                    best_score = sc
                    best_word = w

    if not scored_words:
        if debug:
            print(f"[DEBUG] No dictionary candidate within weighted cost <= {max_edits:.2f}; returning raw.", flush=True)
        return raw, raw, [(raw, 0.0)], top_beams

    top = sorted(scored_words.items(), key=lambda t: t[1], reverse=True)[:max_candidates]
    top_candidates = [(w, float(sc)) for w, sc in top]

    if debug:
        best_seq = None
        best_map_cost = 1e9
        for seq, _ in beams[:min(len(beams), 50)]:
            c, ne, nc, no = seq_word_cost(seq, best_word)
            if c < best_map_cost:
                best_map_cost = c
                best_seq = seq
        if best_seq is not None:
            c, ne, nc, no = seq_word_cost(best_seq, best_word)
            print(f"[DEBUG] best_seq='{best_seq}' -> decoded='{best_word}' cost={c:.2f} (exact={ne}, confusable={nc}, other={no})", flush=True)

    return raw, best_word, top_candidates, top_beams

def _strip_space_class(per_pos_scores, classes):
    """
    Remove SPACE dimension from per_pos_scores and classes.
    Prevents beam search from creating multi-character tokens like 'space'.
    Returns: (new_scores_list, new_classes_array)
    """
    classes = np.asarray(classes)
    if "SPACE" not in classes:
        return per_pos_scores, classes

    space_i = int(np.where(classes == "SPACE")[0][0])
    new_classes = np.delete(classes, space_i)

    new_scores = []
    for s in per_pos_scores:
        s = np.asarray(s, dtype=np.float32)
        new_scores.append(np.delete(s, space_i))
    return new_scores, new_classes


def decode_word_chunk(per_pos_scores, classes, temp=1.3, beam_width=50, topk_per_pos=8, max_edits=1.0, debug=False):
    """
    Decode a single word chunk (no spaces inside), using your beam + confusion-aware decoder.
    Automatically strips SPACE class before decoding.
    Returns: raw_letters, decoded_word, top_candidates, top_beams
    """

    L = len(per_pos_scores)
    if L <= 2:
        best, top = decode_short_word(per_pos_scores, classes, temp=temp, debug=debug)
        if best is not None:
            # raw from top-1 (letters only)
            scores2, classes2 = _strip_space_class(per_pos_scores, classes)
            cls = [str(c).lower() for c in classes2]
            raw = "".join([cls[int(np.argmax(s))] for s in scores2])
            return raw, best, top, []

    scores2, classes2 = _strip_space_class(per_pos_scores, classes)
    raw, decoded, top_candidates, top_beams = decode_word_beam_autocorrect(
        scores2,
        classes2,
        L,
        beam_width=beam_width,
        topk_per_pos=topk_per_pos,
        temp=temp,
        max_edits=max_edits,
        debug=debug
    )
    return raw, decoded, top_candidates, top_beams

SHORT_WORDS_1 = ["a", "i"]
SHORT_WORDS_2 = [
    "of","to","in","is","it","at","on","as","an","or","we","me","my","no","up","do","go","so","he","be","by"
]

def decode_short_word(per_pos_scores, classes, temp=1.3, confusion_weight=0.7, debug=False):
    """
    Specialized decoder for word length 1 or 2.
    Only considers extremely common short words and snaps aggressively.
    """
    L = len(per_pos_scores)
    if L == 1:
        cand = SHORT_WORDS_1
    elif L == 2:
        cand = SHORT_WORDS_2
    else:
        return None, []  # not applicable

    # strip SPACE if present
    scores2, classes2 = _strip_space_class(per_pos_scores, classes)
    cls = [str(c).lower() for c in classes2]
    idx = {ch: i for i, ch in enumerate(cls)}

    # raw top1 string
    raw = "".join([cls[int(np.argmax(s))] for s in scores2])

    # precompute log-probs per position over letters
    pos_logp = []
    for s in scores2:
        p = probs_from_scores(s, temp=temp)
        pos_logp.append(np.log(p + 1e-12))

    best_w = raw
    best_sc = -1e18
    scored = []

    for w in cand:
        if len(w) != L:
            continue

        # acoustic score
        sc = 0.0
        ok = True
        for i, ch in enumerate(w):
            j = idx.get(ch, None)
            if j is None:
                ok = False
                break
            sc += float(pos_logp[i][j])
        if not ok:
            continue

        # confusion-aware penalty (use your existing seq_word_cost)
        cost, ne, nc, no = seq_word_cost(raw, w)
        sc -= confusion_weight * cost

        scored.append((w, sc, cost))

        if sc > best_sc:
            best_sc = sc
            best_w = w

    scored.sort(key=lambda t: t[1], reverse=True)
    top = [(w, sc) for (w, sc, cost) in scored[:5]]

    if debug:
        print(f"[DEBUG] shortword raw='{raw}' -> best='{best_w}' top={top}", flush=True)

    return best_w, top

# =========================
# MODE: demo (WORD MODE + single-letter + adaptive)
# =========================
def mode_demo():
    print("\nDEMO mode  [LIVE CURRENT]")
    print("Options:")
    print("  1) single-letter (press same key N times)  [supports adaptive retrain]")
    print("  2) fixed-length word (type a word of length L)")
    print("  3) sentence (type a short sentence with spaces)\n")

    if not (os.path.exists(SVM_OUT) and os.path.exists(SCALER_OUT)):
        raise SystemExit("Model not found. Run: python kbd_live_pipeline.py train")

    svm = joblib.load(SVM_OUT)
    scaler = joblib.load(SCALER_OUT)

    # show model meta if present
    if os.path.exists(META_OUT):
        try:
            meta = joblib.load(META_OUT)
            print(f"[INFO] Loaded model v{meta.get('model_version','?')} trained_at={meta.get('trained_at','?')}", flush=True)
        except:
            pass

    mode = (input("Choose mode (1=letter, 2=word, 3=sentence) [3]: ").strip() or "3").strip()

    # ============================================================
    # MODE 1: single-letter (keeps your debug + adaptive retrain)
    # ============================================================
    if mode == "1":
        vote_n = int(input("How many presses to vote on? (suggest 3): ").strip() or "3")
        rec_seconds = float(input("Recording length seconds (suggest 6-10): ").strip() or "8")

        print("\nInstructions:")
        print(f"- Press Enter to record for {rec_seconds:.1f}s.")
        print(f"- During recording, press the SAME key about {vote_n} times.")
        print("- Type commands fully: 'retry' or 'quit' (so letters R/Q don't collide).")
        print("- If wrong, type the TRUE letter A-Z to append + retrain.\n")

        corrected_taps_since_retrain = 0

        while True:
            cmd = input("Press Enter to record (or type 'retry' / 'quit'): ").strip().lower()
            if cmd == "quit":
                return
            if cmd == "retry":
                print("\nRetrying...\n", flush=True)
                continue
            if cmd != "":
                print("Type only: retry / quit / or press Enter.\n", flush=True)
                continue

            print(f"\nRecording {rec_seconds:.1f}s... start pressing now!", flush=True)
            audio = sd.rec(int(rec_seconds * SR), samplerate=SR, channels=1, dtype="float32", device=DEVICE)
            sd.wait()
            ysig = audio[:, 0].astype(np.float32)

            ks_list, centers, thresh, n_peaks = extract_keystrokes_from_signal(ysig, SR)
            print(f"Segmentation: peaks_found={n_peaks}, extracted={len(ks_list)}, thresh={thresh:.6f}", flush=True)

            if len(ks_list) < vote_n:
                print("❌ Not enough keystrokes extracted. Try longer recording.\n", flush=True)
                continue

            # For now: use first N taps in time order (stable)
            chosen = ks_list[:vote_n]

            # =========================
            # DEBUG (ALWAYS PRINTS)
            # =========================
            if DEBUG_TAP_STATS:
                print("\n" + "=" * 68, flush=True)
                print("[DEBUG] Chosen window stats (tap windows used for prediction):", flush=True)
                for i, clip in enumerate(chosen, 1):
                    cf, rms, peak = crest_factor(clip)
                    ppos = peak_pos_ratio(clip)
                    clip_pct = clipped_fraction(clip, thr=0.98) * 100.0
                    print(
                        f"[DEBUG] tap#{i}: peak_pos={ppos:.3f}  crest={cf:.2f}  "
                        f"rms={rms:.6f}  peak={peak:.3f}  clip%={clip_pct:.3f}",
                        flush=True
                    )
                print("=" * 68 + "\n", flush=True)
                sys.stdout.flush()

            # Predict
            preds = []
            for clip in chosen:
                feat = extract_mfcc_features_array(clip, SR).reshape(1, -1)
                z = scaler.transform(feat)
                pred = str(svm.predict(z)[0])
                preds.append(LABEL_DISPLAY_LETTER_UI.get(pred, pred))

            counts = Counter(preds)
            maj, maj_count = counts.most_common(1)[0]
            conf = maj_count / vote_n

            print("\n=== RESULT ===", flush=True)
            print(f"Preds={preds} -> FINAL={maj} (vote_conf={conf:.2f})", flush=True)

            print("\nFeedback:", flush=True)
            print("  Enter  = correct", flush=True)
            print("  A-Z    = true letter (if wrong / append taps)", flush=True)
            print("  retry  = retry", flush=True)
            print("  quit   = quit", flush=True)

            raw = input("Your input: ").strip()
            cmd2 = raw.lower()

            if cmd2 == "quit":
                return
            if cmd2 == "retry":
                print("\nRetrying...\n", flush=True)
                continue

            fb = raw.upper()

            if fb == "":
                print("✅ Marked correct.\n", flush=True)

            elif len(fb) == 1 and fb in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                true_letter = fb
                print(f"❌ Marked true letter = {true_letter}. Appending these taps to training...", flush=True)

                if ADAPTIVE_TRAINING:
                    adapt_stamp = time.strftime("%Y%m%d_%H%M%S")
                    prefix = f"{true_letter}_ADAPT_{adapt_stamp}"
                    n_saved = save_demo_keystrokes(true_letter, chosen, prefix)
                    corrected_taps_since_retrain += n_saved

                    print(f"   Saved {n_saved} tap(s). Corrected taps since retrain: "
                          f"{corrected_taps_since_retrain}/{RETRAIN_EVERY_N_CORRECTED_TAPS}",
                          flush=True)

                    if corrected_taps_since_retrain >= RETRAIN_EVERY_N_CORRECTED_TAPS:
                        print("   Retraining on the FULL live dataset now (stable holdout report)...", flush=True)
                        svm2, scaler2 = retrain_full_and_report(
                            reason=f"live corrections added: {corrected_taps_since_retrain}",
                            verbose=True
                        )
                        if svm2 is not None:
                            svm, scaler = svm2, scaler2
                            print("   ✅ Model hot-swapped in memory for next attempt.\n", flush=True)
                        corrected_taps_since_retrain = 0

                print("", flush=True)

            else:
                print("Invalid input.\n", flush=True)

            again = input("Do another attempt? (Enter=yes / type 'quit' to stop): ").strip().lower()
            if again == "quit":
                return
            print("", flush=True)

    elif mode == "3":
        # ============================================================
        # MODE 3: sentence (stream -> words via SPACE -> decode each word)
        # ============================================================
        print("\nSENTENCE mode (type a short sentence incl. spaces)")

        rec_seconds = float(input("Recording length seconds (suggest 8-15): ").strip() or "10")

        # Decoder knobs
        temp = 1.3
        beam_width = 50
        topk_per_pos = 8

        # Sentence segmentation knobs (more recall)
        sent_threshold_k = 1.4
        sent_debounce_ms = 60

        print("\nInstructions:")
        print(f"- Press Enter to record for {rec_seconds:.1f}s.")
        print("- Type a short sentence with spaces (normal typing).")
        print("- Commands must be typed fully: 'retry' or 'quit'.\n")

        while True:
            cmd = input("Press Enter to record (or type 'retry' / 'quit'): ").strip().lower()
            if cmd == "quit":
                return
            if cmd == "retry":
                print("\nRetrying...\n")
                continue
            if cmd != "":
                print("Type only: retry / quit / or press Enter.\n")
                continue

            print(f"\nRecording {rec_seconds:.1f}s... start typing now!")
            audio = sd.rec(int(rec_seconds * SR), samplerate=SR, channels=1, dtype="float32", device=DEVICE)
            sd.wait()
            ysig = audio[:, 0].astype(np.float32)

            # Use sentence-tuned segmentation (more sensitive than Mode 1/2)
            ks_list, centers, thresh, n_peaks = extract_keystrokes_from_signal(
                ysig, SR,
                threshold_k=sent_threshold_k,
                debounce_ms=sent_debounce_ms
            )
            print(f"Segmentation: peaks_found={n_peaks}, extracted={len(ks_list)}, thresh={thresh:.6f}")

            if len(ks_list) == 0:
                print("❌ No taps extracted. Try slower typing / longer recording / increase sensitivity.\n")
                continue

            # Convert each tap to decision scores + predicted label
            per_tap_scores = []
            tap_labels = []
            for clip in ks_list:
                feat = extract_mfcc_features_array(clip, SR).reshape(1, -1)
                z = scaler.transform(feat)
                df = svm.decision_function(z)

                if df.ndim == 2 and df.shape[0] == 1:
                    s = df[0].astype(np.float32)
                elif df.ndim == 1:
                    s = df.astype(np.float32)
                else:
                    s = None

                if (s is None) or (s.shape[0] != len(svm.classes_)):
                    pred = svm.predict(z)[0]
                    s = np.full((len(svm.classes_),), -10.0, dtype=np.float32)
                    pred_i = int(np.where(svm.classes_ == pred)[0][0])
                    s[pred_i] = 10.0

                per_tap_scores.append(s)
                lab = str(svm.classes_[int(np.argmax(s))])
                tap_labels.append(lab)

            # Raw stream display
            disp = []
            for lab in tap_labels:
                disp.append(" " if lab == "SPACE" else lab.lower())
            raw_stream = "".join(disp)
            label_stream = " ".join(["<SPACE>" if x == "SPACE" else x for x in tap_labels])

            print("\n=== SENTENCE (RAW) ===")
            print("Labels:", label_stream)
            print("Text  :", raw_stream)
            print("")

            # Split taps into word chunks at SPACE and decode each chunk
            words_raw, words_decoded, words_top = [], [], []
            cur_scores = []

            def decode_chunk(scores):
                L = len(scores)
                if L <= 0:
                    return "", "", []
                dyn_edits = min(3.0, max(1.0, 0.35 * L))
                raw, decoded, topc, _ = decode_word_chunk(
                    scores, svm.classes_,
                    temp=temp, beam_width=beam_width, topk_per_pos=topk_per_pos,
                    max_edits=float(dyn_edits), debug=False
                )
                opts = [w for w, _ in topc[:5]] if topc else [decoded]
                return raw, decoded, opts

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

            print("=== SENTENCE (DECODED) ===")
            print("Raw words   :", raw_words)
            print("Decoded sent:", decoded_sentence if decoded_sentence else "(empty)")
            if words_top:
                print("Top candidates per word:")
                for i, opts in enumerate(words_top, 1):
                    print(f"  word{i}: " + ", ".join(opts))
            print("")

            fb = input("Feedback: Enter=correct | type TRUE sentence | retry | quit : ").strip().lower()
            if fb == "quit":
                return
            if fb == "retry":
                print("\nRetrying...\n")
                continue

            if fb == "":
                print("✅ Marked correct.\n")
            else:
                print(f"❌ Marked true sentence = {fb}\n")

            again = input("Do another sentence attempt? (Enter=yes / type 'quit' to stop): ").strip().lower()
            if again == "quit":
                return
            print()

# =========================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python kbd_live_pipeline.py [collect_raw|train|demo]")
        sys.exit(1)

    mode = sys.argv[1].lower().strip()
    if mode == "collect_raw":
        mode_collect_raw()
    elif mode == "train":
        mode_train()
    elif mode == "demo":
        mode_demo()
    else:
        print("Unknown mode. Use: collect_raw | train | demo")
