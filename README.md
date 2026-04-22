# Acoustic Side Channel Attacks: Keystroke Classification from Microphone Recordings
**Domantas Prialgauskas · C00285607**

This project demonstrates an **acoustic side-channel attack** where **keyboard keystrokes** can be inferred using **only microphone audio**.

**Pipeline (lightweight + interpretable):**  
RMS segmentation → onset-aligned tap windows → MFCC (+Δ,+ΔΔ) → StandardScaler → SVM (RBF) → word/sentence decoding (beam + dictionary)

---

## Hardware used (reference setup)
- Keyboard: HyperX Alloy Origins — Cherry MX Red  
- Microphone: Fifine USB Microphone (48,000 Hz capture)

Results vary heavily depending on keyboard, switches, mic placement, room acoustics, and typing style.

---

## Repository contents
Core files:
- `app.py` — Streamlit GUI (3 modes)
- `kbd_live_pipeline.py` — collect/train/demo pipeline
- `requirements.txt`
- `README.md`

Folders:
- `datasets_live_current/` — training WAVs (A–Z + SPACE)
- `models_live_current/` — trained model artifacts (`.joblib`, meta, etc.)

Word decoding:
- `wordlist_common_clean.txt` (used via `WORDLIST_PATH` in `kbd_live_pipeline.py`)

---

## Requirements
- Windows 10/11
- Python 3.10+ recommended
- A working microphone (USB mic or laptop mic)

---

## Windows microphone permissions (important)
If recording fails (PortAudio/MME errors), enable mic access:

Settings → Privacy & security → Microphone
- Microphone access: ON
- Let apps access your microphone: ON
- Let desktop apps access your microphone: ON

Close any apps using the mic (Teams/Discord/Zoom/OBS/browser tabs).

---

## Install (PowerShell)
Open PowerShell in the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### If PowerShell blocks activation
Run once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

## Launch the Streamlit app
With venv activated:

```powershell
python -m streamlit run app.py
```

The app opens in a browser (usually `http://localhost:8501`).

---

## Using the Streamlit app
### 1) Choose microphone
Use the **Microphone / Input device** dropdown in the sidebar.
- If recording fails, try another device.
- Close other apps that might be holding the mic.

### 2) Choose a mode

#### Mode 1 — Single letter (vote)
- Press **Record & Decode**
- Press the same key `N` times during the recording window
- The app prints per-tap predictions + a majority-vote result

Suggested tests: `a`, `q`, `t`, `b`, `SPACE` (spacebar)

#### Mode 2 — Fixed-length word
- Set **Word length (L)** (e.g., 4 for “this”, 5 for “vote”)
- Press **Record & Decode**
- Type one word of exactly `L` letters

Suggested tests: `this`, `vote`, `quit`, `devil`, `line`

#### Mode 3 — Sentence
- Press **Record & Decode**
- Type a short sentence with spaces
- The app shows:
  - raw tap stream (including spaces)
  - decoded words per chunk

Sentence mode is harder — type slightly slower and increase recording length if needed.

### 3) Sidebar tuning knobs (quick meaning)
- Sensitivity / Threshold (K): higher = fewer taps (more strict), lower = more taps (more false positives)
- Debounce (ms): minimum gap between taps; increase if double-taps appear
- Temperature / Beam width / Top-K: decoding controls for word/sentence modes  
  - higher beam/top-K = more candidates (slower but can improve decoding)
  - temperature > 1.0 makes decoding less overconfident/more flexible

---

## CLI commands (collect / train / demo)
From the project folder (venv activated):

### Collect dataset (record → segment → save taps)
```powershell
python kbd_live_pipeline.py collect_raw
```

### Train (retrain on full live dataset)
```powershell
python kbd_live_pipeline.py train
```

### Terminal demo
```powershell
python kbd_live_pipeline.py demo
```

---

## If the app says “Model not found”
The app needs these files in `models_live_current/`:

```
models_live_current/
  svm_model_live.joblib
  scaler_live.joblib
  meta_live.joblib
```

To generate them:
```powershell
python kbd_live_pipeline.py train
```

Then launch the app:
```powershell
python -m streamlit run app.py
```

---

## Training on your own microphone and keyboard (step-by-step)
If you want the system to work well on your own hardware, record your own dataset and retrain.

### Step 1 — Clear the existing dataset (recommended)
Delete everything inside:
- `datasets_live_current/`

Keep the folder itself — just empty the contents.

(Optional) Also delete the contents of `models_live_current/` to avoid confusion. Retraining overwrites models anyway.

### Step 2 — Record your dataset (A–Z + SPACE)
Activate the venv:

```powershell
.\.venv\Scripts\Activate.ps1
```

Start recording:
```powershell
python kbd_live_pipeline.py collect_raw
```

You will be prompted for:
- record duration per session (seconds)
- minimum expected keystrokes
- label to collect (`A`–`Z` or `SPACE`)

Collection tips:
- Keep mic placement fixed for the entire dataset (distance + angle)
- Do multiple sessions per letter (e.g., 3–6 sessions per letter)
- Also collect SPACE (spacebar) — it helps sentence mode a lot

When finished, type `done` at the label prompt.

### Step 3 — Train
```powershell
python kbd_live_pipeline.py train
```

This retrains on the entire dataset and overwrites:
- `models_live_current/svm_model_live.joblib`
- `models_live_current/scaler_live.joblib`
- `models_live_current/meta_live.joblib`

### Step 4 — Launch the app
```powershell
python -m streamlit run app.py
```

---

## Troubleshooting
### Recording fails / MME error 1 / PortAudio error
- Enable Windows mic permissions (see above)
- Close apps using mic (Teams/Discord/Zoom/OBS)
- Try another input device in the Streamlit sidebar
- Disable Exclusive Mode:
  - Control Panel → Sound → Recording → Mic → Properties → Advanced
  - untick “Allow applications to take exclusive control”

### Training says “No training data found”
Your dataset must contain WAVs in this structure:

```
datasets_live_current/
  A/keystrokes/*.wav
  B/keystrokes/*.wav
  ...
  SPACE/keystrokes/*.wav
```

Fix by running:
```powershell
python kbd_live_pipeline.py collect_raw
```
