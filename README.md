# Acoustic Side Channel Attacks: Keystroke Classification from Microphone Recordings
**Domantas Prialgauskas · C00285607**

This project demonstrates an **acoustic side-channel attack** where **keyboard keystrokes** are classified using **only microphone audio**.  
It uses a lightweight, interpretable pipeline:

**RMS segmentation → onset-aligned tap windows → MFCC (+Δ,+ΔΔ) → StandardScaler → SVM (RBF) → word/sentence decoding (beam + dictionary)**

The live demo is presented via a **Streamlit GUI** and a CLI pipeline script.  
> Note: Offline “defensible evaluation” datasets are not included here. This repo is intended for the **live demo system**.

---

## Project Structure (minimal demo)
Required for demo:
- `kbd_live_pipeline.py` — core segmentation + MFCC + SVM + decoding
- `app.py` — Streamlit GUI
- `models_live_current/` — trained artifacts (must include):
  - `svm_model_live.joblib`
  - `scaler_live.joblib`
  - `meta_live.joblib` (optional but recommended)
- `wordlist_common_clean.txt`
- `AcousticSideChannelAttacks.html` (showcase site)

---

## Requirements
- Windows 10/11
- Python 3.10+ recommended
- A working microphone (USB mic or laptop mic)

---

## Setup (Laptop)
Open PowerShell in the project folder and run:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt

## Train the model using downloaded datasets & run the application
```powershell
python kbd_pipeline.py train
python -m streamlit run app.py

## Using the Streamlit Application (after launch)
Once the app opens in your browser (usually `http://localhost:8501`):

### 1) Select a microphone
Use the **Microphone / Input device** dropdown in the left sidebar.
- If recording fails or no taps are detected, try a different input device.
- Close other apps using the mic (Teams/Discord/Zoom/OBS).

### 2) Choose a demo mode
In the sidebar, select one of:

**Mode 1 — Single letter (vote)**
- Press **Record & Decode**
- During recording, press the **same key** `N` times
- The app shows each tap prediction and a final majority-vote result  
Recommended for quick validation (e.g., `a`, `q`, `t`, `b`, `SPACE`).

**Mode 2 — Fixed-length word**
- Set **Word length (L)** (e.g., 4 for “this”, 5 for “vote”)
- Press **Record & Decode**
- Type **one word** of exactly `L` letters
- The app shows:
  - raw top-1 letters
  - beam-search candidates
  - decoded dictionary word  
Recommended words: `this`, `vote`, `quit`, `devil`, `line`.

**Mode 3 — Sentence**
- Press **Record & Decode**
- Type a short sentence with spaces
- The app outputs:
  - raw tap stream (including spaces)
  - decoded words per chunk  
Sentence mode is more sensitive to missed taps; type slightly slower.

### 3) Tuning knobs (when needed)
- **Sensitivity / Threshold (K)**: higher = fewer taps detected (more strict), lower = more taps detected (more false positives)
- **Debounce (ms)**: minimum time between taps; raise it if double-taps are being detected
- **Temperature / Beam width / Top-K**: decoding controls for word/sentence modes  
  - higher beam/top-K = more candidates (slower but often better)
  - temperature > 1.0 makes decoding less “overconfident” and more flexible

---

## Troubleshooting
### “Model not found”
Run training or copy the trained model folder:
```powershell
python kbd_live_pipeline.py train
