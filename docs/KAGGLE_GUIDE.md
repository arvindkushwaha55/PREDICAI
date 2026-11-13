# PREDICAI — Kaggle Training Guide (Shaam ke liye, step-by-step)

Notebook: `notebooks/PREDICAI_Kaggle_Training.ipynb` — **ek hi file, Run All, bas.**

## Steps

1. **Kaggle kholo** → Create → New Notebook → File → **Import Notebook** →
   `PREDICAI_Kaggle_Training.ipynb` upload karo.
2. **Settings (right side panel):**
   - Accelerator → **GPU T4 x1** (free quota kaafi hai; CPU par bhi chalega, bas slow)
   - Internet → **ON** (dataset auto-download ke liye chahiye)
3. **Dataset ke do raste (koi ek kaafi hai):**
   - *Asaan:* kuch mat karo — notebook khud NASA ki official site se
     `CMAPSSData.zip` download karke FD001 nikal lega (internet ON ho to).
   - *Ya:* Add Data → "NASA C-MAPSS / Turbofan Engine Degradation" dataset
     attach kar do — notebook use auto-detect kar lega.
4. **Run All** dabao. Training early-stopping ke saath chalegi (max 60 epochs,
   GPU par kuch minutes). Har epoch ki train/val RMSE screen par dikhegi.
5. **Parity check khud hoga** — aakhri cell me "PARITY OK" aana chahiye.
   Agar "PARITY FAILED" aaye to ruk jao aur mujhe batao, aage mat badho.
6. **Output download karo** (right side → Output) — ye 6 files:
   - `best_model.pt`
   - `cmapss_lstm.onnx`
   - `preprocessing.json`
   - `metrics.json`  ← **asli result isi me hai**
   - `training_curves.png`
   - `pred_vs_actual.png` (aur ho sake to `test_predictions.npy` bhi)
7. **Ye files mujhe bhej do** — uske baad Part 5 (evaluation verify) aur
   Part 7 (ONNX → Rust) main yahin se aage badha dunga.

## Dhyaan rahe

- Kuch change mat karna notebook me — config (seed 42, cap 125, 14 sensors)
  repo ke code se exact match rakha hai. Change karna ho to pehle mujhe bolo,
  warna Python ↔ Rust parity toot jayegi.
- `metrics.json` ke numbers hi asli result hain. Notebook ke bahar ka koi
  number kahin mat likhna.
- GPU quota khatam ho to CPU par chala lena — result wahi aayega, time zyada lagega.
