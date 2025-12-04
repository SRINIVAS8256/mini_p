## Brain Tumor Detection from MRI using Deep Learning

This project is a **Flask web application** that predicts whether a brain MRI scan contains a tumor and, if so, which type.  
It wraps a pre‑trained deep learning model (TensorFlow / Keras) in a simple browser interface so anyone can upload an MRI image and get an instant prediction.

---

## Features

- **Web UI for MRI upload**  
  Upload a brain MRI image (JPG/PNG). The app shows the image and the prediction result on the same page.

- **Tumor type classification**  
  The model predicts one of four categories:
  - `pituitary`
  - `glioma`
  - `meningioma`
  - `notumor`

- **Confidence score**  
  For every prediction, the app displays a confidence percentage (e.g. `84.32%`).

- **Production-ready serving**  
  Uses `gunicorn` + Flask, and is configured for deployment on **Render**.

- **GPU‑optional**  
  Runs on CPU‑only environments (like Render free tier); GPU is not required.

> **Disclaimer**: This project is for **educational and experimental purposes only** and must **not** be used for real medical diagnosis.

---

## Tech Stack

- **Backend**: Python, Flask
- **Deep Learning**: TensorFlow / Keras (`models/model.h5`)
- **Frontend**: HTML (Jinja2 templates)
- **Server**: gunicorn (for production), Flask dev server (for local testing)
- **Deployment**: Render (free web service)

---

## Project Structure

```text
mini_p/
├─ main.py              # Flask app and model loading
├─ models/
│  └─ model.h5          # Trained TensorFlow/Keras model
├─ templates/
│  └─ index.html        # Single-page UI for upload & results
├─ uploads/             # Uploaded MRI images (runtime)
├─ requirements.txt     # Python dependencies
├─ runtime.txt          # Python runtime version for Render
├─ render.yaml          # Render deployment configuration
└─ brain_tumour_detection_using_deep_learning.ipynb  # Training / experiments (Jupyter)
```

---

## How It Works

1. **Model loading**
   - On startup, `main.py` loads `models/model.h5` using TensorFlow / Keras.
   - A small compatibility patch is applied to the Keras `Flatten` layer to support **Keras 3.x** models.

2. **Image preprocessing**
   - Uploaded MRI image is resized to **128×128** pixels.
   - Pixel values are normalized to the range \([0, 1]\).
   - The image is converted into a batch of shape `(1, 128, 128, 3)` and fed to the model.

3. **Prediction logic**
   - The model outputs a probability vector over the four classes.
   - The app chooses the **argmax** class and the corresponding probability.
   - If the class is `notumor`, the UI shows **“No Tumor”**; otherwise, it shows **“Tumor: \<class\>”** and the confidence.

---

## Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/SRINIVAS8256/mini_p.git
cd mini_p
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> Note: Installing TensorFlow can take some time, especially on CPU‑only machines.

### 4. Run the Flask app (development mode)

```bash
python main.py
```

Then open your browser at `http://127.0.0.1:5000`.

### 5. Run with gunicorn (production style, local)

```bash
gunicorn main:app --bind 0.0.0.0:5000
```

Open `http://127.0.0.1:5000` (or your machine’s IP if running on a server).

---

## Deployment on Render

This repository is pre‑configured to deploy on **Render**.

### Key files

- `render.yaml` – defines a **Python web service**:
  - **Build command**: `pip install -r requirements.txt`
  - **Start command**: `gunicorn main:app --bind 0.0.0.0:$PORT`
- `runtime.txt` – pins Python version (3.11.x) for consistency.

### High‑level steps

1. Push the project to a GitHub repository.
2. Log into `render.com` and create a **New → Web Service**.
3. Connect the GitHub repo (`SRINIVAS8256/mini_p`).
4. Make sure the **Start Command** is:

   ```bash
   gunicorn main:app --bind 0.0.0.0:$PORT
   ```

5. Choose the **Free** plan (for demos / small usage).
6. Deploy and use the public URL provided by Render.

---

## Dataset & Training (High Level)

The model in `models/model.h5` was trained on MRI images of brain tumors.  
While the exact training notebook and dataset steps are summarized in `brain_tumour_detection_using_deep_learning.ipynb`, the high‑level pipeline is:

1. **Data preparation**
   - Collect MRI images labeled as `glioma`, `meningioma`, `pituitary`, and `notumor`.
   - Resize to a fixed resolution (128×128).
   - Normalize pixel values and optionally apply augmentation (flip, rotation, etc.).

2. **Model architecture**
   - A CNN built using Keras (Conv2D, MaxPooling, Flatten, Dense layers).
   - Final softmax layer over 4 classes.

3. **Training**
   - Loss: categorical cross‑entropy.
   - Optimizer: Adam (or similar).
   - Evaluation: accuracy on validation/test set.

The resulting weights are stored in `models/model.h5` and loaded by the Flask app.

---

## Limitations & Future Work

- **Not a medical tool** – predictions should never replace professional medical advice.
- **Model size / performance** – TensorFlow model is relatively heavy for CPU‑only free hosting:
  - First request after a cold start may be slow.
  - Future improvement: convert to **TensorFlow Lite** or use a more lightweight architecture.
- **Data generalization** – performance may drop on images from scanners or distributions very different from the training data.
- **UI/UX** – can be extended with:
  - Multiple image uploads
  - Better visualization (heatmaps / Grad‑CAM)
  - History of predictions

---

## License

This project is shared for **learning and portfolio purposes**.  
If you reuse this code, please give appropriate credit and verify all dependencies and licenses (especially the dataset you use for training).

---

## Acknowledgements

- TensorFlow and Keras teams for the deep learning framework.
- Open‑source MRI brain tumor datasets used for training and experimentation.
- Render for providing an easy way to host small ML web apps.
