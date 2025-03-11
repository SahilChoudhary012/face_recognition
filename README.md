# Face Recognition 

This repository provides a simple yet effective face recognition system using **OpenCV**, **FaceNet**, and **PyTorch**. It detects faces in images, extracts embeddings, and identifies known actors based on pre-trained embeddings.

## 📋 Requirements
Install the required dependencies using the following command:
```bash
pip install -r requirements.txt
```

**Main Libraries Used:**
- `OpenCV` - For face detection and image processing
- `FaceNet-PyTorch` - For generating face embeddings
- `NumPy` - For array manipulations
- `SciPy` - For calculating cosine similarity

---

## ⚙️ Environment Setup
1. Clone this repository:
   ```bash
   git clone <repository_link>
   cd <repository_name>
   ```
2. Create a virtual environment:
   ```bash
   python -m venv face_recognition_env
   source face_recognition_env/bin/activate  # On Linux/Mac
   face_recognition_env\Scripts\activate     # On Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Ensure the following files are present in the directory:
   - `deploy.prototxt.txt` - DNN model structure for face detection
   - `res10_300x300_ssd_iter_140000.caffemodel` - Pre-trained model for face detection

---

## 📂 Project Structure
```
.
├── data/                   # Contains sample images of Hollywood male actors (for embedding creation)
├── images/                 # Contains test images to identify the actors
├── .gitignore
├── README.md
├── deploy.prototxt.txt     # DNN model structure for face detection
├── extract_embedding.py    # Extracts embeddings from known actors
├── face_recognition.py     # Face detection and visualization
├── recognize_faces.py      # Main file to recognize faces in new images
├── requirements.txt        # Required libraries
├── res10_300x300_ssd_iter_140000.caffemodel  # Pre-trained model for face detection
```

---

## 📝 File Descriptions and Execution Order
1. **`extract_embedding.py`**
   - Extracts face embeddings from images in the `data/` folder.
   - Run this file first to generate embeddings for known actors.
   ```bash
   python extract_embedding.py
   ```

2. **`face_recognition.py`**
   - Detects and highlights faces in images for visualization.
   - Useful for testing face detection before recognition.
   ```bash
   python face_recognition.py
   ```

3. **`recognize_faces.py`**
   - Identifies faces in the `images/` folder by comparing them with known embeddings.
   - Displays identified actor names or marks them as `Unknown`.
   ```bash
   python recognize_faces.py
   ```

---

## 🧠 Code Flow
1. **Face Detection:**
   - Uses the `deploy.prototxt.txt` and `res10_300x300_ssd_iter_140000.caffemodel` for face detection.
   - Detects **only the most confident face** in case multiple faces are found.

2. **Embedding Generation:**
   - Extracts embeddings from face crops using the `InceptionResnetV1` model from FaceNet.

3. **Recognition Process:**
   - Compares embeddings with known embeddings using **cosine similarity**.
   - Faces with similarity scores below `0.6` are marked as `Unknown`.

---

## 🔎 Notes
✅ The `data/` folder contains images of some **Hollywood male actors** for training.  
✅ The `images/` folder contains test images that the model attempts to recognize.  
✅ The code processes image formats `.jpg`, `.jpeg`, and `.png` only.  
✅ All detected faces are highlighted with bounding boxes for better visualization.

If you have any questions or issues, feel free to raise an issue in the repository. 🚀

