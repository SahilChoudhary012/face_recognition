import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import InceptionResnetV1
from scipy.spatial.distance import cosine
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

known_embeddings = np.load("embeddings.npy")
known_labels = np.load("labels.npy")

def preprocess_image(image):
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  
    ])
    return transform(image).unsqueeze(0).to(device)

def get_face_embedding(face_image):
    processed_image = preprocess_image(face_image)
    with torch.no_grad():
        embedding = model(processed_image)
    return embedding.cpu().numpy().flatten()

prototxt_path = "deploy.prototxt.txt"
caffemodel_path = "res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

def detect_faces(image, confidence_threshold=0.5):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    best_box = None
    best_confidence = 0.0

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > confidence_threshold and confidence > best_confidence:
            best_confidence = confidence
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w, endX), min(h, endY)

            best_box = (startX, startY, endX, endY)

    return [best_box] if best_box else []


def recognize_face(image):
    detected_faces = detect_faces(image)
    recognized_faces = []

    for (startX, startY, endX, endY) in detected_faces:
        face_roi = image[startY:endY, startX:endX]
        
        # Resize detected face for consistent display
        face_roi = cv2.resize(face_roi, (160, 160))  # Standard size for FaceNet
        
        embedding = get_face_embedding(face_roi)

        if embedding.size == 0:
            recognized_faces.append((startX, startY, endX, endY, "Unknown"))
            continue

        distances = [cosine(embedding, known_emb) for known_emb in known_embeddings]

        if len(distances) == 0:
            recognized_faces.append((startX, startY, endX, endY, "Unknown"))
            continue

        best_match_index = np.argmin(distances)
        label = known_labels[best_match_index] if distances[best_match_index] < 0.6 else "Unknown"

        recognized_faces.append((startX, startY, endX, endY, label))

    for (startX, startY, endX, endY, label) in recognized_faces:
        color = (0, 255, 0) if label != "Unknown" else (255, 0, 0)
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
        cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)  # Reduced font size

    return image

st.title("🎯 Face Recognition System")
st.write("Upload an image to recognize the face")

uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = np.array(Image.open(uploaded_file))
    result_image = recognize_face(image)
    
    st.image(result_image, caption="Result", use_column_width=True)
