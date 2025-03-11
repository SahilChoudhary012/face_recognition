import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
import os
from scipy.spatial.distance import cosine

prototxt_path = "deploy.prototxt.txt"
caffemodel_path = "res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

def detect_faces(image, confidence_threshold=0.5):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    best_face = None
    best_confidence = 0.0
    best_box = None

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold and confidence > best_confidence:
            best_confidence = confidence
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w, endX), min(h, endY)

            best_box = (startX, startY, endX, endY)
            best_face = image[startY:endY, startX:endX]

    return [best_box] if best_face is not None else []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

known_embeddings = np.load("embeddings.npy")
known_labels = np.load("labels.npy")

def preprocess_image(image):
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

def recognize_face(image):
    detected_faces = detect_faces(image)  
    recognized_faces = []

    for (startX, startY, endX, endY) in detected_faces:
        face_roi = image[startY:endY, startX:endX]
        embedding = get_face_embedding(face_roi)

        if embedding.size == 0:
            recognized_faces.append((startX, startY, endX, endY, "Unknown"))
            continue  

        distances = [cosine(embedding, known_emb) for known_emb in known_embeddings]

        if len(distances) == 0:  
            recognized_faces.append((startX, startY, endX, endY, "Unknown"))
            continue

        best_match_index = np.argmin(distances)
        
        if distances[best_match_index] < 0.6:
            recognized_faces.append((startX, startY, endX, endY, known_labels[best_match_index]))
        else:
            recognized_faces.append((startX, startY, endX, endY, "Unknown"))

    for (startX, startY, endX, endY, label) in recognized_faces:
        color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
        cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    return image

def display_image(image, window_name="Face Recognition"):
    fixed_width, fixed_height = 600, 400  
    resized_image = cv2.resize(image, (fixed_width, fixed_height))  
    cv2.imshow(window_name, resized_image)

image_folder = "images/"
for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)
    
    if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        test_image = cv2.imread(image_path)
        result_image = recognize_face(test_image)

        display_image(result_image, f"Result - {image_name}")
        cv2.waitKey(0)

cv2.destroyAllWindows()
