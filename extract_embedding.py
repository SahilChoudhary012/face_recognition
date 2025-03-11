import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
import os
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

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

cropped_faces_path = "cropped_faces/"
embeddings = []
labels = []

valid_extensions = ['.jpg', '.jpeg', '.png']

for person_folder in os.listdir(cropped_faces_path):
    person_path = os.path.join(cropped_faces_path, person_folder)
    
    if not os.path.isdir(person_path):
        continue

    for img_name in os.listdir(person_path):
        if any(img_name.lower().endswith(ext) for ext in valid_extensions):
            img_path = os.path.join(person_path, img_name)
            image = cv2.imread(img_path)

            if image is None:
                print(f"Image not loaded (possibly corrupt): {img_path}")
                continue

            if image.shape[0] < 10 or image.shape[1] < 10:
                print(f"Skipping tiny image (Invalid face crop): {img_path}")
                continue

            embedding = get_face_embedding(image)
            embeddings.append(embedding)

            labels.append(person_folder)

if len(embeddings) > 0:
    np.save("embeddings.npy", np.array(embeddings))
    np.save("labels.npy", np.array(labels))
    print(f"Feature extraction completed successfully! Total embeddings: {len(embeddings)}")
else:
    print("No embeddings were generated. Check the dataset or face detection logic.")
