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

for img_name in os.listdir(cropped_faces_path):
    img_path = os.path.join(cropped_faces_path, img_name)
    image = cv2.imread(img_path)

    if image is not None:
        embedding = get_face_embedding(image)
        embeddings.append(embedding)

        label = img_name.split('_')[0]
        labels.append(label)

np.save("embeddings.npy", np.array(embeddings))
np.save("labels.npy", np.array(labels))

print("Feature extraction and encoding completed successfully!")
