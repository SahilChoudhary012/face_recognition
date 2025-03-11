import cv2
import numpy as np
import os

prototxt_path = "deploy.prototxt.txt"
caffemodel_path = "res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

def detect_faces(image, confidence_threshold=0.5):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    faces = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            face = image[startY:endY, startX:endX]
            faces.append(face)
            
            text = f"{confidence * 100:.2f}%"
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(image, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image, faces

data_path = "data/"
output_path = "cropped_faces/" 
os.makedirs(output_path, exist_ok=True)

for person_folder in os.listdir(data_path):
    person_path = os.path.join(data_path, person_folder)
    if os.path.isdir(person_path):
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            image = cv2.imread(img_path)

            if image is not None:
                result_image, faces = detect_faces(image)

                for idx, face in enumerate(faces):
                    save_path = os.path.join(output_path, f"{person_folder}_{img_name.split('.')[0]}_{idx}.jpg")
                    cv2.imwrite(save_path, face)
                
                cv2.imshow(f"Detected Faces - {img_name}", result_image)
                cv2.waitKey(1000) 
                cv2.destroyAllWindows()

print("Face detection and cropping completed successfully!")
