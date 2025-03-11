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

    if best_face is not None:
        cv2.rectangle(image, (best_box[0], best_box[1]), (best_box[2], best_box[3]), (0, 255, 0), 2)
        text = f"{best_confidence * 100:.2f}%"
        cv2.putText(image, text, (best_box[0], best_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return image, [best_face] if best_face is not None else []

data_path = "data/"
output_path = "cropped_faces/"
os.makedirs(output_path, exist_ok=True)

for person_folder in os.listdir(data_path):
    person_path = os.path.join(data_path, person_folder)
    if os.path.isdir(person_path):
        person_output_path = os.path.join(output_path, person_folder)
        os.makedirs(person_output_path, exist_ok=True)

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            image = cv2.imread(img_path)

            if image is not None:
                result_image, faces = detect_faces(image)

                for idx, face in enumerate(faces):
                    save_path = os.path.join(person_output_path, f"{img_name.split('.')[0]}_{idx}.jpg")
                    cv2.imwrite(save_path, face)
                
                cv2.imshow(f"Detected Faces - {img_name}", cv2.resize(result_image, (600, 400)))
                cv2.waitKey(1000)  
                cv2.destroyAllWindows()

print("Face detection and cropping completed successfully!")
