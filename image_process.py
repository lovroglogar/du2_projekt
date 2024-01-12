import os
import argparse

import cv2
import mediapipe as mp


def process_img(img, face_detection,method):

    H, W, _ = img.shape

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box

            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)

            # print(x1, y1, w, h)

            # blur faces with gaussian blur
            if method == "gauss":
                img[y1:y1 + h, x1:x1 + w, :] = cv2.GaussianBlur(img[y1:y1 + h, x1:x1 + w, :], (25, 25), 0)
            elif method == "avg":
                img[y1:y1 + h, x1:x1 + w, :] = cv2.blur(img[y1:y1 + h, x1:x1 + w, :], (25, 25))
    return img


# Set up output directories
input_dir = 'dub2_data/celeba_hq_256'
output_gauss_dir = './gauss_blurred_images'
output_avg_dir = './avg_blurred_images'

if not os.path.exists(output_gauss_dir):
    os.makedirs(output_gauss_dir)
if not os.path.exists(output_avg_dir):
    os.makedirs(output_avg_dir)

# Initialize face detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

import torchvision
import numpy as np
from PIL import Image

# Loop over each image in the dataset
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Checking file extension
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        trans = torchvision.transforms.Compose([torchvision.transforms.Resize((64, 64))])
        img = np.array(trans(Image.fromarray(img)))

        if img is not None:
            try:
                # Process with Gaussian Blur
                img_gauss = process_img(img.copy(), face_detection, method="gauss")
                cv2.imwrite(os.path.join(output_gauss_dir, 'gauss_' + filename), img_gauss)

                # Process with Averaging Blur
                img_avg = process_img(img.copy(), face_detection, method="avg")
                cv2.imwrite(os.path.join(output_avg_dir, 'avg_' + filename), img_avg)
            except Exception as e:
                print("Error processing image: ", filename)

face_detection.close()
