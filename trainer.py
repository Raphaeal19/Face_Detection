import os
import cv2
from PIL import Image
import numpy as np
import pickle

face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

current_id = 0
label_ids = {}
y_label = []
x_train = []

for root, dirs, files in os.walk(image_dir):
	for file in files:
		if file.endswith("png") or file.endswith("jpg"):
			path = os.path.join(root, file)
			label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
			#print(label)
			if label in label_ids:
				pass
			else:
				label_ids[label] = current_id
				current_id += 1

			id_ = label_ids[label]
			pil_image = Image.open(path).convert("L")
			image_array = np.array(pil_image, "uint8")
			#print(image_array)
			faces = face_cascade.detectMultiScale(image_array)

			for(x, y, w, h) in faces:
				roi = image_array[y:y+h, x:x+w]
				x_train.append(roi)
				y_label.append(id_)

#print(label_ids)
#print(y_label)
#print(x_train)

with open("trainer_data/labels.pickle", 'wb') as f:
	pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_label))
recognizer.save("trainer_data/trainer.yml")