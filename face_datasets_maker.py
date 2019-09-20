#!/usr/bin/env python3

# Import OpenCV2 for image processing
import cv2
import os

name = input("What is the name of the new person?")
images_folder = os.path.join("images", name)
os.makedirs(images_folder)
#images_folder = os.path.join(images_folder, "/")
# Start capturing video 
vid_cam = cv2.VideoCapture(0)

# Detect object in video stream using Haarcascade Frontal Face
face_detector = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')

# Initialize sample face image
count = 0
# Start looping
while(True):

    # Capture video frame
    _, image_frame = vid_cam.read()
 	
    # Convert frame to grayscale
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)

    # Detect frames of different sizes, list of faces rectangles
    faces = face_detector.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 5)

    # Loops for each faces
    for (x,y,w,h) in faces:

        # Crop the image frame into rectangle
        cv2.rectangle(image_frame, (x,y), (x+w,y+h), (255,0,0), 2)
        
        # Increment sample face image
        count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite(images_folder + "/"  + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        # Display the video frame, with bounded rectangle on the person's face
        cv2.imshow('frame', image_frame)

    # To stop taking video, press 'q' for at least 100ms
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

    # If image taken reach 100, stop taking video
    elif count>2:
        break

# Stop video
vid_cam.release()

# Close all started windows
cv2.destroyAllWindows()

# from distutils.dir_util import copy_tree
# copy_tree("images_raw", "images/new")
