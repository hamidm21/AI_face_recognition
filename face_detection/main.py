import cv2
import argparse

# Get supplied values
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-c", "--cascade", required=True,
	help="path to input image")
args = vars(ap.parse_args())

imagePath = args["image"]
cascPath = args["cascade"]

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.14,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
)

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("target found", image)
cv2.waitKey(0)