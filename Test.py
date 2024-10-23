import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# Initialize webcam
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("converted_keras\\keras_model.h5", "converted_keras\\labels.txt")

offset = 20
imgSize = 300
labels = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", 
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", 
    "U", "V", "W", "X", "Y", "Z"
]

while True:
    success, img = cap.read()
    if not success:  # Check if the frame is captured successfully
        print("Failed to capture image")
        break
        
    
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        
        y1 = max(0, y - offset)#Define the region to crop
        y2 = min(img.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)

        imgCrop = img[y1:y2, x1:x2]
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCropShape = imgCrop.shape
        aspectRatio = h / w
        
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
        
        prediction, index = classifier.getPrediction(imgWhite, draw=False)
        print(prediction, index)

        # Draw a thicker rectangle with contrasting color for visibility
        border_color = (0, 255, 0)  # Green color
        border_thickness = 3  # Thicker border
        shadow_thickness = 5  # Shadow thickness

        # Draw shadow (optional for effect)
        cv2.rectangle(imgOutput, (x1 - shadow_thickness, y1 - 50 - shadow_thickness),(x1 + 90 + shadow_thickness, y1 + shadow_thickness), (0, 0, 0), cv2.FILLED)

        # Draw the main rectangle
        cv2.rectangle(imgOutput, (x1, y1 - 50), (x1 + 90, y1), border_color, cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x1, y1 - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x1, y1), (x2, y2), border_color, border_thickness)

        # Show the cropped and resized images
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)

    # Exit option
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()
