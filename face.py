from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt

import os

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error opening video capture.")
    exit()

ret, frame = cap.read()

cap.release()

desktop_path = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
filename = "captured_image.jpg"
filepath = os.path.join(desktop_path, filename)
cv2.imwrite(filepath, frame)

print(f"Image saved to {filepath}")
converted_path = filepath.replace("\\", "\\\\")
print(converted_path)


#Add your photo for the compare camera capture photo and default photo.
img1 = cv2.imread("C:\\Users\\arda.peker\\Desktop\\ArdaPeker.png")
plt.imshow(img1[:,:,::-1])



img2 = cv2.imread(converted_path)
plt.imshow(img2[:,:,::-1])

result= DeepFace.verify(img1,img2)

print(result)