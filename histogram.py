import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('my_selfie.jpg', cv2.IMREAD_GRAYSCALE)
_, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
se = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

dilation = cv2.dilate(bin_img, se)
erosion = cv2.erode(bin_img, se)
opening = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, se)
closing = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, se)

titles = ['Binary', 'Dilation(Hit)', 'Erosion(Fit)', 'Opening', 'Closing']
images = [bin_img, dilation, erosion, opening, closing]

plt.figure(figsize=(15, 5))
for i in range(5):
    plt.subplot(1, 5, i+1); plt.imshow(images[i], cmap='gray')
    plt.title(titles[i]); plt.axis('off')

plt.savefig('assignment_4.png')
print("Assignment 4 saved.")