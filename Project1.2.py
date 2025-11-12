import cv2
import numpy as np


img = cv2.imread('C://D//kscs//hustCV//Proj1.2//1_wIXlvBeAFtNVgJd49VObgQ.png_Salt_Pepper_Noise1.png', cv2.IMREAD_GRAYSCALE)

# Extracting the height and width of an image
h, w = img.shape[:2]
# Displaying the height and width
print("Height = {}, Width = {}".format(h, w))

'''
ksize = (h, w)
img = cv2.blur(img, ksize) 
'''

'''
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

cv2.imshow('Original Image', gray_image)
cv2.imshow('Blurred Image', blurred_image)
'''


'''
'''
median_blurred_img = cv2.medianBlur(img, 5)
#cv2.imshow('Blurred Image', median_blurred_img)




edged = cv2.Canny(median_blurred_img, 30, 200)
contours, hierarchy = cv2.findContours(edged,
                      cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
print("Number of Contours Found = " + str(len(contours)))
cv2.imshow('Blurred Image', edged)

cv2.waitKey(0)
cv2.destroyAllWindows()








