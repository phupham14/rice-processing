# Basic processing pipeline to count rice grains in an image
# Retrieved and adapted from: https://docs.opencv.org/4.x/d2/d96/tutorial_py_table_of_contents_imgproc.html

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Đọc ảnh xám
img = cv2.imread('Proj1.2\\4.png', cv2.IMREAD_GRAYSCALE)

# 1. Cân bằng sáng
# Dùng Gaussian blur lớn để ước lượng nền sáng
background = cv2.GaussianBlur(img, (55, 55), 0)
corrected = cv2.subtract(img, background)
corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)

# 2. Lọc nhiễu
blur = cv2.GaussianBlur(corrected, (5, 5), 0)

# 3. Ngưỡng hóa 
# Dùng ngưỡng Otsu để tách hạt gạo ra khỏi nền
_, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# Đảo màu nếu hạt gạo bị đen
if np.mean(blur[thresh == 255]) < np.mean(blur[thresh == 0]):
    thresh = cv2.bitwise_not(thresh)

# 4. Morphology để làm sạch 
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

# 5. Distance Transform để tách các hạt dính
dist = cv2.distanceTransform(closing, cv2.DIST_L2, 5)
dist = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

# Ngưỡng dựa theo phần trăm max distance
_, sure_fg = cv2.threshold(dist, 0.3, 1.0, cv2.THRESH_BINARY)
sure_fg = np.uint8(sure_fg * 255)

# 6. Đếm các hạt
num_labels, labels = cv2.connectedComponents(sure_fg)

print("Số hạt gạo phát hiện được:", num_labels - 1)

# Tìm contour của các hạt gạo
contours, _ = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Tạo ảnh màu để vẽ bounding box
result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# Vẽ bounding box cho mỗi hạt gạo
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(corrected, cv2.COLOR_GRAY2RGB))
plt.title('Cân bằng sáng')

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB))
plt.title('Ảnh nhị phân')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title(f'Số hạt gạo phát hiện được ({len(contours)} hạt)')
plt.show()
