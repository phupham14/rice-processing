import cv2
import numpy as np
from scipy.ndimage import uniform_filter

#  Đọc ảnh grayscale
img = cv2.imread('Proj1.2//3.png', cv2.IMREAD_GRAYSCALE)
h, w = img.shape

#  FFT 2D
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

#  Lấy magnitude, log và chuẩn hóa
magnitude_spectrum = np.log(np.abs(fshift) + 1)
magnitude_log = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

#  Tạo mask ứng cử viên nhiễu
size_filter = 2
k = 2*size_filter + 1
local_mean = uniform_filter(magnitude_log.astype(np.float32), size=k)
diff = magnitude_log.astype(np.float32) - local_mean

center_row, center_col = h//2, w//2
diff[center_row, center_col] = 0

threshold = 50
mask_candidates = diff > threshold

#  Tạo notch filter
mask = np.ones((h, w), np.uint8)
mask[mask_candidates] = 0
coords = np.argwhere(mask_candidates)
for r, c in coords:
    mask[2*center_row - r, 2*center_col - c] = 0

#  Áp dụng mask
fshift_filtered = fshift * mask


#  Magnitude log sau khi lọc
magnitude_filtered = np.log(np.abs(fshift_filtered) + 1)
magnitude_filtered_log = cv2.normalize(magnitude_filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Chuyển về ảnh 2D
f_ishift = np.fft.ifftshift(fshift_filtered)
img_filtered = np.fft.ifft2(f_ishift)
img_filtered = np.abs(img_filtered).astype(np.uint8)


# ----------------------
#  Gaussian blur 5x5 làm mịn ảnh
blur = cv2.GaussianBlur(img_filtered, (5,5), 0)

#  Padding tránh lỗi ở rìa ảnh
pad_size = 5
blur_padded = cv2.copyMakeBorder(blur, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT)

#  CLAHE cân bằng histogram
clipLimit = 5
tileGridSize = (w//50, h//50)
clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
clahe_img = clahe.apply(blur_padded)

#  Adaptive Threshold (Gaussian) nhị aphn ảnh
binary = cv2.adaptiveThreshold(
    clahe_img, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    55,
    -22
)
binary = binary[pad_size:pad_size+h, pad_size:pad_size+w]  # cắt lại ảnh ban đầu

#  Close Morphological
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
binary_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

#  Erosion
binary_eroded = cv2.erode(binary_closed, kernel, iterations=1)

# : Tìm contours và lọc theo diện tích
contours, _ = cv2.findContours(binary_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
areas = [cv2.contourArea(c) for c in contours]
if areas:
    max_area = max(areas)
    filtered_contours = [c for c, a in zip(contours, areas) if a >= 0.08*max_area]
else:
    filtered_contours = []

#Đếm số hạt gạo
num_rice = len(filtered_contours)
print("Số hạt gạo:", num_rice)

#  Hiển thị kết quả
result = cv2.cvtColor(img_filtered, cv2.COLOR_GRAY2BGR)
cv2.drawContours(result, filtered_contours, -1, (0,0,255), 1)
#  Hiển thị ảnh miền tần số trước và sau lọc

cv2.imshow('Filtered', img_filtered)
cv2.imshow('Orginal', img)
cv2.imshow('Magnitude Log Before Filtering', magnitude_log)
cv2.imshow('Magnitude Log After Filtering', magnitude_filtered_log)
cv2.imshow('draw', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
