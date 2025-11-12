# 🍚 Đếm số hạt gạo bằng OpenCV

Dự án này minh họa quy trình **xử lý ảnh cơ bản** để **đếm số hạt gạo** tự động trong một bức ảnh sử dụng **OpenCV** và **NumPy**.  
Quy trình bao gồm các bước kinh điển như cân bằng sáng, lọc nhiễu, tách hạt bằng Distance Transform và ngưỡng hóa Otsu.

---

## 📷 Ảnh đầu vào

Ví dụ: ảnh xám chứa nhiều hạt gạo trên nền tương phản.


---

## ⚙️ Quy trình xử lý (Pipeline)

| Bước | Kỹ thuật | Mục đích |
|------|-----------|-----------|
| 1️⃣ | **Gaussian Blur (lớn)** | Ước lượng & trừ nền sáng không đều |
| 2️⃣ | **Gaussian Blur (nhỏ)** | Giảm nhiễu trước khi tách nền |
| 3️⃣ | **Ngưỡng Otsu** | Tự động tách vật thể (hạt gạo) khỏi nền |
| 4️⃣ | **Phép hình thái học (Morphology)** | Làm sạch ảnh nhị phân (xóa nhiễu, lấp lỗ nhỏ) |
| 5️⃣ | **Distance Transform** | Tách các hạt dính nhau |
| 6️⃣ | **Connected Components** | Đếm các vùng hạt riêng biệt |
| 7️⃣ | **Contours + Bounding Box** | Vẽ khung quanh từng hạt gạo để minh họa |

---

## 💻 Code minh họa

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 1. Đọc ảnh xám
img = cv2.imread('Proj1.2\\4.png', cv2.IMREAD_GRAYSCALE)

# 2. Cân bằng sáng
background = cv2.GaussianBlur(img, (55, 55), 0)
corrected = cv2.subtract(img, background)
corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)

# 3. Lọc nhiễu
blur = cv2.GaussianBlur(corrected, (5, 5), 0)

# 4. Ngưỡng hóa Otsu
_, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
if np.mean(blur[thresh == 255]) < np.mean(blur[thresh == 0]):
    thresh = cv2.bitwise_not(thresh)

# 5. Morphology để làm sạch
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

# 6. Distance Transform để tách hạt
dist = cv2.distanceTransform(closing, cv2.DIST_L2, 5)
dist = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
_, sure_fg = cv2.threshold(dist, 0.3, 1.0, cv2.THRESH_BINARY)
sure_fg = np.uint8(sure_fg * 255)

# 7. Đếm số hạt
num_labels, labels = cv2.connectedComponents(sure_fg)
print("Số hạt gạo phát hiện được:", num_labels - 1)

# 8. Vẽ kết quả
contours, _ = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

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
plt.title(f'Số hạt gạo phát hiện ({len(contours)} hạt)')
plt.show()
