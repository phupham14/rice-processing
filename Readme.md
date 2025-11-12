#  Dự án: Đếm số hạt gạo bằng xử lý ảnh tần số (FFT) và OpenCV

##  Giới thiệu
Dự án này thực hiện **đếm số hạt gạo** trong ảnh chụp thông qua các bước **lọc nhiễu trong miền tần số** kết hợp **tiền xử lý không gian** và **phân đoạn ảnh nhị phân**.  
Phương pháp giúp loại bỏ nhiễu tuần hoàn (nhiễu sin, nhiễu dạng sọc), làm mịn ảnh và phát hiện các hạt gạo riêng biệt chính xác hơn.

---

##  Quy trình xử lý

### B1. Đọc ảnh và biến đổi Fourier
- Ảnh được đọc ở dạng **grayscale**.  
- Thực hiện biến đổi **FFT 2D** để đưa ảnh sang **miền tần số**.  
- Áp dụng **log và chuẩn hóa magnitude spectrum** để dễ dàng nhận biết các điểm nhiễu.

```python
img = cv2.imread('Proj1.2//3.png', cv2.IMREAD_GRAYSCALE)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
```

### B2. Phát hiện và loại bỏ nhiễu trong miền tần số

* Tính giá trị trung bình cục bộ (local mean) bằng uniform_filter.

* So sánh điểm ảnh với trung bình lân cận → điểm nào lệch quá ngưỡng (threshold = 50) được coi là ứng cử viên nhiễu.

* Tạo notch filter để loại bỏ đối xứng các điểm nhiễu quanh tâm phổ.

``` python 
diff = magnitude_log.astype(np.float32) - local_mean
mask_candidates = diff > threshold
mask[mask_candidates] = 0
```
### B3. Lọc nhiễu và chuyển ngược về miền không gian

* Nhân phổ ảnh với mask notch filter để loại bỏ tần số gây nhiễu.

* Biến đổi ngược IFFT để thu lại ảnh đã lọc nhiễu.

```python
fshift_filtered = fshift * mask
img_filtered = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift_filtered))).astype(np.uint8)
``` 

### B4. Tiền xử lý không gian

* Gaussian Blur (5×5) để làm mịn biên.

* CLAHE (Contrast Limited Adaptive Histogram Equalization) để tăng tương phản cục bộ.

* Adaptive Threshold (Gaussian) để chuyển sang ảnh nhị phân.

```python
blur = cv2.GaussianBlur(img_filtered, (5,5), 0)
clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(w//50, h//50))
clahe_img = clahe.apply(blur)
binary = cv2.adaptiveThreshold(...)
```

### B5. Xử lý hình thái học

Closing để nối các vùng bị đứt nét.

Erosion để loại bỏ nhiễu nhỏ còn sót lại.

```python
binary_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
binary_eroded = cv2.erode(binary_closed, kernel, iterations=1)
``` 

### B6. Đếm số hạt gạo

Tính diện tích của từng contour.

Giữ lại các contour có diện tích ≥ 8% diện tích lớn nhất.

In ra số lượng hạt gạo đếm được và vẽ kết quả.

```python
filtered_contours = [c for c, a in zip(contours, areas) if a >= 0.08*max_area]
print("Số hạt gạo:", len(filtered_contours))
```

### B7. Kết quả hiển thị

Chương trình hiển thị các cửa sổ:

* Orginal: ảnh gốc

* Magnitude Log Before Filtering: phổ tần trước khi lọc

* Magnitude Log After Filtering: phổ tần sau khi lọc

* Filtered: ảnh sau khử nhiễu

* Draw: ảnh có viền đỏ quanh các hạt gạo được đếm

```python
cv2.imshow('Orginal', img)
cv2.imshow('Filtered', img_filtered)
cv2.imshow('Magnitude Log Before Filtering', magnitude_log)
cv2.imshow('Magnitude Log After Filtering', magnitude_filtered_log)
cv2.imshow('Draw', result)
```

## Kết luận

Phương pháp sử dụng FFT và lọc nhiễu trong miền tần số giúp:

* Loại bỏ hiệu quả các nhiễu tuần hoàn.

* Cải thiện độ tương phản và tách biên rõ ràng hơn.

* Đếm chính xác số lượng hạt gạo trong ảnh có nhiễu nền phức tạp.

Tuy nhiên khi các hạt gạo ở gần nhau, dễ bị gom thành 1 object

