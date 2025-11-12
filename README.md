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

📦 Hướng dẫn cài đặt
1️⃣ Yêu cầu hệ thống

Python ≥ 3.8

OpenCV ≥ 4.5

NumPy

Matplotlib

2️⃣ Cài đặt thư viện
pip install opencv-python numpy matplotlib

3️⃣ Chạy chương trình

Lưu file thành rice_counter.py, sau đó chạy:

python rice_counter.py


Kết quả sẽ hiển thị:

Ảnh sau cân bằng sáng

Ảnh nhị phân sau ngưỡng hóa

Ảnh có khung xanh quanh các hạt gạo và tổng số lượng phát hiện được

✅ Ưu điểm

Tự động xác định ngưỡng tách vật thể (Otsu)

Không cần học máy, dễ triển khai

Hoạt động tốt với ảnh có nền tương phản rõ

Có thể mở rộng sang các ứng dụng đếm đối tượng khác (hạt cà phê, tế bào, v.v.)

⚠️ Nhược điểm

Kém hiệu quả khi ảnh có ánh sáng không đều

Otsu giả định histogram có hai đỉnh rõ ràng (bimodal)

Các hạt dính nhau có thể bị đếm thiếu nếu tách chưa tốt

Cần tinh chỉnh tham số Distance Transform hoặc kích thước kernel để đạt kết quả tối ưu

🚀 Hướng phát triển

🧩 Watershed Segmentation: tách ranh giới hạt dính nhau chính xác hơn

🌗 Adaptive Thresholding: xử lý ảnh có ánh sáng không đều

🔍 Bộ lọc diện tích contour: bỏ qua các vật thể nhỏ không phải hạt gạo

📈 Thống kê kích thước trung bình: phân tích hình dạng hoặc kích thước hạt
