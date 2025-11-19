# 🏌️‍♂️ Golf AI Coach — Phân Tích Tư Thế & Cú Swing Golf (DataStorm 2025)

Dự án tham dự **DataStorm 2025** với chủ đề *Phân tích hành vi thể thao từ video (Golf Pose Analysis)*.  
Hệ thống áp dụng Computer Vision & Biomechanics để phân tích kỹ thuật cú swing golf dựa trên video người dùng cung cấp.

---

## 📜 Giới thiệu

**Golf AI Coach** là hệ thống hỗ trợ huấn luyện golf bằng AI, với khả năng:

- Trích xuất tư thế cơ thể (pose estimation)
- Tính toán các góc và chỉ số cơ sinh học
- Tạo dashboard video phân tích kỹ thuật
- Xây dựng bộ dữ liệu đặc trưng cho mô hình ML
- Sinh "Hồ sơ swing lý tưởng" từ các mẫu chuẩn

Dự án đặt mục tiêu hỗ trợ huấn luyện viên và người chơi golf cải thiện kỹ thuật dựa trên phân tích khách quan.

---

## ✨ Tính năng nổi bật

### 📐 1. Trích xuất đặc trưng cơ sinh học  
Sử dụng MediaPipe để trích xuất **keypoints**, sau đó tính toán:

- Góc khuỷu tay (trái/phải)  
- Góc đầu gối (trái/phải)  
- Độ nghiêng vai  
- Độ nghiêng hông  
- Góc xoay cơ thể  
- Biên độ backswing – downswing  

---

### ⚙️ 2. Xử lý video hàng loạt  
-   **Đầu vào:** Một file video (`.mp4`) quay lại cú swing.
-   **Xử lý:**
    1.  Tự động nhận diện người và trích xuất 33 điểm khớp trên cơ thể trong từng khung hình.
    2.  Tính toán các chỉ số cơ sinh học quan trọng (góc, độ nghiêng).
    3.  (Tùy chọn) So sánh các chỉ số này với một "Hồ sơ Swing Lý tưởng" được xây dựng từ dữ liệu.
-   **Đầu ra:** Một video dashboard tổng hợp, trực quan hóa toàn bộ quá trình phân tích.
  
---

### 📊 3. Xây dựng Hồ sơ Swing Lý tưởng  
Tự động tính toán các thông số trung bình từ các cú swing "Tốt" để tạo ra một điểm chuẩn (benchmark) khách quan.

---

### 🎥 4. Tạo Video Phân Tích 4-trong-1  
Dashboard gồm:

1. Video gốc  
2. Video vẽ skeleton  
3. Bảng dữ liệu thời gian thực  

### 🚀 Demo Sản phẩm (Vòng loại 1)
Video kết quả hiển thị khả năng tính toán và hiển thị các thông số góc một cách chính xác từ video đầu vào.


![simple_analyzed_Untitled00014096 (2)](https://github.com/user-attachments/assets/75fc495f-24c2-4796-89fc-be6d8ded0452)


### 🛠️ 5. Kiến trúc dự án
```css
Data_storm_2025/
│
├── notebooks/
│   └── cal_pose.ipynb
│
├── src/
│   ├── pose_extractor.py      → Trích xuất pose & keypoints
│   ├── build_feature.py       → Sinh dataset đặc trưng
│   ├── swing_profile.py       → Tạo hồ sơ lý tưởng
│   └── video_analyzer.py      → Tạo video phân tích
│
├── requirements.txt
└── README.md
```
### 🔧 Cài đặt & Chạy thử
1️⃣ Clone repo từ github:  
```bash
git clone https://github.com/hovannhuy/Data_storm_2025.git
cd Data_storm_2025
```
2️⃣ Sử dụng Google Colab và và tải lên notebooks cal_pose.ipynb
3️⃣ Kết nối thời gian chạy, tải video lên bộ nhớ phiên của Google Colab và cho chạy tuần tự các cell
Xem kết quả trong thư mục:
```bash
results/
```
### 🗺️ Lộ trình phát triển
✔️ Vòng 1

- Trích xuất khung xương

- Tính toán góc

- Tạo video phân tích

🔄 Vòng 2

- ML phân loại Good vs Bad Swing

- Xây dựng Swing Score

🏆 Chung kết

- Web app với Streamlit

- Tích hợp AI Coach đề xuất cải thiện kỹ thuật
