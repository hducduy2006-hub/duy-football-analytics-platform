# ⚽ Football Analytics & Big Data Platform (SofaScore & FotMob Alternative)

Nền tảng phân tích bóng đá chuyên sâu và trực quan hóa dữ liệu lớn (Big Data Analytics), phát triển dựa trên **Streamlit**, **Apache Spark Cluster (Docker)**, **SQLite**, **REST API Microservice**, mô hình dự đoán **Monte Carlo & Phân phối Poisson**, tích hợp kho dữ liệu đồ sộ hơn **2.479 trận đấu** từ **StatsBomb Open Data** và **Opta UEFA Champions League Parquet**.

---

## 🌟 Điểm nổi bật & Tính năng cốt lõi (45/45 tiêu chí so với SofaScore, FotMob, AiScore)

1. **🏟️ Trung tâm trận đấu (Match Analytics Center):**
   - Đội hình ra sân 2D (Interactive Pitch 2D) với đầy đủ 25 vị trí chiến thuật thực tế và ảnh chân dung cầu thủ.
   - Bản đồ dứt điểm 2D Shot Map (tính toán cự ly, góc sút, kích thước điểm theo chỉ số bàn thắng kỳ vọng xG).
   - Biểu đồ đường đua xG theo phút (FotMob xG Momentum Race).
   - Bản đồ nhiệt chạm bóng cầu thủ (Player Touch Heatmap 2D).
   - Mạng lưới chuyền bóng chiến thuật (Passing Network) với độ dày đường chuyền và vị trí trung bình.
   - Sóng áp lực thế trận theo phút (SofaScore Attack Momentum).

2. **🔴 Trung tâm trực tiếp & Đa trận (Live Match Center):**
   - Giả lập diễn biến trận đấu tự động chạy (Auto-run) với bình luận trực tiếp (Live Commentary Feed).
   - Chế độ xem song song nhiều trận cùng lúc (Multi-Match Split Screen 2–4 trận).

3. **🏆 Bảng xếp hạng & Trung tâm giải đấu:**
   - Hỗ trợ xem Bảng xếp hạng của **8 giải đấu hoàn chỉnh**:
     - 🏆 FIFA World Cup 2022 (Bảng A–H)
     - ⚽ UEFA Champions League 2024/25 & 2025/26 (League Phase 36 CLB)
     - 🏆 FIFA World Cup 2018 (Bảng A–H)
     - 🏆 UEFA Euro 2024 & Euro 2020 (Bảng A–F)
     - 🏆 CONMEBOL Copa América 2024 (Bảng A–D)
     - 🌍 Africa Cup of Nations 2023 (Bảng A–F)
   - Lịch thi đấu & kết quả lọc thông minh theo giải và mùa giải.
   - Cơ sở dữ liệu chuyển nhượng & chỉ số cầu thủ FIFA (17.9k+ cầu thủ).

4. **🤖 AI Dự đoán, So sánh & Xuất dữ liệu:**
   - Thuật toán mô phỏng 10.000 kịch bản trận đấu Monte Carlo và ma trận xác suất Poisson 6×6.
   - So sánh trực tiếp 2 cầu thủ với biểu đồ Radar kỹ năng 6 trục.
   - Xuất dữ liệu báo cáo trận đấu chuẩn HTML In / Tải về.

5. **⚡ Kiến trúc Big Data & REST API:**
   - Backend REST API Microservice chạy trên cổng `8502` cung cấp 5 endpoints JSON.
   - Cụm Apache Spark Master & 2 Workers điều phối qua Docker Compose.
   - Cơ sở dữ liệu SQLite quản lý tài khoản người dùng, danh sách yêu thích và lịch sử dự đoán.

---

## 📁 Cấu trúc thư mục

```text
├── app.py                      # Ứng dụng chính Streamlit (Giao diện SofaScore Dark Theme)
├── pro_features.py             # Module tính năng mở rộng: Live Replay, AI Monte Carlo, Radar, BXH...
├── api_server.py               # REST API Microservice (Port 8502)
├── build_opta_ucl_store.py     # Script xử lý & tối ưu 1.828 trận Opta Champions League Parquet
├── fetch_statsbomb_data.py     # Script tải đa luồng 586 trận StatsBomb Open Data
├── docker-compose.yml          # Cấu hình cụm Apache Spark (spark-master, worker-1, worker-2)
├── fifa18_complete_dataset.csv # Dữ liệu chỉ số & định giá cầu thủ FIFA
├── worldcup_2018_shots.csv     # Dữ liệu tọa độ dứt điểm chi tiết World Cup 2018
├── worldcup_match_stats.csv    # Thống kê tổng hợp trận đấu
└── sample_data/                # Kho dữ liệu mẫu & danh mục trận đấu
```

---

## 🚀 Hướng dẫn cài đặt & Chạy ứng dụng

### 1. Yêu cầu hệ thống
- Python 3.10+
- (Tùy chọn) Docker Desktop nếu muốn chạy cụm Apache Spark

### 2. Cài đặt thư viện Python
```bash
pip install streamlit pandas numpy plotly pyarrow requests
```

### 3. Khởi động ứng dụng

- **Chạy giao diện người dùng (Streamlit):**
```bash
streamlit run app.py --server.port 8501
```
Truy cập: [http://localhost:8501](http://localhost:8501)

- **Chạy REST API Server:**
```bash
python api_server.py
```
Truy cập: [http://localhost:8502/api/health](http://localhost:8502/api/health)

- **(Tùy chọn) Khởi động cụm Apache Spark với Docker:**
```bash
docker compose up -d
```
Xem Spark Master UI tại: [http://localhost:8080](http://localhost:8080)

---

## 📥 Tải thêm dữ liệu sự kiện mở rộng (Tùy chọn)

Nếu bạn vừa clone repository mới về và muốn nạp lại toàn bộ kho dữ liệu mở rộng:

1. **Tải toàn bộ dữ liệu StatsBomb Open Data (586 trận):**
```bash
python fetch_statsbomb_data.py
```

2. **Xây dựng kho Opta UEFA Champions League Parquet (1.828 trận):**
```bash
python build_opta_ucl_store.py
```

---

## 👥 Bản quyền & Dữ liệu tham khảo
- Dữ liệu sự kiện mở từ **StatsBomb Open Data** và **Opta Data** (`peteowen1/pannadata`).
- Giao diện và thuật toán phân tích được thiết kế bởi nhóm phát triển.
