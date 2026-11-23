# ⚽ Football Analytics & Big Data Platform (SofaScore, FotMob, Opta Analyst & Understat Pro)

Nền tảng phân tích bóng đá chuyên sâu, nghiên cứu trước trận và trực quan hóa dữ liệu lớn (Big Data Analytics), phát triển dựa trên **Streamlit**, **Apache Spark Cluster (Docker)**, **SQLite**, **REST API Microservice**, mô hình dự đoán **Opta/Panna Deep ML + Dixon-Coles Bivariate Poisson + Monte Carlo**, tích hợp kho dữ liệu khổng lồ hơn **7.120+ trận đấu** (**5.190+ trận đã đá full tọa độ sự kiện ~9.6M sự kiện** + **1.928 trận chưa diễn ra mùa 2026/27** + **60.012 trận lịch sử có xG & Elo** + **Đồng bộ Live ESPN Real-Time**).

---

## 🌟 Điểm nổi bật & Tính năng cốt lõi (Phiên bản v3.0 Pro)

1. **🔮 Trung Tâm Nghiên Cứu & Dự Đoán Trận Đấu Chưa Diễn Ra (Pre-Match Research Lab):**
   - **1.928 trận chưa diễn ra mùa 2026/27** trên 13 giải đấu lớn (Premier League, La Liga, Serie A, Bundesliga, Ligue 1, UCL, UEL, Conference League, Championship, Liga Portugal, Eredivisie, Scottish Premiership, Süper Lig).
   - **Mô hình AI Ensemble Đa Tầng**: Kết hợp **Opta/Panna Deep ML (110 đặc trưng đội hình & Elo)** + **Dixon-Coles Bivariate Poisson (Phong độ xG gần nhất)** + **Kèo Thị Trường Live (DraftKings/ESPN)** ➔ Tự động tính **Value Edge % (Chênh lệch AI vs Nhà cái)**.
   - **Đầy đủ thị trường xác suất**: Ma trận tỷ số 6×6 Dixon-Coles, Top 6 tỷ số sáng nhất, Tài/Xỉu (1.5 / 2.5 / 3.5), BTTS (Hai đội cùng ghi bàn), Cơ hội kép (1X/X2/12), Clean Sheet %, Kèo chấp (-1.5), Phân phối tổng bàn thắng.
   - **So sánh sức mạnh 4 tuyến (GK / DEF / MID / FWD — Panna RAPM)** & **Biểu đồ Radar Chiến Thuật 8 Trục** + Nhận định chuyên gia tự động bằng Tiếng Việt.
   - **Phong độ 10 trận gần nhất & xG Regression**: Phát hiện đội đang ghi bàn vượt xG (Overperforming) hay kém may mắn (Underperforming), tách riêng hiệu suất Sân Nhà vs Sân Khách + **Lịch sử đối đầu (H2H)** có xG & tỷ số hiệp 1 + **Vua phá lưới/Kiến tạo & Tin tức Live ESPN**.
   - **Phòng thí nghiệm giả lập kịch bản "What-If"**: Điều chỉnh chấn thương/treo giò, sân trung lập, mưa lớn, nhịp độ trận đấu và mô phỏng lại tức thì 20.000 kịch bản Monte Carlo.
   - **Bộ Quét Vòng Đấu (Matchday Radar Scanner)**: Tự động xếp hạng trận tâm điểm (High Elo), trận mưa bàn thắng (High xG/O2.5), cửa thắng chắc nhất (Bankers), cảnh báo bất ngờ & Value Edges.

2. **🏟️ Trung tâm trận đấu (Match Analytics Center — 5.190+ trận sự kiện đến T11/2026):**
   - Bao phủ **World Cup 2026 trọn bộ 104 trận**, **World Cup 2022 & 2018**, **Top 5 Giải VĐQG Châu Âu 2025/26 – 2026/27** (Premier League, La Liga, Serie A, Bundesliga, Ligue 1), **UEFA Champions League (~1.846 trận)**, **UEFA Europa League (2025/26–2026/27)**, **Euro 2024/2020**, **Copa América 2024**, **AFCON 2023**.
   - Đội hình ra sân 2D (Interactive Pitch 2D) với 25 vị trí chiến thuật, điểm số SofaScore và ảnh cầu thủ.
   - Chỉ số chuyên sâu hiện đại: **Field Tilt % (Độ nghiêng sân 1/3 cuối)**, **PPDA (Cường độ Pressing)**, **Chạm bóng vòng cấm (Box Touches)**, **xG/Shot**, **Shot Map 2D**, **xG Momentum Race**, **Player Heatmap**, **Passing Network**, **Attack Momentum Wave**.

3. **🔴 Trung tâm trực tiếp & Đa trận (Live Match Center):**
   - Giả lập diễn biến trận đấu tự động chạy (Auto-run) với bình luận trực tiếp (Live Commentary Feed).
   - Chế độ xem song song nhiều trận cùng lúc (Multi-Match Split Screen 2–4 trận).

4. **🏆 Bảng Xếp Hạng Thực Tế & Điểm Kỳ Vọng xPTS (Understat / FotMob Style):**
   - **18 bảng xếp hạng đa giải đấu** bao gồm toàn bộ **Top 5 Châu Âu 2026/27 & 2025/26, UCL 2026/27, UEL 2026/27, World Cup 2026/2022/2018, Euro 2024/2020, Copa América 2024, AFCON 2023**.
   - Bảng xếp hạng **Expected Points (xPTS)** + **Biểu đồ Tứ Phân Vị Chiến Thuật (xG/trận vs xGA/trận Quadrant)**.
   - Lịch thi đấu tích hợp cả **Trận sắp diễn ra (có xác suất AI & nút Phân tích sâu)** và **Trận đã kết thúc**.
   - Cơ sở dữ liệu chuyển nhượng & chỉ số cầu thủ FIFA (17.9k+ cầu thủ).

5. **⚡ Kiến trúc Big Data & REST API (Port 8502):**
   - Backend REST API Microservice cung cấp **7 live JSON endpoints** (`/api/health`, `/api/upcoming`, `/api/xpts`, `/api/matches`, `/api/standings`, `/api/players/top`, `/api/predict`).
   - Cụm Apache Spark Master & 2 Workers điều phối qua Docker Compose + CSDL SQLite lưu tài khoản, yêu thích & dự đoán cá nhân.

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

1. **Xây dựng kho hiện đại 2025/26 – 2026/27, World Cup 2026 & 1.928 Trận Chưa Diễn Ra + Live ESPN:**
```bash
python build_pro_analytics_store.py
```

2. **Tải toàn bộ dữ liệu StatsBomb Open Data (586 trận):**
```bash
python fetch_statsbomb_data.py
```

3. **Xây dựng kho Opta UEFA Champions League Parquet (1.828 trận):**
```bash
python build_opta_ucl_store.py
```

---

## 👥 Bản quyền & Dữ liệu tham khảo
- Dữ liệu sự kiện mở từ **StatsBomb Open Data** và **Opta Data** (`peteowen1/pannadata`).
- Giao diện và thuật toán phân tích được thiết kế bởi nhóm phát triển.
