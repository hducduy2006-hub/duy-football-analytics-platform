from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, count, round

# 1. Khởi tạo SparkSession
spark = SparkSession.builder \
    .appName("Shopee_Data_Processing") \
    .getOrCreate()

print("\n==================================================")
print("🚀 ĐANG ĐỌC VÀ XỬ LÝ DỮ LIỆU ĐƠN HÀNG SHOPEE")
print("==================================================\n")

# 2. Đọc file CSV từ đường dẫn trong container
df = spark.read.csv("/opt/spark-data/don_hang_shopee.csv", header=True, inferSchema=True)

print("--- Cấu trúc dữ liệu (Schema) ---")
df.printSchema()

print("\n--- 5 Dòng dữ liệu mẫu ---")
df.show(5)

# 3. Tính tổng số lượng đơn và tổng tiền theo trạng thái
# (Code tự động nhận diện tên cột trong file của bạn)
cols = [c.lower() for c in df.columns]

# Tìm tên cột trạng thái và cột tiền
status_col = next((c for c in df.columns if "trang_thai" in c.lower() or "status" in c.lower()), df.columns[1] if len(df.columns) > 1 else None)
price_col = next((c for c in df.columns if "tien" in c.lower() or "price" in c.lower() or "amount" in c.lower() or "val" in c.lower()), df.columns[2] if len(df.columns) > 2 else None)

if status_col:
    print(f"\n---> Đang thực hiện GroupBy theo cột: '{status_col}'...")
    if price_col:
        result = df.groupBy(status_col) \
            .agg(
                count("*").alias("so_luong_don"),
                round(sum(price_col), 2).alias("tong_gia_tri")
            )
    else:
        result = df.groupBy(status_col).count()

    print("\n==================================================")
    print("📊 KẾT QUẢ TỔNG HỢP")
    print("==================================================")
    result.show()

    # 4. Ghi kết quả ra thư mục output
    output_path = "/opt/spark-data/output_shopee_result"
    result.write.mode("overwrite").option("header", "true").csv(output_path)
    print(f"\n✅ Đã ghi kết quả thành công vào: {output_path}")
# 5. Tạm dừng 120 giây để thong thả vào Spark UI (http://localhost:4040) xem DAG/Stages
    import time
    print("\n⏳ Đang tạm dừng 120 giây... Hãy mở ngay http://localhost:4040 trên trình duyệt!")
    time.sleep(120)

spark.stop()
