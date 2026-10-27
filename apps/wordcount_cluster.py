"""
Chạy job này bằng spark-submit (không phải python3 thường),
vì cần kết nối tới Spark Master thật qua giao thức spark://

Cách chạy (trong terminal của container spark-client):
    spark-submit --master spark://spark-master:7077 /opt/spark-apps/wordcount_cluster.py
"""
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("WordCountCluster") \
    .getOrCreate()   # master đã được truyền qua --master khi spark-submit

sc = spark.sparkContext

# Ghi dữ liệu mẫu ra thư mục dùng chung (mount sẵn với các Worker)
text = """spark chay tren cluster that
co master va nhieu worker
du lieu duoc phan tan tren cac worker
day la vi du thuc hanh spark cluster"""

with open("/opt/spark-data/sample.txt", "w") as f:
    f.write(text)

rdd = sc.textFile("/opt/spark-data/sample.txt")
counts = rdd.flatMap(lambda line: line.split()) \
            .map(lambda w: (w, 1)) \
            .reduceByKey(lambda a, b: a + b) \
            .sortBy(lambda x: -x[1])

print("\n=== KẾT QUẢ TỪ CLUSTER THẬT (Master + 2 Worker) ===")
for word, count in counts.collect():
    print(f"{word}: {count}")

# In thông tin cluster để thấy job thật sự chạy trên nhiều Executor
print(f"\nSố Executor đang hoạt động: {len(sc._jsc.sc().statusTracker().getExecutorInfos())}")

spark.stop()
