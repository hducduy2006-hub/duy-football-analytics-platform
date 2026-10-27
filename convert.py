import os
import pantab

# Đường dẫn tới thư mục chứa các file .hyper
hyper_dir = r"C:\Users\Admin\Downloads\Edd Webster - World Cup 2018 Analysis and Dashboard\Data\tableau-temp"
out_dir = r"C:\Users\Admin\hdfs-practice"

for idx, file_name in enumerate(os.listdir(hyper_dir)):
    if file_name.endswith(".hyper"):
        full_path = os.path.join(hyper_dir, file_name)
        print(f"Đang xử lý: {file_name}...")
        try:
            frames = pantab.frames_from_hyper(full_path)
            for t_idx, (table_name, df) in enumerate(frames.items()):
                # Lấy tên bảng chuẩn từ tuple
                t_str = table_name[-1] if isinstance(table_name, (tuple, list)) else str(table_name)
                clean_tname = str(t_str).replace('"', '').replace(' ', '_').replace("'", "")
                csv_path = os.path.join(out_dir, f"worldcup_{idx}_{clean_tname}.csv")
                df.to_csv(csv_path, index=False)
                print(f"✅ Đã xuất thành công: {csv_path} ({len(df)} dòng)")
        except Exception as e:
            print(f"Lỗi: {e}")

print("🎉 Hoàn tất chuyển đổi tất cả file Hyper sang CSV!")