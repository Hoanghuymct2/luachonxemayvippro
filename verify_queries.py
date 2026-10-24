"""
Script to verify real benchmark queries and check constraints.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
from recommender import MotoRecommender, PURPOSE_WEIGHTS

rec = MotoRecommender("dataset.csv")
print("=== VERIFICATION RUN ON REAL DATASET ===")
stats = rec.get_stats()
print(f"Total rows: {stats['total_rows']}")
print(f"Total feature dimensions: {stats['feature_dims']}")

# Query 1: Sinh viên có ngân sách giới hạn <= 45 triệu
print("\n--- QUERY 1: Sinh viên (Budget <= 45M, Học sinh, Sinh viên) ---")
q1 = {"Giá(VND)": 45_000_000.0, "Kiểu xe": "Tay ga"}
res1 = rec.recommend(q1, weights=PURPOSE_WEIGHTS["Học sinh, Sinh viên"], k=5)
for _, r in res1.iterrows():
    p_str = f"{int(r['Giá(VND)']):,} đ".replace(",", ".")
    print(f"Rank {r['Thứ hạng']}: {r['Tên xe']} | Giá: {p_str} | Kiểu: {r['Kiểu xe']} | Cốp: {r['Dung tích cốp (L)']}L | Điểm: {r['Độ phù hợp (%)']}% (d={r['Khoảng cách KNN']})")
assert (res1["Giá(VND)"] <= 45_000_000.0).all(), "Lỗi: Có xe vượt ngân sách 45M!"
print("-> Xác nhận lọc ngân sách: PASS. Tất cả các xe đều <= 45.000.000 VNĐ.")

# Query 2: Người đi làm cần cốp rộng (Đi làm, đi phố & Gia đình, cốp 28L, budget <= 70M)
print("\n--- QUERY 2: Người đi làm (Budget <= 70M, Cốp 28L, Đi làm, đi phố & Gia đình) ---")
q2 = {"Giá(VND)": 70_000_000.0, "Dung tích cốp (L)": 28.0, "Kiểu xe": "Tay ga"}
res2 = rec.recommend(q2, weights=PURPOSE_WEIGHTS["Đi làm, đi phố & Gia đình"], k=5)
for _, r in res2.iterrows():
    p_str = f"{int(r['Giá(VND)']):,} đ".replace(",", ".")
    print(f"Rank {r['Thứ hạng']}: {r['Tên xe']} | Giá: {p_str} | Kiểu: {r['Kiểu xe']} | Cốp: {r['Dung tích cốp (L)']}L | Điểm: {r['Độ phù hợp (%)']}% (d={r['Khoảng cách KNN']})")
assert (res2["Giá(VND)"] <= 70_000_000.0).all(), "Lỗi: Có xe vượt ngân sách 70M!"
print("-> Xác nhận lọc ngân sách: PASS. Tất cả các xe đều <= 70.000.000 VNĐ.")

# Query 3: Người quan tâm xe điện hoặc công suất cao
print("\n--- QUERY 3: Xe điện hoặc Công suất cao (Nhiên liệu: Điện, CS: 10kW, Đi phượt, Thể thao & Đam mê tốc độ) ---")
q3 = {"Nhiên liệu": "Điện", "Công suất tối đa (Kw)": 10.0}
res3 = rec.recommend(q3, weights=PURPOSE_WEIGHTS["Đi phượt, Thể thao & Đam mê tốc độ"], k=5)
for _, r in res3.iterrows():
    p_str = f"{int(r['Giá(VND)']):,} đ".replace(",", ".")
    print(f"Rank {r['Thứ hạng']}: {r['Tên xe']} | Nhiên liệu: {r['Nhiên liệu']} | CS: {r['Công suất tối đa (Kw)']} kW | Giá: {p_str} | Điểm: {r['Độ phù hợp (%)']}% (d={r['Khoảng cách KNN']})")

print("\n=== TẤT CẢ KIỂM CHỨNG TRUY VẤN HOÀN TOÀN THÀNH CÔNG ===")

