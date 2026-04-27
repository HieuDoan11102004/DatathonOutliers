import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def check_v4_data_leakage():
    print("=" * 60)
    print("🔍 KIỂM TRA DATA LEAKAGE TRONG V4 (DLINEAR)")
    print("=" * 60)
    
    v4_train_path = 'v4_dlinear_tree/train.csv'
    v4_test_path = 'v4_dlinear_tree/test.csv'
    
    if not os.path.exists(v4_test_path):
        print("❌ Không tìm thấy test.csv của V4.")
        return
        
    train_df = pd.read_csv(v4_train_path)
    test_df = pd.read_csv(v4_test_path)
    
    # 1. Kiểm tra rò rỉ qua dlinear_pred
    corr_test = test_df['Revenue'].corr(test_df['dlinear_pred'])
    corr_train = train_df['Revenue'].corr(train_df['dlinear_pred'])
    
    print(f"🔹 Tương quan (Pearson) giữa Revenue và dlinear_pred trên tập TRAIN: {corr_train:.4f}")
    print(f"🔹 Tương quan (Pearson) giữa Revenue và dlinear_pred trên tập TEST:  {corr_test:.4f}")
    
    if corr_test > 0.8:
        print("⚠️ CẢNH BÁO ĐỎ (DATA LEAKAGE): dlinear_pred có tương quan quá cao với Revenue trên tập Test.")
        print("   -> Lỗi logic trong code V4: DLinear đã được huấn luyện (fit) trên toàn bộ internal_df (bao gồm cả Test 2022),")
        print("   sau đó lại quay ngược lại sinh ra dlinear_pred cho Tree Ensemble dự đoán Test.")
        print("   Điều này khiến Tree Ensemble 'nhìn trộm' được tương lai ở tập Test nhưng ra thực tế (Submission) sẽ sụp đổ.")
    else:
        print("✅ Không phát hiện rò rỉ dữ liệu rõ ràng qua tương quan.")
        
    # 2. Kiểm tra rò rỉ mục tiêu chéo
    if 'log_revenue' in test_df.columns:
        print("⚠️ CẢNH BÁO (LEAKAGE): Cột 'log_revenue' đang nằm chung trong bộ tính năng (cần bị drop).")

def compare_v4_v6_submissions():
    print("\n" + "=" * 60)
    print("📊 SO SÁNH PREDICTIONS: V4 (DLinear) vs V6 (LRU)")
    print("=" * 60)
    
    v4_sub = 'v4_dlinear_tree/submission_v4.csv'
    v6_sub = 'v6_lru_tree/submission.csv' # Assuming defaults
    
    if not os.path.exists(v4_sub) or not os.path.exists(v6_sub):
        print(f"❌ Cần có đủ 2 file submission để so sánh: {v4_sub} và {v6_sub}")
        return
        
    df_v4 = pd.read_csv(v4_sub, parse_dates=['Date'])
    df_v6 = pd.read_csv(v6_sub, parse_dates=['Date']) # Hoặc submission_v6.csv nếu có đổi tên
    
    v4_rev = df_v4['Revenue']
    v6_rev = df_v6['Revenue']
    
    diff = np.abs(v4_rev - v6_rev)
    mae_diff = diff.mean()
    
    print(f"🔹 Doanh thu trung bình dự đoán - V4: {v4_rev.mean():,.0f}")
    print(f"🔹 Doanh thu trung bình dự đoán - V6: {v6_rev.mean():,.0f}")
    print(f"🔹 Độ lệch trung bình (MAE) giữa 2 bản: {mae_diff:,.0f}")
    
    corr = v4_rev.corr(v6_rev)
    print(f"🔹 Mức độ tương quan dự báo: {corr:.4f}")
    
    if corr < 0.5:
        print("⚠️ Hai mô hình cho kết quả dự báo hoàn toàn khác nhau về xu hướng!")
    else:
        print("✅ Hai mô hình dự báo tương đồng về xu hướng nhưng khác biệt về biên độ vi mô.")

if __name__ == '__main__':
    check_v4_data_leakage()
    compare_v4_v6_submissions()
