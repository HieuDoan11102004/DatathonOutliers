# DatathonOutliers

Dự án dự báo cho thử thách Datathon 2026 "The Gridbreakers".
Kho lưu trữ tập trung vào việc dự đoán hằng ngày `Revenue` và `COGS` cho một doanh nghiệp thương mại điện tử bán lẻ, dựa trên sự kết hợp giữa dữ liệu bán hàng lịch sử, các bảng giao dịch và các tín hiệu vận hành.

## Tổng quan

Bộ dữ liệu gồm nhiều nhóm file CSV:

- Dữ liệu master: sản phẩm, khách hàng, khuyến mãi, địa lý
- Dữ liệu giao dịch: đơn hàng, chi tiết đơn hàng, thanh toán, vận chuyển, trả hàng, đánh giá
- Dữ liệu phân tích: mục tiêu doanh thu theo ngày và định dạng nộp bài
- Dữ liệu vận hành: ảnh chụp tồn kho và lưu lượng web

Mục tiêu chính của dự án là dự báo doanh số theo ngày. Phần lớn các thử nghiệm đều tạo file nộp theo đúng định dạng `sample_submission.csv`, với các cột:

- `Date`
- `Revenue`
- `COGS`

## Có gì trong repo

- `EDA1.ipynb`, `EDA2.ipynb` - các notebook khám phá dữ liệu
- `baseline.ipynb` - phần xây dựng mô hình baseline
- `timesfm/` - các thử nghiệm mô hình, tiện ích kiểm định và các output đã sinh
- `submission*.csv` - các file nộp bài đã lưu
- `visual_outputs/` bên trong `timesfm/` - bảng kiểm định, feature importance, SHAP outputs và chẩn đoán mô hình
- Các file `*.csv` ở thư mục gốc - dữ liệu cuộc thi được dùng bởi notebook và script

## Hướng tiếp cận mô hình

Dự án kết hợp nhiều họ mô hình dự báo:

- Baseline mùa vụ có trọng số theo độ mới của dữ liệu
- Mô hình residual XGBoost trên các đặc trưng lịch và nghiệp vụ đã được thiết kế
- Feature engineering từ đơn hàng, thanh toán, vận chuyển, trả hàng, đánh giá, khuyến mãi và lưu lượng web
- Dự báo bằng các foundation model như TimesFM, Chronos và TTM
- Thử nghiệm DLinear cho dự báo chuỗi thời gian gọn nhẹ
- Các script meta-ensemble và tìm kiếm trọng số để so sánh hoặc trộn các mô hình ứng viên

## Script chính để ra submission cuối

Luồng chính để tạo file nộp cuối cùng gồm đúng 3 script trong `timesfm/`:

1. `ratio_tuned_selection_v2.py`
2. `meta_ensemble_search.py`
3. `timesfm_tuned_search.py`

Các script khác trong repo là thử nghiệm bổ sung, không bắt buộc trong luồng nộp bài cuối.

## Thiết lập môi trường

1. Đảm bảo đang ở thư mục gốc của repository.
2. Tạo môi trường ảo nếu muốn tách biệt thư viện:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Cài đặt các thư viện Python:

```bash
pip install -r requirements.txt
```

4. Đảm bảo các file dữ liệu của cuộc thi đã nằm ở thư mục gốc, đặc biệt là:

- `sales.csv`
- `sample_submission.csv`
- các file CSV master/giao dịch/vận hành khác

Một số script sử dụng các mô hình pretrained lớn từ Hugging Face và có thể cần:

- GPU để chạy trong thời gian hợp lý
- kết nối mạng ở lần chạy đầu tiên để tải weight mô hình
- đủ dung lượng đĩa cho cache mô hình cục bộ

## Cách chạy luồng 3 bước (khuyến nghị)

Chạy lệnh từ thư mục `timesfm` để tránh lỗi đường dẫn tương đối:

```bash
cd timesfm
```

### Bước 1 - Ratio tuned selection v2

```bash
python ratio_tuned_selection_v2.py
```

Output chính:

- `visual_outputs/ratio_tuned_selection_v2_folds.csv`
- `visual_outputs/ratio_tuned_selection_v2_leaderboard.csv`
- `submission_best_ratio_tuned_v2.csv`

### Bước 2 - Meta ensemble search

```bash
python meta_ensemble_search.py
```

Output chính:

- `visual_outputs/meta_ensemble_validation_predictions.csv`
- `visual_outputs/meta_ensemble_weights.csv`
- `visual_outputs/meta_ensemble_fold_predictions.csv`
- `visual_outputs/meta_ensemble_folds.csv`
- `visual_outputs/meta_ensemble_scores.csv`
- `submission_best_meta_ensemble.csv`

### Bước 3 - TimesFM tuned search (tạo submission cuối)

```bash
python timesfm_tuned_search.py
```

Output chính:

- `visual_outputs/timesfm_tuned_folds.csv`
- `visual_outputs/timesfm_tuned_scores.csv`
- `visual_outputs/timesfm_tuned_validation_predictions.csv`
- `submission_best_timesfm_tuned.csv` (file submission cuối)
- nếu lỗi: `visual_outputs/timesfm_tuned_error.txt`

Tóm tắt luồng phụ thuộc:

- Bước 2 dùng kết quả từ bước 1.
- Bước 3 dùng meta submission từ bước 2 làm anchor.

### Biến môi trường hữu ích

Các biến môi trường mà một số script sử dụng:

- `RESIDUAL_WEIGHT` - trọng số trộn residual cho các script ensemble
- `TIMESFM_BACKEND` - `gpu` hoặc `cpu`
- `TIMESFM_REPO` - ID repo Hugging Face cho checkpoint TimesFM
- `TIMESFM_LOCAL_DIR` - thư mục cache cục bộ cho weight mô hình
- `TIMESFM_CONTEXT_LEN` - độ dài context cho TimesFM
- `TIMESFM_OUTPUT` - đường dẫn output cho file submission TimesFM
- `TIMESFM_TUNED_SCENARIOS` - danh sách scenario phân tách bằng dấu phẩy để giới hạn fold kiểm định
- `TIMESFM_TUNED_MAX_FOLDS` - giới hạn số fold được đánh giá
- `TIMESFM_TUNED_MAX_HORIZON` - giới hạn horizon tối đa được dùng trong tuned search

## Làm việc với notebook

Các notebook có thể được mở bằng Jupyter nếu bạn muốn xem EDA hoặc thử nghiệm thủ công:

```bash
jupyter lab
```

hoặc

```bash
jupyter notebook
```

## Output

Các file submission quan trọng trong luồng chính:

- `timesfm/submission_best_ratio_tuned_v2.csv`
- `timesfm/submission_best_meta_ensemble.csv`
- `timesfm/submission_best_timesfm_tuned.csv` (nộp cuối)

Các bảng kiểm định và leaderboard nằm trong `timesfm/visual_outputs/*.csv`.

## Ghi chú

- Dự án được tổ chức quanh một bài toán dự báo theo kiểu cuộc thi, không phải một ứng dụng đa năng.
- Phần lớn script yêu cầu các file CSV của cuộc thi phải có sẵn ở thư mục gốc của repository.
- `check_versions.py` là script tiện ích để so sánh các phiên bản mô hình và kiểm tra rò rỉ dữ liệu giữa các biến thể dự báo.
