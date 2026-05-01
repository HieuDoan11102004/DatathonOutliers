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

## Script chính

Các script có thể chạy chính nằm trong `timesfm/`:

- `timesfm/ensemble_forecast.py` - baseline mùa vụ kết hợp ensemble residual XGBoost
- `timesfm/ensemble_forecast_v2.py` - bổ sung feature bên ngoài cho mô hình residual
- `timesfm/ensemble_forecast_v3.py` - biến thể baseline mùa vụ với lọc mạnh hơn cho các năm COVID
- `timesfm/timesfm_forecast.py` - sinh submission zero-shot dựa trên TimesFM
- `timesfm/chronos_full_validation.py` - kiểm định Chronos và trộn submission
- `timesfm/dlinear_forecast.py` - mô hình dự báo DLinear
- `timesfm/ttm_forecast.py` - tiện ích dự báo TinyTimeMixer
- `timesfm/meta_ensemble_search.py` - tìm kiếm meta-ensemble và tạo output kiểm định

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

## Cách chạy

Các script dưới đây nên được chạy từ thư mục gốc của repo vì chúng dùng đường dẫn tương đối:

### 1. Chạy baseline ensemble

```bash
python timesfm/ensemble_forecast.py
```

Script này sẽ:

- huấn luyện baseline mùa vụ + residual XGBoost
- in kết quả walk-forward validation
- tạo file `submission_ensemble.csv`

### 2. Chạy ensemble có external features

```bash
python timesfm/ensemble_forecast_v2.py
```

Script này bổ sung các feature từ đơn hàng, thanh toán, vận chuyển, trả hàng, đánh giá, khuyến mãi và web traffic.

### 3. Chạy biến thể baseline mùa vụ v3

```bash
python timesfm/ensemble_forecast_v3.py
```

Biến thể này dùng cấu hình lọc mạnh hơn cho các năm COVID và phục vụ thử nghiệm decay search.

### 4. Chạy TimesFM zero-shot

```bash
python timesfm/timesfm_forecast.py
```

Script này tải checkpoint TimesFM và sinh submission zero-shot.

### 5. Chạy Chronos validation và blend

```bash
python timesfm/chronos_full_validation.py
```

Script này so sánh Chronos với baseline và tạo file submission blend tốt nhất.

### 6. Chạy DLinear

```bash
python timesfm/dlinear_forecast.py
```

### 7. Chạy TinyTimeMixer

```bash
python timesfm/ttm_forecast.py
```

### 8. Chạy meta-ensemble search

```bash
python timesfm/meta_ensemble_search.py
```

Script này tổng hợp nhiều mô hình và tạo các file kiểm định trong `timesfm/visual_outputs/`.

### Biến môi trường hữu ích

Các biến môi trường mà một số script sử dụng:

- `RESIDUAL_WEIGHT` - trọng số trộn residual cho các script ensemble
- `TIMESFM_BACKEND` - `gpu` hoặc `cpu`
- `TIMESFM_REPO` - ID repo Hugging Face cho checkpoint TimesFM
- `TIMESFM_LOCAL_DIR` - thư mục cache cục bộ cho weight mô hình
- `TIMESFM_CONTEXT_LEN` - độ dài context cho TimesFM
- `TIMESFM_OUTPUT` - đường dẫn output cho file submission TimesFM

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

Repo hiện đã có sẵn một số file submission và artifact kiểm định đã sinh, bao gồm:

- `submission.csv`
- `timesfm/submission*.csv`
- `timesfm/submission_best_*.csv`
- `timesfm/visual_outputs/*.csv`

Các file này ghi lại những biến thể mô hình khác nhau và kết quả tìm kiếm trong quy trình dự báo.

## Ghi chú

- Dự án được tổ chức quanh một bài toán dự báo theo kiểu cuộc thi, không phải một ứng dụng đa năng.
- Phần lớn script yêu cầu các file CSV của cuộc thi phải có sẵn ở thư mục gốc của repository.
- `check_versions.py` là script tiện ích để so sánh các phiên bản mô hình và kiểm tra rò rỉ dữ liệu giữa các biến thể dự báo.
