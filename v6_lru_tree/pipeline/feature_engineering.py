import os
import pandas as pd
import numpy as np
from pipeline.config import *


# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────

# Ngày Tết Nguyên Đán (âm lịch → dương lịch)
TET_DATES = {
    2012: '2012-01-23', 2013: '2013-02-10', 2014: '2014-01-31',
    2015: '2015-02-19', 2016: '2016-02-08', 2017: '2017-01-28',
    2018: '2018-02-16', 2019: '2019-02-05', 2020: '2020-01-25',
    2021: '2021-02-12', 2022: '2022-02-01', 2023: '2023-01-22',
    2024: '2024-02-10', 2025: '2025-01-29',
}

# Ngày Giỗ Tổ Hùng Vương (10 tháng 3 âm lịch → dương lịch)
HUNG_KING_DATES = {
    2012: '2012-03-31', 2013: '2013-04-19', 2014: '2014-04-09',
    2015: '2015-04-27', 2016: '2016-04-16', 2017: '2017-04-06',
    2018: '2018-04-25', 2019: '2019-04-14', 2020: '2020-04-02',
    2021: '2021-04-21', 2022: '2022-04-10', 2023: '2023-04-29',
    2024: '2024-04-18', 2025: '2025-04-07',
}

# Ngày lễ dương lịch cố định (month, day)
FIXED_HOLIDAYS = {
    'new_year':        (1,  1),
    'liberation_day':  (4, 30),   # 30/4
    'labor_day':       (5,  1),   # 1/5
    'national_day':    (9,  2),   # 2/9
    'xmas':            (12, 25),
}

# Ngày đôi / sale lớn (month, day)
DOUBLE_DAY_SALES = {
    '1_1':  (1,  1),
    '2_2':  (2,  2),
    '3_3':  (3,  3),
    '4_4':  (4,  4),
    '5_5':  (5,  5),
    '6_6':  (6,  6),
    '7_7':  (7,  7),
    '8_8':  (8,  8),
    '9_9':  (9,  9),
    '10_10':(10,10),
    '11_11':(11,11),
    '12_12':(12,12),
    'val':  (2, 14),   # Valentine
    'women':(3,  8),   # 8/3
    'mid_year':(6, 18),# mid-year sale
    'bf':   (11,29),   # Black Friday (last Fri Nov ≈ 29)
    'harb': (12, 12),  # 12/12
}

# Ngày bắt đầu tập train (để tính trend)
TRAIN_START = pd.Timestamp('2012-07-04')

# COVID breakpoints
COVID_START  = pd.Timestamp('2020-03-01')
COVID_PEAK   = pd.Timestamp('2021-07-01')   # lockdown nặng nhất VN
COVID_RECOVERY = pd.Timestamp('2021-11-01')

# Lag tối thiểu an toàn (test dài ~547 ngày)
MIN_SAFE_LAG = 548


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def _load_sales():
    df = pd.read_csv(SALES_FILE, parse_dates=['Date'])
    return df.sort_values('Date').reset_index(drop=True)


def load_sales():
    """Public alias — kept for backward compat with train.py."""
    return _load_sales()


def _proximity(series_date, target_ts, pre_days=30, post_days=14):
    """
    Tính khoảng cách đến một mốc thời gian quan trọng.
    Trả về (days_to, days_since, is_window).
    """
    diff = (series_date - target_ts).dt.days
    days_to    = np.where((diff < 0) & (diff >= -pre_days),  -diff, pre_days + 1)
    days_since = np.where((diff > 0) & (diff <= post_days), diff, post_days + 1)
    in_window  = ((diff >= -pre_days) & (diff <= post_days)).astype(int)
    return days_to, days_since, in_window


# ─────────────────────────────────────────────
#  GROUP 1: TIME ENCODINGS
# ─────────────────────────────────────────────

def add_time_features(df):
    """Lịch + cyclical encoding."""
    d = df['Date']
    df['year']         = d.dt.year
    df['month']        = d.dt.month
    df['day']          = d.dt.day
    df['day_of_week']  = d.dt.dayofweek          # 0=Mon
    df['day_of_year']  = d.dt.dayofyear
    df['week_of_year'] = d.dt.isocalendar().week.astype(int)
    df['quarter']      = d.dt.quarter
    df['is_weekend']   = (d.dt.dayofweek >= 5).astype(int)
    df['is_month_start']= (d.dt.day <= 3).astype(int)
    df['is_month_end']  = (d.dt.day >= 28).astype(int)

    # Mùa theo khí hậu VN
    season_map = {1:4, 2:1, 3:1, 4:1, 5:2, 6:2, 7:2, 8:3, 9:3, 10:3, 11:4, 12:4}
    df['season'] = df['month'].map(season_map)

    # Cyclical: month + day_of_week + week_of_year
    df['month_sin']      = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos']      = np.cos(2 * np.pi * df['month'] / 12)
    df['dow_sin']        = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos']        = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['woy_sin']        = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['woy_cos']        = np.cos(2 * np.pi * df['week_of_year'] / 52)
    df['quarter_sin']    = np.sin(2 * np.pi * df['quarter'] / 4)
    df['quarter_cos']    = np.cos(2 * np.pi * df['quarter'] / 4)
    df['day_sin']        = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos']        = np.cos(2 * np.pi * df['day'] / 31)

    return df


# ─────────────────────────────────────────────
#  GROUP 2: FOURIER SEASONALITY (FIXED)
# ─────────────────────────────────────────────

def add_fourier_features(df):
    """
    Fourier harmonics với base phù hợp từng chu kỳ:
    - Weekly  → dùng day_of_week  (0-6, chu kỳ = 7)
    - Monthly → dùng day          (1-31, chu kỳ = 30.4)
    - Yearly  → dùng day_of_year  (1-365, chu kỳ = 365.25)

    FIX so với version cũ: weekly không nên dùng day_of_year.
    """
    n_harmonics = 3   # tăng từ 2 lên 3 cho richer seasonality

    # Weekly (base = day_of_week, period = 7)
    dow = df['day_of_week']
    for k in range(1, n_harmonics + 1):
        df[f'f_week_sin_{k}'] = np.sin(2 * np.pi * k * dow / 7)
        df[f'f_week_cos_{k}'] = np.cos(2 * np.pi * k * dow / 7)

    # Monthly (base = day, period = 30.4)
    day = df['day']
    for k in range(1, n_harmonics + 1):
        df[f'f_month_sin_{k}'] = np.sin(2 * np.pi * k * day / 30.4)
        df[f'f_month_cos_{k}'] = np.cos(2 * np.pi * k * day / 30.4)

    # Yearly (base = day_of_year, period = 365.25)
    doy = df['day_of_year']
    for k in range(1, n_harmonics + 1):
        df[f'f_year_sin_{k}'] = np.sin(2 * np.pi * k * doy / 365.25)
        df[f'f_year_cos_{k}'] = np.cos(2 * np.pi * k * doy / 365.25)

    # Quarterly Fourier (hữu ích cho pattern theo quý)
    q_day = (df['month'] % 3) * 30.4 + day   # ngày trong quý
    for k in range(1, 3):
        df[f'f_quarter_sin_{k}'] = np.sin(2 * np.pi * k * q_day / 91.25)
        df[f'f_quarter_cos_{k}'] = np.cos(2 * np.pi * k * q_day / 91.25)

    return df


# ─────────────────────────────────────────────
#  GROUP 3: TẾT + NGÀY LỄ VN ĐẦY ĐỦ
# ─────────────────────────────────────────────

def add_event_features(df):
    """
    Tết, Giỗ Tổ, ngày lễ dương lịch, ngày đôi/sale,
    payday, quarter-end effects.
    """
    d = df['Date']

    # ── Tết Nguyên Đán ──────────────────────────────────────
    df['days_to_tet']    = 60
    df['days_since_tet'] = 60
    df['is_tet_week']    = 0
    df['is_tet_month']   = 0

    for year, tet_str in TET_DATES.items():
        tet_dt   = pd.Timestamp(tet_str)
        diff     = (d - tet_dt).dt.days
        pre_mask = (diff < 0)  & (diff >= -45)
        post_mask= (diff >= 0) & (diff <= 21)

        df.loc[pre_mask,  'days_to_tet']    = (-diff[pre_mask]).clip(0, 60)
        df.loc[post_mask, 'days_since_tet'] = diff[post_mask].clip(0, 60)
        df.loc[(diff >= -7) & (diff <= 7),  'is_tet_week']  = 1
        df.loc[(diff >= -30) & (diff <= 10),'is_tet_month'] = 1

    # Smooth proximity: e^(-d/scale) — mạnh hơn cho non-linear models
    df['tet_proximity'] = np.exp(-np.minimum(df['days_to_tet'],
                                              df['days_since_tet']) / 10.0)

    # ── Giỗ Tổ Hùng Vương ───────────────────────────────────
    df['is_hung_king'] = 0
    for year, hk_str in HUNG_KING_DATES.items():
        hk_dt = pd.Timestamp(hk_str)
        diff  = (d - hk_dt).dt.days
        df.loc[(diff >= -2) & (diff <= 2), 'is_hung_king'] = 1

    # ── Ngày lễ dương lịch cố định ──────────────────────────
    for name, (m, day_n) in FIXED_HOLIDAYS.items():
        df[f'is_{name}']          = ((df['month'] == m) & (df['day'] == day_n)).astype(int)
        df[f'days_to_{name}']     = np.abs(
            (d - d.apply(lambda x: pd.Timestamp(x.year, m, day_n))).dt.days
        ).clip(0, 14)
        df[f'pre_{name}_3d']      = (df[f'days_to_{name}'] <= 3).astype(int)

    # ── Ngày đôi / Flash sale ────────────────────────────────
    df['is_double_day'] = 0
    for name, (m, day_n) in DOUBLE_DAY_SALES.items():
        mask = (df['month'] == m) & (df['day'] == day_n)
        df.loc[mask, 'is_double_day'] = 1

    # Pre-sale window (2 ngày trước ngày đôi cũng có traffic cao)
    df['pre_double_day'] = 0
    for _, (m, day_n) in DOUBLE_DAY_SALES.items():
        for offset in [1, 2]:
            pre_d = day_n - offset
            if pre_d >= 1:
                mask = (df['month'] == m) & (df['day'] == pre_d)
                df.loc[mask, 'pre_double_day'] = 1

    # ── Payday effect ────────────────────────────────────────
    # Ngày lương: 1 và 15 hàng tháng
    # days_since_last_payday = khoảng cách đến kỳ lương gần nhất
    days_since_1st  = (df['day'] - 1).clip(0)
    days_since_15th = (df['day'] - 15).clip(0)
    df['days_since_payday'] = np.minimum(days_since_1st, days_since_15th).clip(0, 14)
    df['is_payday_week']    = (df['days_since_payday'] <= 3).astype(int)

    # ── Quarter-end / Year-end effects ───────────────────────
    df['days_to_quarter_end'] = (
        df['Date'].apply(lambda x: (
            pd.Timestamp(x.year, 3, 31) if x.month <= 3 else
            pd.Timestamp(x.year, 6, 30) if x.month <= 6 else
            pd.Timestamp(x.year, 9, 30) if x.month <= 9 else
            pd.Timestamp(x.year, 12, 31)
        ) - x).dt.days
    ).clip(0, 30)
    df['is_year_end_month']   = (df['month'] == 12).astype(int)
    df['is_year_start_month'] = (df['month'] == 1).astype(int)

    # ── Tuần của tháng (1–5) ─────────────────────────────────
    df['week_of_month'] = (df['day'] - 1) // 7 + 1

    return df


# ─────────────────────────────────────────────
#  GROUP 4 (NEW): TREND FEATURES
# ─────────────────────────────────────────────

def add_trend_features(df):
    """
    Encode long-term growth trend — THIẾU trong version cũ.
    Tree models không extrapolate trend tự nhiên nếu không encode rõ.
    """
    t = (df['Date'] - TRAIN_START).dt.days.astype(float)

    # Linear trend
    df['days_from_start'] = t
    df['trend_linear']    = t / 3652.0   # normalize về [0, 1]

    # Log trend (bắt growth chậm dần)
    df['trend_log']       = np.log1p(t)

    # Sqrt trend (trung gian)
    df['trend_sqrt']      = np.sqrt(t.clip(0))

    # ── COVID piecewise breakpoints ──────────────────────────
    # VN có 3 làn sóng rõ nét ảnh hưởng e-commerce:
    # 1. Pre-COVID (bình thường)
    # 2. Lockdown (demand drop với một số ngành, spike với online khác)
    # 3. Recovery + acceleration post-COVID
    t_covid  = (df['Date'] - COVID_START).dt.days.clip(0).astype(float)
    t_peak   = (df['Date'] - COVID_PEAK).dt.days.clip(0).astype(float)
    t_recov  = (df['Date'] - COVID_RECOVERY).dt.days.clip(0).astype(float)

    df['post_covid_days']     = t_covid
    df['post_lockdown_days']  = t_peak
    df['post_recovery_days']  = t_recov

    # Flags
    df['is_pre_covid']    = (df['Date'] <  COVID_START).astype(int)
    df['is_covid_period'] = ((df['Date'] >= COVID_START) &
                              (df['Date'] <  COVID_RECOVERY)).astype(int)
    df['is_post_covid']   = (df['Date'] >= COVID_RECOVERY).astype(int)

    # Interaction: trend × era (cho phép model fit slope khác nhau mỗi giai đoạn)
    df['trend_x_post_covid']   = df['trend_linear'] * df['is_post_covid']
    df['trend_x_covid_period'] = df['trend_linear'] * df['is_covid_period']

    return df


# ─────────────────────────────────────────────
#  GROUP 5 (ENHANCED): REVENUE + COGS LAGS
# ─────────────────────────────────────────────

def add_lag_features(df):
    """
    Safe lags (≥ MIN_SAFE_LAG ngày) + YoY ratios + COGS lags.
    Test period: 01/01/2023 → 01/07/2024 (~547 ngày)
    → Lag tối thiểu = 548 ngày để predict ngay ngày đầu tiên của test.
    """
    rev  = df['Revenue'] if 'Revenue' in df.columns else pd.Series(np.nan, index=df.index)
    cogs = df['COGS']    if 'COGS'    in df.columns else pd.Series(np.nan, index=df.index)

    # ── Raw revenue lags ─────────────────────────────────────
    lag_days = [548, 728, 912, 1092, 1456]   # 1.5y, 2y, 2.5y, 3y, 4y
    for lag in lag_days:
        df[f'rev_lag_{lag}'] = rev.shift(lag)

    # ── Raw COGS lags (stable signal, phản ánh mix sản phẩm) ─
    for lag in [548, 728, 912]:
        df[f'cogs_lag_{lag}'] = cogs.shift(lag)

    # ── Rolling statistics từ nhiều điểm gốc ────────────────
    for lag_base in [548, 728, 912]:
        shifted_rev  = rev.shift(lag_base)
        shifted_cogs = cogs.shift(lag_base)
        for window in [7, 14, 30, 60, 90]:
            min_p = max(1, window // 3)
            df[f'rev_rmean_{lag_base}_{window}'] = shifted_rev.rolling(window, min_periods=min_p).mean()
            if window in [30, 90]:   # std chỉ cần cửa sổ lớn
                df[f'rev_rstd_{lag_base}_{window}'] = shifted_rev.rolling(window, min_periods=min_p).std()
        df[f'cogs_rmean_{lag_base}_30'] = shifted_cogs.rolling(30, min_periods=5).mean()

    # ── Year-over-Year features (QUAN TRỌNG — version cũ thiếu) ─
    # YoY ratio encode momentum tăng trưởng tốt hơn raw lag rất nhiều
    for (l1, l2) in [(548, 912), (728, 1092), (912, 1456)]:
        r1 = df[f'rev_lag_{l1}']
        r2 = df[f'rev_lag_{l2}']
        safe_r2 = r2.replace(0, np.nan)
        df[f'rev_yoy_ratio_{l1}_{l2}']  = (r1 / safe_r2).clip(0, 5)
        df[f'rev_yoy_delta_{l1}_{l2}']  = (r1 - r2) / (r2.abs() + 1)

    # COGS/Revenue ratio (gross margin proxy từ lag)
    for lag in [548, 728]:
        r = df[f'rev_lag_{lag}']
        c = df[f'cogs_lag_{lag}']
        safe_r = r.replace(0, np.nan)
        df[f'gross_margin_lag_{lag}'] = ((r - c) / safe_r).clip(-1, 1)
        df[f'cogs_ratio_lag_{lag}']   = (c / safe_r).clip(0, 2)

    # ── Seasonality index (so sánh ngày này với trung bình năm trước) ─
    # rev_lag_548 / rev_rmean_548_365 ≈ "ngày này mạnh/yếu hơn average không"
    annual_mean_safe = df.get('rev_rmean_548_90', pd.Series(np.nan, index=df.index)).replace(0, np.nan)
    df['rev_seasonality_idx'] = (df['rev_lag_548'] / annual_mean_safe).clip(0, 5)

    return df


# ─────────────────────────────────────────────
#  GROUP 6 (NEW): ACTIVE PROMOTION FEATURES
# ─────────────────────────────────────────────

def add_promotion_features(df):
    """
    Tính trực tiếp từ promotions.csv — hoàn toàn test-safe
    vì promotions có start_date/end_date biết trước.
    """
    if not os.path.exists(PROMOTIONS_FILE):
        print("  [WARN] promotions.csv not found — skipping promotion features")
        return df

    promos = pd.read_csv(PROMOTIONS_FILE, parse_dates=['start_date', 'end_date'])
    dates  = df['Date'].values

    n       = len(dates)
    n_active_all        = np.zeros(n, dtype=int)
    n_active_pct        = np.zeros(n, dtype=int)
    n_active_fixed      = np.zeros(n, dtype=int)
    max_discount        = np.zeros(n, dtype=float)
    sum_discount        = np.zeros(n, dtype=float)
    has_stackable       = np.zeros(n, dtype=int)
    n_channels          = np.zeros(n, dtype=int)
    min_order_threshold = np.zeros(n, dtype=float)

    for _, row in promos.iterrows():
        s, e = row['start_date'], row['end_date']
        mask = (dates >= np.datetime64(s)) & (dates <= np.datetime64(e))
        idx  = np.where(mask)[0]
        if len(idx) == 0:
            continue

        n_active_all[idx]    += 1
        if row['promo_type'] == 'percentage':
            n_active_pct[idx]  += 1
            max_discount[idx]   = np.maximum(max_discount[idx], row['discount_value'])
        else:
            n_active_fixed[idx]+= 1

        sum_discount[idx]        += row['discount_value']
        has_stackable[idx]        = np.maximum(has_stackable[idx], row.get('stackable_flag', 0))
        if pd.notna(row.get('promo_channel')):
            n_channels[idx]      += 1
        if pd.notna(row.get('min_order_value')):
            min_order_threshold[idx] = np.maximum(min_order_threshold[idx],
                                                    row['min_order_value'])

    df['promo_n_active']          = n_active_all
    df['promo_n_percentage']      = n_active_pct
    df['promo_n_fixed']           = n_active_fixed
    df['promo_max_discount']      = max_discount
    df['promo_sum_discount']      = sum_discount
    df['promo_has_stackable']     = has_stackable
    df['promo_n_channels']        = n_channels
    df['promo_min_order_thresh']  = min_order_threshold
    df['promo_is_active']         = (n_active_all > 0).astype(int)

    # Promo intensity score (kết hợp số lượng + giá trị)
    df['promo_intensity'] = (
        df['promo_n_active'] * 0.4 +
        (df['promo_max_discount'] / 100.0).clip(0, 1) * 0.6
    )

    print(f"  Promotion features added — coverage: "
          f"{df['promo_is_active'].mean()*100:.1f}% of days have active promo")
    return df


# ─────────────────────────────────────────────
#  GROUP 7 (ENHANCED): AUXILIARY FEATURES
# ─────────────────────────────────────────────

def _yearly_monthly_map(target_df, aux_df, date_col, value_cols, agg_funcs, prefix,
                         lookback_years=3):
    """
    Tạo seasonal features theo (year, month) thay vì global average.

    Chiến lược:
    1. Tính average theo (year, month) → giữ được trend.
    2. Với mỗi ngày trong df, merge (year-1, month) → "cùng tháng năm ngoái".
    3. Tính thêm (year-2, month) rồi lấy YoY delta.

    Điều này khắc phục 2 vấn đề của version cũ:
    - Mất trend do average toàn bộ lịch sử.
    - Subtle leakage khi time-series cross-validation.
    """
    aux_df = aux_df.copy()
    aux_df['_year']  = aux_df[date_col].dt.year
    aux_df['_month'] = aux_df[date_col].dt.month

    agg_dict  = {c: f for c, f in zip(value_cols, agg_funcs)}
    rename_map= {c: f'{prefix}_{c}' for c in value_cols}

    ym_avg = (aux_df.groupby(['_year', '_month'])
                    .agg(agg_dict)
                    .reset_index()
                    .rename(columns={'_year':'_yr', '_month':'_mo', **rename_map}))

    feat_cols = [f'{prefix}_{c}' for c in value_cols]

    # Expand the aggregated data to cover all future years so that when shifting, 
    # we don't end up with NaNs for future test data (which would need auxiliary data from 1-2 years prior).
    # We forward-fill missing years for each month (i.e. using the most recent past year's data for that same month).
    min_yr = ym_avg['_yr'].min() if not ym_avg.empty else 2012
    max_yr = 2026 # cover safely up to future dates
    all_yrs = pd.DataFrame([(y, m) for y in range(int(min_yr), max_yr + 1) for m in range(1, 13)], columns=['_yr', '_mo'])
    
    ym_full = pd.merge(all_yrs, ym_avg, on=['_yr', '_mo'], how='left')
    ym_full = ym_full.sort_values(['_mo', '_yr'])
    ym_full[feat_cols] = ym_full.groupby('_mo')[feat_cols].ffill()

    # Merge năm ngoái (t-1) và năm kia (t-2) → safe cho test
    result = pd.DataFrame(index=target_df.index)

    for lag_yr in [1, 2]:
        suffix  = f'_lag{lag_yr}y'
        # Tạo lookup key: (year - lag_yr, month)
        lookup = ym_full.copy()
        lookup['_yr'] = lookup['_yr'] + lag_yr   # shift lên để match

        tmp = pd.merge(
            pd.DataFrame({'_yr': target_df['year'], '_mo': target_df['month']}),
            lookup.rename(columns={c: c + suffix for c in feat_cols}),
            on=['_yr', '_mo'], how='left'
        )
        for c in feat_cols:
            if c + suffix in tmp.columns:
                result[c + suffix] = tmp[c + suffix].values

    return result


def add_auxiliary_features(df):
    """
    Auxiliary features với chiến lược yearly-monthly thay vì global average.
    Thêm nhiều signal hơn từ mỗi bảng.
    """
    print("  Building auxiliary features (yearly-monthly strategy)...")

    # Helper để merge kết quả vào df
    def _merge(result_df):
        for col in result_df.columns:
            df[col] = result_df[col].values
        return df

    # ── 1. Orders ────────────────────────────────────────────
    if os.path.exists(ORDERS_FILE):
        orders = pd.read_csv(ORDERS_FILE,
                             usecols=['order_id', 'order_date',
                                      'order_status', 'payment_method', 'device_type'],
                             parse_dates=['order_date'])
        o = orders.copy()
        o['is_mobile']    = (o['device_type'] == 'mobile').astype(int)
        o['is_delivered'] = (o['order_status'] == 'delivered').astype(int)
        o_daily = o.groupby('order_date').agg(
            n_orders      =('order_id',    'count'),
            n_mobile      =('is_mobile',   'sum'),
            n_delivered   =('is_delivered','sum'),
        ).reset_index()
        o_daily['order_date'] = pd.to_datetime(o_daily['order_date'])

        res = _yearly_monthly_map(
            df, o_daily, 'order_date',
            ['n_orders', 'n_mobile', 'n_delivered'],
            ['mean', 'mean', 'mean'], 'orders'
        )
        df = _merge(res)

    # ── 2. Web Traffic ───────────────────────────────────────
    if os.path.exists(WEB_TRAFFIC_FILE):
        traffic = pd.read_csv(WEB_TRAFFIC_FILE, parse_dates=['date'])
        traffic['date'] = pd.to_datetime(traffic['date'])
        traffic['cvr_proxy'] = traffic['sessions'] / (traffic['unique_visitors'] + 1)

        res = _yearly_monthly_map(
            df, traffic, 'date',
            ['sessions', 'unique_visitors', 'page_views', 'bounce_rate',
             'avg_session_duration_sec', 'cvr_proxy'],
            ['mean'] * 6, 'web'
        )
        df = _merge(res)

    # ── 3. Returns ───────────────────────────────────────────
    if os.path.exists(RETURNS_FILE):
        returns = pd.read_csv(RETURNS_FILE, parse_dates=['return_date'])
        r_daily = returns.groupby('return_date').agg(
            n_returns    =('return_id',     'count'),
            total_refund =('refund_amount', 'sum'),
            avg_refund   =('refund_amount', 'mean'),
        ).reset_index()
        r_daily['return_date'] = pd.to_datetime(r_daily['return_date'])

        res = _yearly_monthly_map(
            df, r_daily, 'return_date',
            ['n_returns', 'total_refund', 'avg_refund'],
            ['mean', 'mean', 'mean'], 'returns'
        )
        df = _merge(res)

    # ── 4. Inventory ─────────────────────────────────────────
    if os.path.exists(INVENTORY_FILE):
        inv = pd.read_csv(INVENTORY_FILE, parse_dates=['snapshot_date'],
                          usecols=['snapshot_date', 'sell_through_rate',
                                   'stockout_days', 'fill_rate',
                                   'stockout_flag', 'overstock_flag'])
        inv_agg = inv.groupby('snapshot_date').agg(
            avg_sell_through =('sell_through_rate', 'mean'),
            avg_stockout_days=('stockout_days',      'mean'),
            avg_fill_rate    =('fill_rate',          'mean'),
            pct_stockout     =('stockout_flag',      'mean'),
            pct_overstock    =('overstock_flag',     'mean'),
        ).reset_index()

        res = _yearly_monthly_map(
            df, inv_agg, 'snapshot_date',
            ['avg_sell_through', 'avg_stockout_days', 'avg_fill_rate',
             'pct_stockout', 'pct_overstock'],
            ['mean'] * 5, 'inv'
        )
        df = _merge(res)

    # ── 5. Reviews ───────────────────────────────────────────
    if os.path.exists(REVIEWS_FILE):
        reviews = pd.read_csv(REVIEWS_FILE,
                              usecols=['review_date', 'rating'],
                              parse_dates=['review_date'])
        r_daily = reviews.groupby('review_date').agg(
            avg_rating =('rating', 'mean'),
            n_reviews  =('rating', 'count'),
            pct_5star  =('rating', lambda x: (x == 5).mean()),
            pct_1star  =('rating', lambda x: (x == 1).mean()),
        ).reset_index()

        res = _yearly_monthly_map(
            df, r_daily, 'review_date',
            ['avg_rating', 'n_reviews', 'pct_5star', 'pct_1star'],
            ['mean', 'mean', 'mean', 'mean'], 'reviews'
        )
        df = _merge(res)

    # ── 6. Shipments ─────────────────────────────────────────
    if os.path.exists(SHIPMENTS_FILE):
        ships = pd.read_csv(SHIPMENTS_FILE,
                            usecols=['ship_date', 'delivery_date', 'shipping_fee'],
                            parse_dates=['ship_date', 'delivery_date'])
        ships['delivery_days'] = (ships['delivery_date'] - ships['ship_date']).dt.days.clip(0, 30)
        s_daily = ships.groupby('ship_date').agg(
            avg_ship_fee    =('shipping_fee',  'mean'),
            n_shipments     =('shipping_fee',  'count'),
            avg_delivery_days=('delivery_days','mean'),
            free_ship_rate  =('shipping_fee',  lambda x: (x == 0).mean()),
        ).reset_index()

        res = _yearly_monthly_map(
            df, s_daily, 'ship_date',
            ['avg_ship_fee', 'n_shipments', 'avg_delivery_days', 'free_ship_rate'],
            ['mean', 'mean', 'mean', 'mean'], 'ship'
        )
        df = _merge(res)

    # ── 7. Payments ──────────────────────────────────────────
    if os.path.exists(PAYMENTS_FILE) and os.path.exists(ORDERS_FILE):
        pay = pd.read_csv(PAYMENTS_FILE,
                          usecols=['order_id', 'payment_value',
                                   'installments', 'payment_method'])
        ord_ = pd.read_csv(ORDERS_FILE,
                           usecols=['order_id', 'order_date'],
                           parse_dates=['order_date'])
        pay = pay.merge(ord_, on='order_id', how='left').dropna(subset=['order_date'])
        pay['is_installment'] = (pay['installments'] > 1).astype(int)
        p_daily = pay.groupby('order_date').agg(
            avg_payment      =('payment_value',  'mean'),
            avg_installments =('installments',   'mean'),
            installment_rate =('is_installment', 'mean'),
            total_gmv        =('payment_value',  'sum'),
        ).reset_index()

        res = _yearly_monthly_map(
            df, p_daily, 'order_date',
            ['avg_payment', 'avg_installments', 'installment_rate', 'total_gmv'],
            ['mean', 'mean', 'mean', 'mean'], 'pay'
        )
        df = _merge(res)

    # ── 8. YoY delta cho auxiliary (nếu có lag1y và lag2y) ───
    # Tính tỉ lệ tăng trưởng năm ngoái so năm kia cho từng metric
    aux_prefixes = ['orders_n_orders', 'web_sessions', 'web_unique_visitors',
                    'returns_n_returns', 'pay_total_gmv']
    for base in aux_prefixes:
        c1 = f'{base}_lag1y'
        c2 = f'{base}_lag2y'
        if c1 in df.columns and c2 in df.columns:
            safe_c2 = df[c2].replace(0, np.nan)
            df[f'{base}_yoy_ratio'] = (df[c1] / safe_c2).clip(0, 5)

    print(f"  Auxiliary features added: {len([c for c in df.columns if any(p in c for p in ['orders_', 'web_', 'returns_', 'inv_', 'reviews_', 'ship_', 'pay_'])])} cols")
    return df


# ─────────────────────────────────────────────
#  GROUP 8 (NEW): INTERACTION FEATURES
# ─────────────────────────────────────────────

def add_interaction_features(df):
    """
    Cross features giữa các nhóm — quan trọng cho tree-based models
    vì chúng không tự học interaction bậc cao tốt bằng linear models.
    """
    # Trend × Seasonality (model khác nhau theo giai đoạn + mùa)
    if 'trend_linear' in df.columns:
        df['trend_x_month_sin'] = df['trend_linear'] * df['month_sin']
        df['trend_x_month_cos'] = df['trend_linear'] * df['month_cos']
        df['trend_x_quarter']   = df['trend_linear'] * df['quarter']
        df['trend_x_season']    = df['trend_linear'] * df['season']

    # Tet × Promo (Tết + khuyến mãi = revenue spike cực lớn)
    if 'promo_is_active' in df.columns and 'is_tet_month' in df.columns:
        df['tet_x_promo']        = df['is_tet_month'] * df['promo_is_active']
        df['tet_x_promo_intense']= df['tet_proximity'] * df['promo_intensity']

    # Weekend × Promo
    if 'promo_is_active' in df.columns:
        df['weekend_x_promo']    = df['is_weekend'] * df['promo_is_active']
        df['payday_x_promo']     = df['is_payday_week'] * df['promo_is_active']

    # Double day × Promo intensity
    if 'is_double_day' in df.columns and 'promo_intensity' in df.columns:
        df['double_day_x_promo'] = df['is_double_day'] * df['promo_intensity']

    # YoY ratio × Trend (phân biệt tăng trưởng có bền vững không)
    if 'rev_yoy_ratio_548_912' in df.columns and 'trend_linear' in df.columns:
        df['yoy_x_trend'] = df['rev_yoy_ratio_548_912'] * df['trend_linear']

    return df


# ─────────────────────────────────────────────
#  MAIN ENTRY POINT
# ─────────────────────────────────────────────

def build_features(sales_df=None):
    """
    Pipeline đầy đủ — thứ tự quan trọng (lag cần có year/month trước).
    """
    if sales_df is None:
        df = _load_sales()
    else:
        df = sales_df.copy().sort_values('Date').reset_index(drop=True)

    print("=" * 60)
    print("FEATURE ENGINEERING — Enhanced Pipeline v2")
    print("=" * 60)

    print("\n[1/8] Time encodings + cyclical...")
    df = add_time_features(df)

    print("[2/8] Fourier seasonality (fixed basis)...")
    df = add_fourier_features(df)

    print("[3/8] Event features (Tết + holidays VN + sale days)...")
    df = add_event_features(df)

    print("[4/8] Trend features (linear / log / piecewise COVID)...")
    df = add_trend_features(df)

    print("[5/8] Revenue + COGS lags + YoY ratios...")
    df = add_lag_features(df)

    print("[6/8] Active promotion features...")
    df = add_promotion_features(df)

    print("[7/8] Auxiliary features (yearly-monthly strategy)...")
    df = add_auxiliary_features(df)

    print("[8/8] Interaction features...")
    df = add_interaction_features(df)

    # ── Impute NaN ───────────────────────────────────────────
    # Chỉ impute float cols không phải target
    exclude = {'Date', 'Revenue', 'COGS'}
    float_cols = [c for c in df.columns
                  if c not in exclude and df[c].dtype in (np.float64, np.float32)]
    # Lag features: fill 0 (absence of signal)
    # Ratio features: fill 1 (neutral ratio)
    for col in float_cols:
        if 'ratio' in col or 'yoy' in col or 'idx' in col or 'margin' in col:
            df[col] = df[col].fillna(1.0)
        else:
            df[col] = df[col].fillna(0.0)

    n_features = len([c for c in df.columns if c not in exclude])
    print(f"\n✅ Total features: {n_features}")

    # Feature group summary
    groups = {
        'time':         [c for c in df.columns if any(x in c for x in ['month', 'dow', 'day', 'week', 'quarter', 'season', 'year', 'sin', 'cos']) and 'fourier' not in c and 'f_' not in c],
        'fourier':      [c for c in df.columns if c.startswith('f_')],
        'events':       [c for c in df.columns if any(x in c for x in ['tet', 'prox', 'holiday', 'hung', 'liberation', 'labor', 'national', 'new_year', 'xmas', 'double', 'payday'])],
        'trend':        [c for c in df.columns if 'trend' in c or 'covid' in c or 'days_from' in c],
        'lags':         [c for c in df.columns if 'lag' in c or 'rmean' in c or 'rstd' in c or 'yoy' in c or 'margin' in c or 'cogs_ratio' in c or 'seasonality' in c],
        'promo':        [c for c in df.columns if c.startswith('promo_')],
        'auxiliary':    [c for c in df.columns if any(c.startswith(p) for p in ['orders_', 'web_', 'returns_', 'inv_', 'reviews_', 'ship_', 'pay_'])],
        'interaction':  [c for c in df.columns if '_x_' in c],
    }
    print("\nFeature breakdown:")
    for grp, cols in groups.items():
        print(f"  {grp:<14}: {len(cols):>3} features")

    return df