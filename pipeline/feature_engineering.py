"""
Feature Engineering for Revenue Forecasting.
Loads sales.csv + auxiliary tables, creates all features.
"""
import pandas as pd
import numpy as np
from pipeline.config import *


def load_sales():
    """Load and sort sales data."""
    df = pd.read_csv(SALES_FILE, parse_dates=['Date'])
    return df.sort_values('Date').reset_index(drop=True)


def add_time_features(df):
    """Calendar and cyclical time features from Date."""
    d = df['Date']
    df['year'] = d.dt.year
    df['month'] = d.dt.month
    df['day'] = d.dt.day
    df['day_of_week'] = d.dt.dayofweek
    df['day_of_year'] = d.dt.dayofyear
    df['week_of_year'] = d.dt.isocalendar().week.astype(int)
    df['quarter'] = d.dt.quarter
    
    # Season (1=Spring, 2=Summer, 3=Autumn, 4=Winter) based on Vietnam typical seasons
    # Using simple month mapping
    season_map = {1: 4, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 3, 9: 3, 10: 3, 11: 4, 12: 4}
    df['season'] = df['month'].map(season_map)

    # Boolean flags
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_month_start'] = d.dt.is_month_start.astype(int)
    df['is_month_end'] = d.dt.is_month_end.astype(int)
    df['is_quarter_start'] = d.dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = d.dt.is_quarter_end.astype(int)

    # Cyclical encoding (captures circular nature of calendar)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)

    # Day position within month (0→1)
    df['day_of_month_ratio'] = df['day'] / d.dt.days_in_month

    return df


def add_holiday_features(df):
    """Vietnamese holidays and e-commerce events."""
    d = df['Date']

    # -- Tet Nguyen Dan (approximate: 7-day period) --
    tet_starts = {
        2013: '2013-02-09', 2014: '2014-01-30', 2015: '2015-02-17',
        2016: '2016-02-07', 2017: '2017-01-27', 2018: '2018-02-14',
        2019: '2019-02-02', 2020: '2020-01-23', 2021: '2021-02-10',
        2022: '2022-01-31', 2023: '2023-01-20', 2024: '2024-02-08',
    }
    df['is_tet'] = 0
    for year, start in tet_starts.items():
        start_dt = pd.Timestamp(start)
        mask = (d >= start_dt - pd.Timedelta(days=3)) & (d <= start_dt + pd.Timedelta(days=7))
        df.loc[mask, 'is_tet'] = 1

    # -- Fixed national holidays --
    df['is_national_holiday'] = 0
    for md in ['01-01', '04-30', '05-01', '09-02']:
        df.loc[d.dt.strftime('%m-%d') == md, 'is_national_holiday'] = 1

    # -- E-commerce mega-sale events --
    df['is_ecommerce_event'] = 0
    for md in ['11-11', '12-12']:
        mask = d.dt.strftime('%m-%d') == md
        df.loc[mask, 'is_ecommerce_event'] = 1
    # Black Friday: 4th Friday of November
    nov_mask = d.dt.month == 11
    fri_mask = d.dt.dayofweek == 4
    day_mask = d.dt.day.between(22, 28)
    df.loc[nov_mask & fri_mask & day_mask, 'is_ecommerce_event'] = 1

    # -- Payday effect (1st, 15th of month) --
    df['is_payday'] = ((d.dt.day == 1) | (d.dt.day == 15)).astype(int)

    # Combined holiday flag
    df['is_any_holiday'] = ((df['is_tet'] == 1) |
                            (df['is_national_holiday'] == 1) |
                            (df['is_ecommerce_event'] == 1)).astype(int)

    # -- Distance to holidays --
    # Default to a large number if no holiday found
    holiday_dates = df.loc[df['is_any_holiday'] == 1, 'Date'].values
    if len(holiday_dates) > 0:
        # Avoid apply for speed, use merge_asof or numpy searchsorted
        dates = df['Date'].values
        
        # Days since last holiday
        idx = np.searchsorted(holiday_dates, dates, side='right') - 1
        idx_safe = np.clip(idx, 0, len(holiday_dates) - 1)
        last_holidays = np.where(idx >= 0, holiday_dates[idx_safe], pd.NaT)
        df['days_since_last_holiday'] = (d - pd.to_datetime(last_holidays)).dt.days
        
        # Days to next holiday
        idx_next = np.searchsorted(holiday_dates, dates, side='left')
        idx_next_safe = np.clip(idx_next, 0, len(holiday_dates) - 1)
        next_holidays = np.where(idx_next < len(holiday_dates), holiday_dates[idx_next_safe], pd.NaT)
        df['days_to_next_holiday'] = (pd.to_datetime(next_holidays) - d).dt.days

        df['days_since_last_holiday'] = df['days_since_last_holiday'].fillna(365).clip(0, 365)
        df['days_to_next_holiday'] = df['days_to_next_holiday'].fillna(365).clip(0, 365)
    else:
        df['days_since_last_holiday'] = 365
        df['days_to_next_holiday'] = 365

    return df


def add_revenue_lag_features(df):
    """Lag, rolling, EWM, diff features from Revenue."""
    rev = df['Revenue']

    # Lag features (shift ensures no leakage of current day)
    for lag in [1, 2, 3, 7, 14, 28, 30, 90, 180, 365, 730]:
        df[f'rev_lag_{lag}'] = rev.shift(lag)

    # Rolling statistics (shift(1) to avoid current day leakage)
    shifted = rev.shift(1)
    for w in [7, 14, 30, 60, 90, 180, 365]:
        df[f'rev_rmean_{w}'] = shifted.rolling(w, min_periods=1).mean()
    for w in [7, 30, 90]:
        df[f'rev_rstd_{w}'] = shifted.rolling(w, min_periods=1).std()
    for w in [7, 30]:
        df[f'rev_rmin_{w}'] = shifted.rolling(w, min_periods=1).min()
        df[f'rev_rmax_{w}'] = shifted.rolling(w, min_periods=1).max()

    # Exponential weighted mean
    for span in [7, 30, 90]:
        df[f'rev_ewm_{span}'] = shifted.ewm(span=span, min_periods=1).mean()

    # Differences
    df['rev_diff_1'] = rev.diff(1)
    df['rev_diff_7'] = rev.diff(7)

    # Momentum (ratios) - avoid division by zero by adding 1
    df['rev_momentum_1_7'] = rev.shift(1) / (rev.shift(7) + 1)
    df['rev_momentum_1_30'] = rev.shift(1) / (rev.shift(30) + 1)

    # Year-over-year ratio
    df['rev_yoy_ratio'] = rev / (rev.shift(365).replace(0, np.nan))

    return df


def add_cogs_features(df):
    """Lag & rolling features from COGS (correlation 0.976 with Revenue)."""
    return df


def add_auxiliary_features(df):
    """Aggregate ALL auxiliary tables to daily level and merge."""
    import os
    print("  Loading auxiliary tables...")

    # ---- 1. Orders (daily count, cancel rate, device/source mix) ----
    orders = None
    if os.path.exists(ORDERS_FILE):
        print("    orders.csv...")
        orders = pd.read_csv(
            ORDERS_FILE,
            usecols=['order_id', 'order_date', 'order_status',
                     'payment_method', 'device_type', 'order_source', 'zip'],
            parse_dates=['order_date']
        )
        orders_daily = orders.groupby('order_date').agg(
            n_orders=('order_id', 'count'),
            n_cancelled=('order_status', lambda x: (x == 'cancelled').sum()),
            n_delivered=('order_status', lambda x: (x == 'delivered').sum()),
            n_device_mobile=('device_type', lambda x: (x == 'mobile').sum()),
            n_unique_sources=('order_source', 'nunique'),
        ).reset_index()
        orders_daily['cancel_rate'] = orders_daily['n_cancelled'] / (orders_daily['n_orders'] + 1)
        orders_daily['delivery_rate'] = orders_daily['n_delivered'] / (orders_daily['n_orders'] + 1)
        orders_daily['mobile_rate'] = orders_daily['n_device_mobile'] / (orders_daily['n_orders'] + 1)
        df = df.merge(orders_daily, left_on='Date', right_on='order_date', how='left')
        df.drop('order_date', axis=1, inplace=True, errors='ignore')

        # Geography diversity per day (via orders.zip → geography.region)
        GEOGRAPHY_FILE = os.path.join(ROOT_DIR, 'geography.csv')
        if os.path.exists(GEOGRAPHY_FILE):
            print("    geography.csv (region diversity)...")
            geo = pd.read_csv(GEOGRAPHY_FILE, usecols=['zip', 'region', 'city'])
            orders_geo = orders.merge(geo, on='zip', how='left')
            geo_daily = orders_geo.groupby('order_date').agg(
                n_unique_regions=('region', 'nunique'),
                n_unique_cities=('city', 'nunique'),
            ).reset_index()
            df = df.merge(geo_daily, left_on='Date', right_on='order_date', how='left')
            df.drop('order_date', axis=1, inplace=True, errors='ignore')

    # ---- 2. Order Items (AOV, qty, discounts, promo usage) ----
    ORDER_ITEMS_FILE = os.path.join(ROOT_DIR, 'order_items.csv')
    if os.path.exists(ORDER_ITEMS_FILE) and orders is not None:
        print("    order_items.csv...")
        items = pd.read_csv(
            ORDER_ITEMS_FILE,
            usecols=['order_id', 'product_id', 'quantity', 'unit_price',
                     'discount_amount', 'promo_id']
        )
        # Join with orders to get order_date
        items = items.merge(orders[['order_id', 'order_date']], on='order_id', how='left')
        items['line_total'] = items['quantity'] * items['unit_price']
        items['has_promo'] = items['promo_id'].notna().astype(int)

        items_daily = items.groupby('order_date').agg(
            total_qty=('quantity', 'sum'),
            total_line_value=('line_total', 'sum'),
            total_discount=('discount_amount', 'sum'),
            avg_unit_price=('unit_price', 'mean'),
            n_unique_products=('product_id', 'nunique'),
            promo_usage_rate=('has_promo', 'mean'),
            n_line_items=('order_id', 'count'),
        ).reset_index()
        items_daily['avg_order_value'] = items_daily['total_line_value'] / (items_daily['n_line_items'] + 1)
        items_daily['discount_ratio'] = items_daily['total_discount'] / (items_daily['total_line_value'] + 1)

        df = df.merge(items_daily, left_on='Date', right_on='order_date', how='left')
        df.drop('order_date', axis=1, inplace=True, errors='ignore')

        # Join with products for category/segment mix
        PRODUCTS_FILE = os.path.join(ROOT_DIR, 'products.csv')
        if os.path.exists(PRODUCTS_FILE):
            print("    products.csv (category/segment mix)...")
            products = pd.read_csv(PRODUCTS_FILE, usecols=['product_id', 'category', 'segment', 'price', 'cogs'])
            items_prod = items.merge(products, on='product_id', how='left')

            # Category diversity per day
            cat_daily = items_prod.groupby('order_date').agg(
                n_categories=('category', 'nunique'),
                n_segments=('segment', 'nunique'),
                avg_product_price=('price', 'mean'),
                avg_product_margin=('price', lambda x: ((x - items_prod.loc[x.index, 'cogs']) / x).mean()),
            ).reset_index()
            df = df.merge(cat_daily, left_on='Date', right_on='order_date', how='left')
            df.drop('order_date', axis=1, inplace=True, errors='ignore')

    # ---- 3. Payments (daily payment stats) ----
    PAYMENTS_FILE = os.path.join(ROOT_DIR, 'payments.csv')
    if os.path.exists(PAYMENTS_FILE) and orders is not None:
        print("    payments.csv...")
        payments = pd.read_csv(
            PAYMENTS_FILE,
            usecols=['order_id', 'payment_value', 'installments']
        )
        payments = payments.merge(orders[['order_id', 'order_date']], on='order_id', how='left')
        pay_daily = payments.groupby('order_date').agg(
            total_payment_value=('payment_value', 'sum'),
            avg_payment_value=('payment_value', 'mean'),
            avg_installments=('installments', 'mean'),
            pct_installment=('installments', lambda x: (x > 1).mean()),
        ).reset_index()
        df = df.merge(pay_daily, left_on='Date', right_on='order_date', how='left')
        df.drop('order_date', axis=1, inplace=True, errors='ignore')

    # ---- 4. Shipments (delivery time, shipping fees) ----
    SHIPMENTS_FILE = os.path.join(ROOT_DIR, 'shipments.csv')
    if os.path.exists(SHIPMENTS_FILE) and orders is not None:
        print("    shipments.csv...")
        ships = pd.read_csv(
            SHIPMENTS_FILE,
            usecols=['order_id', 'ship_date', 'delivery_date', 'shipping_fee'],
            parse_dates=['ship_date', 'delivery_date']
        )
        ships = ships.merge(orders[['order_id', 'order_date']], on='order_id', how='left')
        ships['delivery_days'] = (ships['delivery_date'] - ships['ship_date']).dt.days
        ships['ship_lag'] = (ships['ship_date'] - ships['order_date']).dt.days

        ship_daily = ships.groupby('order_date').agg(
            avg_delivery_days=('delivery_days', 'mean'),
            avg_ship_lag=('ship_lag', 'mean'),
            total_shipping_fee=('shipping_fee', 'sum'),
            avg_shipping_fee=('shipping_fee', 'mean'),
            pct_free_shipping=('shipping_fee', lambda x: (x == 0).mean()),
        ).reset_index()
        df = df.merge(ship_daily, left_on='Date', right_on='order_date', how='left')
        df.drop('order_date', axis=1, inplace=True, errors='ignore')

    # ---- 5. Returns (daily refunds, return rate) ----
    if os.path.exists(RETURNS_FILE):
        print("    returns.csv...")
        returns = pd.read_csv(
            RETURNS_FILE,
            usecols=['return_id', 'return_date', 'refund_amount', 'return_quantity'],
            parse_dates=['return_date']
        )
        returns_daily = returns.groupby('return_date').agg(
            n_returns=('return_id', 'count'),
            total_refund=('refund_amount', 'sum'),
            avg_refund=('refund_amount', 'mean'),
            total_return_qty=('return_quantity', 'sum'),
        ).reset_index()
        df = df.merge(returns_daily, left_on='Date', right_on='return_date', how='left')
        df.drop('return_date', axis=1, inplace=True, errors='ignore')

    # ---- 6. Reviews (daily ratings) ----
    REVIEWS_FILE = os.path.join(ROOT_DIR, 'reviews.csv')
    if os.path.exists(REVIEWS_FILE):
        print("    reviews.csv...")
        reviews = pd.read_csv(
            REVIEWS_FILE,
            usecols=['review_id', 'review_date', 'rating'],
            parse_dates=['review_date']
        )
        rev_daily = reviews.groupby('review_date').agg(
            n_reviews=('review_id', 'count'),
            avg_rating=('rating', 'mean'),
            pct_5star=('rating', lambda x: (x == 5).mean()),
            pct_low_rating=('rating', lambda x: (x <= 2).mean()),
        ).reset_index()
        df = df.merge(rev_daily, left_on='Date', right_on='review_date', how='left')
        df.drop('review_date', axis=1, inplace=True, errors='ignore')

    # ---- 7. Web traffic ----
    if os.path.exists(WEB_TRAFFIC_FILE):
        print("    web_traffic.csv...")
        traffic = pd.read_csv(WEB_TRAFFIC_FILE, parse_dates=['date'])
        traffic_daily = traffic.groupby('date').agg(
            total_sessions=('sessions', 'sum'),
            total_visitors=('unique_visitors', 'sum'),
            total_page_views=('page_views', 'sum'),
            avg_bounce_rate=('bounce_rate', 'mean'),
            avg_session_duration=('avg_session_duration_sec', 'mean'),
        ).reset_index()
        # Conversion proxy: pages per visitor
        traffic_daily['pages_per_visitor'] = traffic_daily['total_page_views'] / (traffic_daily['total_visitors'] + 1)
        df = df.merge(traffic_daily, left_on='Date', right_on='date', how='left')
        df.drop('date', axis=1, inplace=True, errors='ignore')

    # ---- 8. Promotions (active promo count + discount stats) ----
    if os.path.exists(PROMOTIONS_FILE):
        print("    promotions.csv...")
        promos = pd.read_csv(PROMOTIONS_FILE, parse_dates=['start_date', 'end_date'])
        def promo_stats(date):
            active = promos[(promos['start_date'] <= date) & (promos['end_date'] >= date)]
            n = len(active)
            avg_disc = active['discount_value'].mean() if n > 0 else 0
            has_pct = (active['promo_type'] == 'percentage').sum() if n > 0 else 0
            return pd.Series({'n_active_promos': n, 'avg_promo_discount': avg_disc,
                              'n_pct_promos': has_pct})
        promo_daily = df[['Date']].drop_duplicates().copy()
        promo_daily = promo_daily.join(promo_daily['Date'].apply(promo_stats))
        df = df.merge(promo_daily, on='Date', how='left')

    # ---- 9. Inventory (monthly → forward fill, enriched) ----
    if os.path.exists(INVENTORY_FILE):
        print("    inventory.csv...")
        inv = pd.read_csv(
            INVENTORY_FILE,
            usecols=['snapshot_date', 'stock_on_hand', 'stockout_days',
                     'fill_rate', 'units_sold', 'days_of_supply',
                     'stockout_flag', 'sell_through_rate'],
            parse_dates=['snapshot_date']
        )
        inv_monthly = inv.groupby('snapshot_date').agg(
            total_stock=('stock_on_hand', 'sum'),
            total_stockout_days=('stockout_days', 'sum'),
            avg_fill_rate=('fill_rate', 'mean'),
            total_units_sold=('units_sold', 'sum'),
            avg_days_of_supply=('days_of_supply', 'mean'),
            pct_stockout=('stockout_flag', 'mean'),
            avg_sell_through=('sell_through_rate', 'mean'),
        ).reset_index()
        df = df.merge(inv_monthly, left_on='Date', right_on='snapshot_date', how='left')
        df.drop('snapshot_date', axis=1, inplace=True, errors='ignore')
        inv_cols = ['total_stock', 'total_stockout_days', 'avg_fill_rate',
                    'total_units_sold', 'avg_days_of_supply', 'pct_stockout',
                    'avg_sell_through']
        df[inv_cols] = df[inv_cols].ffill()

    # ---- 10. Customers (new signups per day) ----
    if os.path.exists(CUSTOMERS_FILE):
        print("    customers.csv...")
        customers = pd.read_csv(
            CUSTOMERS_FILE,
            usecols=['customer_id', 'signup_date'],
            parse_dates=['signup_date']
        )
        signups = customers.groupby('signup_date').agg(
            n_new_signups=('customer_id', 'count'),
        ).reset_index()
        df = df.merge(signups, left_on='Date', right_on='signup_date', how='left')
        df.drop('signup_date', axis=1, inplace=True, errors='ignore')

    return df


def add_seasonal_features(df):
    """Historical seasonal averages (computed only from known Revenue)."""
    # These features capture "what's typical for this time of year"
    known = df[df['Revenue'].notna()].copy()

    # Average Revenue by month (across all years)
    month_avg = known.groupby('month')['Revenue'].mean().to_dict()
    df['rev_month_avg'] = df['month'].map(month_avg)

    # Average Revenue by day_of_week
    dow_avg = known.groupby('day_of_week')['Revenue'].mean().to_dict()
    df['rev_dow_avg'] = df['day_of_week'].map(dow_avg)

    # Average Revenue by week_of_year
    woy_avg = known.groupby('week_of_year')['Revenue'].mean().to_dict()
    df['rev_woy_avg'] = df['week_of_year'].map(woy_avg)

    # Average Revenue by quarter
    q_avg = known.groupby('quarter')['Revenue'].mean().to_dict()
    df['rev_quarter_avg'] = df['quarter'].map(q_avg)

    # Average Revenue by season
    s_avg = known.groupby('season')['Revenue'].mean().to_dict()
    df['rev_season_avg'] = df['season'].map(s_avg)

    return df


def build_features(sales_df=None):
    """
    Main entry point: build all features.
    
    If sales_df is None, loads from sales.csv.
    Can also accept a DataFrame with submission dates appended
    (Revenue=NaN for those dates).
    """
    if sales_df is None:
        df = load_sales()
    else:
        df = sales_df.copy().sort_values('Date').reset_index(drop=True)

    print("=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)

    print("  Adding time features...")
    df = add_time_features(df)

    print("  Adding holiday features...")
    df = add_holiday_features(df)

    print("  Adding Revenue lag/rolling features...")
    df = add_revenue_lag_features(df)

    print("  Adding COGS features...")
    df = add_cogs_features(df)

    print("  Adding seasonal averages...")
    df = add_seasonal_features(df)

    print("  Adding auxiliary features...")
    df = add_auxiliary_features(df)

    # Fill remaining NaN in auxiliary columns with 0
    exclude = ['Date', 'Revenue', 'COGS']
    for col in df.columns:
        if col not in exclude and df[col].dtype in [np.float64, np.float32]:
            df[col] = df[col].fillna(0)

    n_features = len([c for c in df.columns if c not in exclude])
    print(f"\n  ✅ Total features created: {n_features}")
    return df
