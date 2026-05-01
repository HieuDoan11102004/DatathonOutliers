from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "order_count",
    "unique_customer_count",
    "delivered_order_count",
    "returned_order_count",
    "cancelled_order_count",
    "mobile_order_share",
    "paid_search_order_share",
    "quantity_sum",
    "gross_item_value",
    "discount_sum",
    "avg_discount_per_order",
    "sessions",
    "unique_visitors",
    "page_views",
    "bounce_rate",
    "avg_session_duration_sec",
    "active_promo_count",
    "active_percentage_promo_count",
    "active_fixed_amount_promo_count",
    "avg_active_discount_value",
    "max_active_discount_value",
    
    # === NEW FEATURES ===
    # "promo_avg_duration_days",
    # "has_stackable_promo",
    # "credit_card_share",
    # "cod_share",
    # "avg_delivery_delay_days",
    # "free_shipping_share",
    # "defective_return_count",
    # "wrong_size_count",
    # "new_customer_count",
    # "returning_customer_count",
    # "reorder_flag_pct",
    # "stockout_flag_pct",
]

PROMOTION_COLUMNS = [
    "active_promo_count",
    "active_percentage_promo_count",
    "active_fixed_amount_promo_count",
    "avg_active_discount_value",
    "max_active_discount_value",
    # "promo_avg_duration_days",
    # "has_stackable_promo",
]

NON_PROMOTION_COLUMNS = [col for col in FEATURE_COLUMNS if col not in PROMOTION_COLUMNS]


def _orders_daily(path: str | Path = "orders.csv") -> pd.DataFrame:
    try:
        orders = pd.read_csv(path, parse_dates=["order_date"])
    except FileNotFoundError:
        return pd.DataFrame(columns=["Date", "order_count", "unique_customer_count", "delivered_order_count", "returned_order_count", "cancelled_order_count", "mobile_order_share", "paid_search_order_share"])
    orders["Date"] = orders["order_date"].dt.normalize()

    grouped = orders.groupby("Date")
    daily = grouped.agg(
        order_count=("order_id", "nunique"),
        unique_customer_count=("customer_id", "nunique"),
    ).reset_index()

    status = orders["order_status"].astype(str)
    device = orders["device_type"].astype(str)
    source = orders["order_source"].astype(str)

    daily["delivered_order_count"] = status.eq("delivered").groupby(orders["Date"]).sum().reindex(daily["Date"]).to_numpy()
    daily["returned_order_count"] = status.eq("returned").groupby(orders["Date"]).sum().reindex(daily["Date"]).to_numpy()
    daily["cancelled_order_count"] = status.eq("cancelled").groupby(orders["Date"]).sum().reindex(daily["Date"]).to_numpy()
    daily["mobile_order_count"] = device.eq("mobile").groupby(orders["Date"]).sum().reindex(daily["Date"]).to_numpy()
    daily["paid_search_order_count"] = source.eq("paid_search").groupby(orders["Date"]).sum().reindex(daily["Date"]).to_numpy()

    denom = daily["order_count"].replace(0, np.nan)
    daily["mobile_order_share"] = daily["mobile_order_count"] / denom
    daily["paid_search_order_share"] = daily["paid_search_order_count"] / denom

    return daily.drop(columns=["mobile_order_count", "paid_search_order_count"]).fillna(0.0)


def _order_items_daily(
    orders_path: str | Path = "orders.csv",
    items_path: str | Path = "order_items.csv",
) -> pd.DataFrame:
    try:
        orders = pd.read_csv(orders_path, usecols=["order_id", "order_date"], parse_dates=["order_date"])
        items = pd.read_csv(
            items_path,
            usecols=["order_id", "quantity", "unit_price", "discount_amount"],
        )
    except FileNotFoundError:
        return pd.DataFrame(columns=["Date", "quantity_sum", "gross_item_value", "discount_sum", "avg_discount_per_order"])
    items["gross_item_value"] = items["quantity"] * items["unit_price"]

    merged = items.merge(orders, on="order_id", how="left")
    merged["Date"] = merged["order_date"].dt.normalize()
    merged = merged.dropna(subset=["Date"])

    daily = merged.groupby("Date").agg(
        quantity_sum=("quantity", "sum"),
        gross_item_value=("gross_item_value", "sum"),
        discount_sum=("discount_amount", "sum"),
        item_order_count=("order_id", "nunique"),
    ).reset_index()

    daily["avg_discount_per_order"] = daily["discount_sum"] / daily["item_order_count"].replace(0, np.nan)
    return daily.drop(columns=["item_order_count"]).fillna(0.0)


def _web_daily(path: str | Path = "web_traffic.csv") -> pd.DataFrame:
    try:
        web = pd.read_csv(path, parse_dates=["date"])
    except FileNotFoundError:
        return pd.DataFrame(columns=["Date", "sessions", "unique_visitors", "page_views", "bounce_rate", "avg_session_duration_sec"])
    web["Date"] = web["date"].dt.normalize()
    return web.groupby("Date").agg(
        sessions=("sessions", "sum"),
        unique_visitors=("unique_visitors", "sum"),
        page_views=("page_views", "sum"),
        bounce_rate=("bounce_rate", "mean"),
        avg_session_duration_sec=("avg_session_duration_sec", "mean"),
    ).reset_index()


def _promotions_daily(dates: pd.Series, path: str | Path = "promotions.csv") -> pd.DataFrame:
    normalized_dates = pd.to_datetime(dates, errors="coerce").dt.normalize()
    try:
        promos = pd.read_csv(path, parse_dates=["start_date", "end_date"])
    except FileNotFoundError:
        rows = [{"Date": d, **{col: 0.0 for col in PROMOTION_COLUMNS}} for d in normalized_dates if not pd.isna(d)]
        return pd.DataFrame(rows)
    promos["start_date"] = pd.to_datetime(promos["start_date"], errors="coerce").dt.normalize()
    promos["end_date"] = pd.to_datetime(promos["end_date"], errors="coerce").dt.normalize()
    promos = promos.dropna(subset=["start_date", "end_date"])

    stats_by_date: dict[pd.Timestamp, dict[str, float]] = {}
    for date in pd.Index(normalized_dates.dropna().unique()):
        active = promos[(promos["start_date"] <= date) & (promos["end_date"] >= date)]
        stats_by_date[date] = {
            "active_promo_count": float(len(active)),
            "active_percentage_promo_count": float(active["promo_type"].eq("percentage").sum()) if "promo_type" in active.columns else 0.0,
            "active_fixed_amount_promo_count": float(active["promo_type"].eq("fixed_amount").sum()) if "promo_type" in active.columns else 0.0,
            "avg_active_discount_value": float(active["discount_value"].mean()) if len(active) and "discount_value" in active.columns else 0.0,
            "max_active_discount_value": float(active["discount_value"].max()) if len(active) and "discount_value" in active.columns else 0.0,
            "promo_avg_duration_days": float((active["end_date"] - active["start_date"]).dt.days.mean()) if len(active) else 0.0,
            "has_stackable_promo": 1.0 if (len(active) and "stackable_flag" in active.columns and active["stackable_flag"].any()) else 0.0,
        }

    rows = []
    for date in normalized_dates:
        if pd.isna(date):
            rows.append({"Date": pd.NaT, **{column: 0.0 for column in PROMOTION_COLUMNS}})
            continue
        rows.append({"Date": date, **stats_by_date.get(date, {column: 0.0 for column in PROMOTION_COLUMNS})})

    return pd.DataFrame(rows)


# === CÁC HÀM XỬ LÝ FEATURE MỚI BỔ SUNG ===

def _payments_daily(payments_path: str | Path = "payments.csv", orders_path: str | Path = "orders.csv") -> pd.DataFrame:
    try:
        pay = pd.read_csv(payments_path, usecols=["order_id", "payment_method"])
        ord_ = pd.read_csv(orders_path, usecols=["order_id", "order_date"], parse_dates=["order_date"])
        merged = pay.merge(ord_, on="order_id", how="left")
        merged["Date"] = merged["order_date"].dt.normalize()
        
        daily = merged.groupby("Date").agg(pay_order_count=("order_id", "nunique")).reset_index()
        
        method = merged["payment_method"].astype(str)
        daily["credit_card_count"] = method.eq("credit_card").groupby(merged["Date"]).sum().reindex(daily["Date"]).to_numpy()
        daily["cod_count"] = method.eq("cod").groupby(merged["Date"]).sum().reindex(daily["Date"]).to_numpy()
        
        denom = daily["pay_order_count"].replace(0, np.nan)
        daily["credit_card_share"] = daily["credit_card_count"] / denom
        daily["cod_share"] = daily["cod_count"] / denom
        
        return daily[["Date", "credit_card_share", "cod_share"]].fillna(0.0)
    except FileNotFoundError:
        return pd.DataFrame(columns=["Date", "credit_card_share", "cod_share"])

def _shipments_daily(shipments_path: str | Path = "shipments.csv", orders_path: str | Path = "orders.csv") -> pd.DataFrame:
    try:
        ships = pd.read_csv(shipments_path, usecols=["order_id", "ship_date", "delivery_date", "shipping_fee"], parse_dates=["ship_date", "delivery_date"])
        ord_ = pd.read_csv(orders_path, usecols=["order_id", "order_date"], parse_dates=["order_date"])
        merged = ships.merge(ord_, on="order_id", how="left")
        merged["Date"] = merged["order_date"].dt.normalize()
        
        merged["delivery_delay"] = (merged["delivery_date"] - merged["ship_date"]).dt.days
        merged["is_free_shipping"] = merged["shipping_fee"] == 0
        
        daily = merged.groupby("Date").agg(
            avg_delivery_delay_days=("delivery_delay", "mean"),
            free_shipping_count=("is_free_shipping", "sum"),
            ship_order_count=("order_id", "nunique")
        ).reset_index()
        
        denom = daily["ship_order_count"].replace(0, np.nan)
        daily["free_shipping_share"] = daily["free_shipping_count"] / denom
        
        return daily[["Date", "avg_delivery_delay_days", "free_shipping_share"]].fillna(0.0)
    except FileNotFoundError:
        return pd.DataFrame(columns=["Date", "avg_delivery_delay_days", "free_shipping_share"])

def _returns_daily(returns_path: str | Path = "returns.csv") -> pd.DataFrame:
    try:
        returns = pd.read_csv(returns_path, usecols=["return_date", "return_reason"], parse_dates=["return_date"])
        returns["Date"] = returns["return_date"].dt.normalize()
        
        reason = returns["return_reason"].astype(str)
        daily = pd.DataFrame({"Date": returns["Date"].dropna().unique()})
        
        daily["defective_return_count"] = reason.eq("defective").groupby(returns["Date"]).sum().reindex(daily["Date"]).to_numpy()
        daily["wrong_size_count"] = reason.eq("wrong_size").groupby(returns["Date"]).sum().reindex(daily["Date"]).to_numpy()
        
        return daily.fillna(0.0)
    except FileNotFoundError:
        return pd.DataFrame(columns=["Date", "defective_return_count", "wrong_size_count"])

def _customers_orders_daily(orders_path: str | Path = "orders.csv") -> pd.DataFrame:
    try:
        orders = pd.read_csv(orders_path, usecols=["customer_id", "order_date"], parse_dates=["order_date"])
        orders["Date"] = orders["order_date"].dt.normalize()
        
        orders["first_order_date"] = orders.groupby("customer_id")["Date"].transform("min")
        orders["is_new_customer"] = orders["Date"] == orders["first_order_date"]
        
        daily = orders.groupby("Date").agg(
            new_customer_count=("is_new_customer", "sum"),
            total_customers=("customer_id", "count")
        ).reset_index()
        
        daily["returning_customer_count"] = daily["total_customers"] - daily["new_customer_count"]
        return daily[["Date", "new_customer_count", "returning_customer_count"]].fillna(0.0)
    except FileNotFoundError:
        return pd.DataFrame(columns=["Date", "new_customer_count", "returning_customer_count"])

def _inventory_monthly_to_daily(inventory_path: str | Path = "inventory.csv") -> pd.DataFrame:
    try:
        inv = pd.read_csv(inventory_path, usecols=["snapshot_date", "reorder_flag", "stockout_flag"], parse_dates=["snapshot_date"])
        inv["Date"] = inv["snapshot_date"].dt.normalize()
        
        monthly = inv.groupby("Date").agg(
            reorder_flag_pct=("reorder_flag", "mean"),
            stockout_flag_pct=("stockout_flag", "mean")
        ).reset_index()
        
        # Snapshot áp dụng cho tháng tiếp theo
        monthly["Date"] = monthly["Date"] + pd.offsets.MonthBegin(1)
        return monthly
    except FileNotFoundError:
        return pd.DataFrame(columns=["Date", "reorder_flag_pct", "stockout_flag_pct"])


def _historical_non_promotion_features() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    history = _orders_daily().merge(_order_items_daily(), on="Date", how="outer")
    history = history.merge(_web_daily(), on="Date", how="outer")
    
    # =========================================================================
    # FEATURE TOGGLES:
    # Nếu bạn KHÔNG muốn dùng một bảng nào đó (hoặc bị lỗi), hãy thêm `#` vào 
    # đầu dòng tương ứng để tắt nó đi. Cột nào bị tắt sẽ tự động được gán = 0.0.
    # =========================================================================
    
    # history = history.merge(_payments_daily(), on="Date", how="outer")
    # history = history.merge(_shipments_daily(), on="Date", how="outer")
    # history = history.merge(_returns_daily(), on="Date", how="outer")
    # history = history.merge(_customers_orders_daily(), on="Date", how="outer")
    # history = history.merge(_inventory_monthly_to_daily(), on="Date", how="outer")
    
    # # Nội suy (forward fill) riêng cho inventory vì dữ liệu snapshot chỉ có ngày mùng 1 hàng tháng
    # history = history.sort_values("Date")
    # for col in ["reorder_flag_pct", "stockout_flag_pct"]:
    #     if col in history.columns:
    #         history[col] = history[col].ffill()

    # =========================================================================

    history = history.sort_values("Date").reset_index(drop=True)
    history[NON_PROMOTION_COLUMNS] = history[NON_PROMOTION_COLUMNS].fillna(0.0).astype("float64")

    profile = history.assign(
        month=history["Date"].dt.month,
        day=history["Date"].dt.day,
    ).groupby(["month", "day"], dropna=False)[NON_PROMOTION_COLUMNS].mean()
    means = history[NON_PROMOTION_COLUMNS].mean(numeric_only=True).reindex(NON_PROMOTION_COLUMNS).fillna(0.0).astype("float64")
    return history, profile, means


def build_external_features(dates: pd.Series) -> pd.DataFrame:
    normalized_dates = pd.to_datetime(dates, errors="coerce").dt.normalize()
    out = pd.DataFrame({"Date": normalized_dates, "_row_id": np.arange(len(normalized_dates), dtype=int)})

    history, profile, means = _historical_non_promotion_features()
    history_non_promo = history[["Date", *NON_PROMOTION_COLUMNS]]
    out = out.merge(history_non_promo, on="Date", how="left")
    out[NON_PROMOTION_COLUMNS] = out[NON_PROMOTION_COLUMNS].astype("float64")

    missing_non_promo = out[NON_PROMOTION_COLUMNS].isna().any(axis=1)
    profile_fill_mask = missing_non_promo & out["Date"].notna()
    if profile_fill_mask.any():
        lookup_index = pd.MultiIndex.from_arrays(
            [
                out.loc[profile_fill_mask, "Date"].dt.month,
                out.loc[profile_fill_mask, "Date"].dt.day,
            ]
        )
        profile_filled = profile.reindex(lookup_index)
        profile_filled = profile_filled.fillna(means).fillna(0.0)
        out.loc[profile_fill_mask, NON_PROMOTION_COLUMNS] = profile_filled.to_numpy(dtype="float64")

    nat_mask = out["Date"].isna()
    if nat_mask.any():
        repeated_means = np.tile(means.to_numpy(dtype="float64"), (int(nat_mask.sum()), 1))
        out.loc[nat_mask, NON_PROMOTION_COLUMNS] = repeated_means
    out[NON_PROMOTION_COLUMNS] = out[NON_PROMOTION_COLUMNS].fillna(0.0)

    promo = _promotions_daily(out["Date"])
    promo["_row_id"] = out["_row_id"].to_numpy()
    out = out.merge(promo.drop(columns=["Date"]), on="_row_id", how="left")

    for column in FEATURE_COLUMNS:
        if column not in out.columns:
            out[column] = 0.0

    out = out.sort_values("_row_id").drop(columns=["_row_id"])
    out = out[["Date", *FEATURE_COLUMNS]]
    return out.fillna(0.0)
