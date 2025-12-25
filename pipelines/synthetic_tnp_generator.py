# Databricks notebook source
# Databricks / PySpark
# Synthetic TNP generator for a single SKU
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, date, time, timedelta
from zoneinfo import ZoneInfo

import math
import random
import numpy as np

from pyspark.sql import SparkSession, Row, DataFrame
from pyspark.sql.types import (
    StructType, StructField, StringType, TimestampType, IntegerType
)


@dataclass
class ErrorRates:
    duplicate_rate: float = 0.0
    null_status_rate: float = 0.0
    null_event_ts_rate: float = 0.0
    out_of_order_rate: float = 0.0
    tz_inconsistency_rate: float = 0.0
    placed_outlier_rate: float = 0.0


def _parse_error_rates(d: Optional[Dict]) -> ErrorRates:
    d = d or {}
    return ErrorRates(
        duplicate_rate=float(d.get("duplicate_rate", 0.0)),
        null_status_rate=float(d.get("null_status_rate", 0.0)),
        null_event_ts_rate=float(d.get("null_event_ts_rate", 0.0)),
        out_of_order_rate=float(d.get("out_of_order_rate", 0.0)),
        tz_inconsistency_rate=float(d.get("tz_inconsistency_rate", 0.0)),
        placed_outlier_rate=float(d.get("placed_outlier_rate", 0.0)),
    )


def _daterange(d0: date, d1: date) -> List[date]:
    ndays = (d1 - d0).days + 1
    return [d0 + timedelta(days=i) for i in range(ndays)]


def _second_sunday(year: int, month: int) -> date:
    # Find second Sunday of given month
    d = date(year, month, 1)
    # weekday(): Monday=0 ... Sunday=6
    first_sunday_offset = (6 - d.weekday()) % 7
    first_sunday = d + timedelta(days=first_sunday_offset)
    return first_sunday + timedelta(days=7)


def _last_friday(year: int, month: int) -> date:
    # last day of month
    if month == 12:
        last = date(year, 12, 31)
    else:
        last = date(year, month + 1, 1) - timedelta(days=1)
    offset = (last.weekday() - 4) % 7  # Friday=4
    return last - timedelta(days=offset)


def _first_monday_after(d: date) -> date:
    # Monday=0 ... Sunday=6
    return d + timedelta(days=((7 - d.weekday()) % 7 or 7))


def _commercial_holidays_for_year(year: int) -> Dict[str, date]:
    # Fixed and movable dates
    holidays = {
        "natal": date(year, 12, 25),
        "dia_das_criancas": date(year, 10, 12),
        "dia_dos_namorados": date(year, 6, 12),
    }
    holidays["dia_das_maes"] = _second_sunday(year, 5)
    holidays["dia_dos_pais"] = _second_sunday(year, 8)
    bf = _last_friday(year, 11)
    holidays["black_friday"] = bf
    holidays["cyber_monday"] = _first_monday_after(bf)
    return holidays


def _build_holiday_multiplier_map(
    start_d: date,
    end_d: date,
    holiday_multipliers: Optional[Dict[str, Dict]]
) -> Dict[date, float]:
    # Defaults if not provided
    if not holiday_multipliers:
        holiday_multipliers = {
            "natal": {"multiplier": 2.0, "pre_days": 2, "post_days": 0},
            "dia_das_criancas": {"multiplier": 1.5, "pre_days": 2, "post_days": 0},
            "dia_das_maes": {"multiplier": 1.8, "pre_days": 2, "post_days": 0},
            "dia_dos_pais": {"multiplier": 1.6, "pre_days": 2, "post_days": 0},
            "dia_dos_namorados": {"multiplier": 1.7, "pre_days": 2, "post_days": 0},
            "black_friday": {"multiplier": 3.0, "pre_days": 3, "post_days": 3},
            "cyber_monday": {"multiplier": 2.0, "pre_days": 0, "post_days": 0},
        }

    years = set([start_d.year, end_d.year])
    if end_d.year - start_d.year > 1:
        years.update(range(start_d.year, end_d.year + 1))

    multipliers_by_date: Dict[date, float] = {}
    for y in years:
        cal = _commercial_holidays_for_year(y)
        for hname, hdate in cal.items():
            spec = holiday_multipliers.get(hname)
            if not spec:
                continue
            mult = float(spec.get("multiplier", 1.0))
            pre = int(spec.get("pre_days", 0))
            post = int(spec.get("post_days", 0))
            for offs in range(-pre, post + 1):
                dt = hdate + timedelta(days=offs)
                if start_d <= dt <= end_d:
                    multipliers_by_date[dt] = max(multipliers_by_date.get(dt, 1.0), mult)
    return multipliers_by_date


def _daily_trend_multiplier(days_since_start: int, monthly_rate: float) -> float:
    # Convert monthly growth to daily compounding over average month length
    # avg month days ~ 30.4375
    if monthly_rate == 0:
        return 1.0
    return (1.0 + monthly_rate) ** (days_since_start / 30.4375)


def _expected_daily_demand(
    d: date,
    start_d: date,
    base_daily: float,
    monthly_rate: float,
    weekly_seasonality: List[float],
    holiday_mult_map: Dict[date, float],
) -> float:
    w = d.weekday()  # Monday=0
    days = (d - start_d).days
    return (
        base_daily
        * _daily_trend_multiplier(days, monthly_rate)
        * weekly_seasonality[w]
        * holiday_mult_map.get(d, 1.0)
    )


@dataclass
class QtyDistConfig:
    small_prob: float = 0.96
    small_weights: Tuple[float, float, float] = (0.6, 0.3, 0.1)  # for 1,2,3
    pareto_alpha: float = 2.5
    pareto_xm: int = 4  # minimum of pareto part
    pareto_cap: int = 50  # cap for sanity


def _qty_sample(rng: np.random.RandomState, cfg: QtyDistConfig) -> int:
    if rng.rand() < cfg.small_prob:
        # 1-3
        return rng.choice([1, 2, 3], p=np.array(cfg.small_weights) / sum(cfg.small_weights))
    # Pareto tail: x = xm * (1 - U)^(-1/alpha)
    u = max(1e-9, min(1 - 1e-9, rng.rand()))
    x = cfg.pareto_xm * ((1 - u) ** (-1.0 / cfg.pareto_alpha))
    q = int(math.floor(x))
    q = min(max(q, cfg.pareto_xm), cfg.pareto_cap)
    return q


def _qty_mean(cfg: QtyDistConfig) -> float:
    # E = p*E_small + (1-p)*E_pareto_capped (approximate by integral)
    e_small = 1 * cfg.small_weights[0] + 2 * cfg.small_weights[1] + 3 * cfg.small_weights[2]
    e_small /= sum(cfg.small_weights)
    # Exact Pareto mean without cap: alpha*xm/(alpha-1)
    if cfg.pareto_alpha <= 1:
        e_pareto = cfg.pareto_xm * 10.0
    else:
        # crude adjust for cap
        raw = cfg.pareto_alpha * cfg.pareto_xm / (cfg.pareto_alpha - 1.0)
        e_pareto = min(raw, cfg.pareto_cap * 0.9)
    return cfg.small_prob * e_small + (1.0 - cfg.small_prob) * e_pareto


def _iso_local_dt(d: date, h: int, m: int, s: int, tz: str) -> datetime:
    return datetime.combine(d, time(h, m, s, tzinfo=ZoneInfo(tz)))


def _forecast_approved_qty(
    start_on: date,
    days_ahead: int,
    start_d: date,
    base_daily: float,
    monthly_rate: float,
    weekly_seasonality: List[float],
    holiday_mult_map: Dict[date, float],
    mean_order_qty: float,
    approval_share: float,
) -> int:
    total = 0.0
    for i in range(days_ahead):
        dd = start_on + timedelta(days=i)
        exp_orders = _expected_daily_demand(dd, start_d, base_daily, monthly_rate, weekly_seasonality, holiday_mult_map)
        total += exp_orders * approval_share * mean_order_qty
    return max(0, int(math.ceil(total)))


def generate_synthetic_tnp(
    spark: SparkSession,
    start_date: str,
    end_date: str,
    tz: str = "America/Sao_Paulo",
    initial_stock: int = 100,
    base_daily_demand: float = 20.0,
    demand_trend_rate_monthly: float = 0.02,
    weekly_seasonality: Optional[List[float]] = None,
    holiday_multipliers: Optional[Dict[str, Dict]] = None,
    sunday_target_weeks_coverage: int = 3,
    emergency_restock_coverage_days: int = 5,
    approval_share: float = 0.7,
    cancel_rate: float = 0.2,
    placed_rate: float = 0.5,  # only used when cancel_rate not provided; by default we rely on cancel_rate
    seed: int = 42,
    catalog: Optional[str] = None,
    schema: Optional[str] = None,
    sales_table: Optional[str] = None,
    acquisitions_table: Optional[str] = None,
    write_mode: str = "none",  # 'none' | 'table' | 'view'
    save_mode: str = "overwrite",  # 'overwrite' | 'append'
    inject_errors: bool = False,
    error_rates: Optional[Dict] = None,
) -> Tuple[DataFrame, DataFrame, int]:
    """
    Gera dados sintéticos de vendas (sales) e aquisições de estoque (stock_acquisitions) para 1 SKU,
    com consistência de estoque e regras de reabastecimento.

    Parâmetros principais:
      - start_date, end_date: 'YYYY-MM-DD'
      - tz: timezone (ex.: 'America/Sao_Paulo')
      - initial_stock: estoque inicial
      - base_daily_demand: média inicial de pedidos/dia
      - demand_trend_rate_monthly: taxa de crescimento mensal (ex.: 0.02)
      - weekly_seasonality: lista de 7 multiplicadores (seg..dom)
      - holiday_multipliers: dict {holiday_name: {"multiplier": float, "pre_days": int, "post_days": int}}
      - sunday_target_weeks_coverage: semanas de cobertura alvo no reabastecimento de domingo
      - emergency_restock_coverage_days: dias de cobertura no reabastecimento emergencial
      - approval_share: fração esperada de pedidos que viram 'approved' quando há estoque
      - cancel_rate: probabilidade de cancelamento entre pedidos não aprovados
      - placed_rate: probabilidade de um pedido não aprovado permanecer 'placed' (se cancel_rate não for usado)
      - seed: semente para reprodutibilidade
      - write_mode: 'none' | 'table' | 'view'
      - save_mode: 'overwrite' | 'append'
      - inject_errors: habilita injeção de “erros” controlados (sem quebrar estoque)
      - error_rates: {'duplicate_rate','null_status_rate','null_event_ts_rate','out_of_order_rate','tz_inconsistency_rate','placed_outlier_rate'}

    Regras implementadas:
      - 'approved' só ocorre se houver estoque suficiente; sem estoque negativo.
      - Reabastece aos domingos (reason='planned'), com exceção emergencial quando estoque chega a 0.
      - No domingo, se o estoque final da última semana aumentou vs a semana anterior, não comprar; exceto se a próxima semana tiver feriado comercial.
      - Conservação: initial_stock + Σ(recebido) - Σ(approved.qty) = final_stock >= 0.
      - Erros injetados apenas em placed/canceled (nunca alteram approved), preservando consistência de estoque.

    Retorno:
      (sales_df, stock_acquisitions_df, final_stock)
    """
    # --- Validations and defaults ---
    d0 = datetime.strptime(start_date, "%Y-%m-%d").date()
    d1 = datetime.strptime(end_date, "%Y-%m-%d").date()
    if d1 < d0:
        raise ValueError("end_date must be >= start_date")

    if weekly_seasonality is None:
        weekly_seasonality = [1.0, 1.0, 1.0, 1.0, 1.1, 1.3, 0.8]  # Mon..Sun
    if len(weekly_seasonality) != 7:
        raise ValueError("weekly_seasonality must have length 7 (Mon..Sun)")

    if write_mode not in ("none", "table", "view"):
        raise ValueError("write_mode must be one of ['none','table','view']")
    if save_mode not in ("overwrite", "append"):
        raise ValueError("save_mode must be one of ['overwrite','append']")

    tzinfo = ZoneInfo(tz)

    # Error rates
    er = _parse_error_rates(error_rates if inject_errors else None)

    rng = np.random.RandomState(seed)
    py_rng = random.Random(seed + 7)

    # Holiday multipliers per date
    hol_mult_map = _build_holiday_multiplier_map(d0, d1, holiday_multipliers)
    holiday_dates_set = set(hol_mult_map.keys())

    # Quantity distribution
    qty_cfg = QtyDistConfig()
    mean_order_qty = _qty_mean(qty_cfg)

    # Iteration state
    stock = int(initial_stock)
    sales_records: List[Dict] = []
    acq_records: List[Dict] = []

    sales_seq = 0
    acq_seq = 0

    # For Sunday re-stock decision: track end-of-week stock (week ends on Saturday)
    # Keep last two weeks' ending stocks
    last_week_end_stock: Optional[int] = None
    prev_week_end_stock: Optional[int] = None

    all_days = _daterange(d0, d1)

    def next_sales_id(n: int) -> str:
        return f"S{n:08d}"

    def next_acq_id(n: int) -> str:
        return f"A{n:08d}"

    def add_acquisition(dt: datetime, reason: str, qty: int):
        nonlocal acq_seq, stock
        if qty <= 0:
            return
        acq_seq += 1
        acq_records.append({
            "acquisition_id": next_acq_id(acq_seq),
            "event_ts": dt,
            "reason": reason,
            "quantity_received": int(qty),
        })
        stock += int(qty)

    def sunday_planned_should_buy(sunday_date: date) -> bool:
        nonlocal prev_week_end_stock, last_week_end_stock
        # Rule: if last week's ending stock increased vs previous week's, do not buy;
        # except if next week has a commercial holiday.
        increased = (
            last_week_end_stock is not None
            and prev_week_end_stock is not None
            and last_week_end_stock > prev_week_end_stock
        )
        # Check holidays in next 7 days (Sun..Sat)
        next_week_has_holiday = any(
            (sunday_date + timedelta(days=off)) in holiday_dates_set for off in range(0, 7)
        )
        if increased and not next_week_has_holiday:
            return False
        return True

    def sunday_planned_qty(sunday_date: date) -> int:
        # Target coverage for next N weeks
        days_cov = max(1, int(sunday_target_weeks_coverage) * 7)
        need = _forecast_approved_qty(
            start_on=sunday_date,
            days_ahead=days_cov,
            start_d=d0,
            base_daily=base_daily_demand,
            monthly_rate=demand_trend_rate_monthly,
            weekly_seasonality=weekly_seasonality,
            holiday_mult_map=hol_mult_map,
            mean_order_qty=mean_order_qty,
            approval_share=approval_share,
        )
        return max(0, need - stock)

    def emergency_qty(today: date) -> int:
        # Coverage for next emergency_restock_coverage_days
        days_cov = max(1, int(emergency_restock_coverage_days))
        need = _forecast_approved_qty(
            start_on=today,
            days_ahead=days_cov,
            start_d=d0,
            base_daily=base_daily_demand,
            monthly_rate=demand_trend_rate_monthly,
            weekly_seasonality=weekly_seasonality,
            holiday_mult_map=hol_mult_map,
            mean_order_qty=mean_order_qty,
            approval_share=approval_share,
        )
        return max(1, need)

    # Main generation loop
    for cur_day in all_days:
        # Sunday planned restock (06:00) before sales
        if cur_day.weekday() == 6:  # Sunday
            if sunday_planned_should_buy(cur_day):
                planned_q = sunday_planned_qty(cur_day)
                if planned_q > 0:
                    add_acquisition(_iso_local_dt(cur_day, 6, 0, 0, tz), "planned", planned_q)

        # Demand for the day
        exp_orders = _expected_daily_demand(
            cur_day, d0, base_daily_demand, demand_trend_rate_monthly, weekly_seasonality, hol_mult_map
        )
        n_orders = int(rng.poisson(lam=max(0.0, exp_orders)))

        # Random timestamps within the day (uniform)
        # generate strictly increasing seconds for processing, but we will not force strictness across the whole horizon
        seconds = sorted(rng.randint(0, 24 * 3600, size=n_orders))
        event_times = [_iso_local_dt(cur_day, 0, 0, 0, tz) + timedelta(seconds=int(s)) for s in seconds]

        # Sales creation
        for et in event_times:
            q = _qty_sample(rng, qty_cfg)
            sales_seq += 1
            status: str

            if stock >= q:
                # With stock: approval possible
                if rng.rand() < approval_share:
                    status = "approved"
                else:
                    # Not approved -> canceled with cancel_rate else placed
                    status = "canceled" if rng.rand() < cancel_rate else "placed"
            else:
                # No stock: cannot approve
                status = "canceled" if rng.rand() < cancel_rate else "placed"

            # Apply and maintain stock (only if approved)
            if status == "approved":
                # Safety guard
                if q > stock:
                    # Should never happen; defensive
                    status = "canceled"
                else:
                    stock -= q

            sales_records.append({
                "sales_id": next_sales_id(sales_seq),
                "event_ts": et,
                "quantity": int(q),
                "status": status,
            })

            # Emergency restock if stock hits 0 and more orders may come later today
            if status == "approved" and stock == 0:
                # Check if there's still time left in the day (and expected more events)
                # We trigger emergency once per zero hit; additional zero hits can trigger again
                emerg_q = emergency_qty(cur_day)
                # Schedule after current sale (e.g., +5 minutes)
                emerg_ts = et + timedelta(minutes=5)
                add_acquisition(emerg_ts, "emergency", emerg_q)

        # End of week (Saturday) bookkeeping
        if cur_day.weekday() == 5:  # Saturday
            prev_week_end_stock, last_week_end_stock = (
                last_week_end_stock,
                stock
            )

    final_stock = stock

    # --- Validations (pre-error-injection) ---
    total_approved_qty = sum(r["quantity"] for r in sales_records if r["status"] == "approved")
    total_recv = sum(r["quantity_received"] for r in acq_records)
    conservation = initial_stock + total_recv - total_approved_qty
    if conservation != final_stock or final_stock < 0:
        raise AssertionError("Inventory conservation failed.")

    # --- Error injection (only in placed/canceled to preserve stock correctness) ---
    if inject_errors and sales_records:
        idx_non_approved = [i for i, r in enumerate(sales_records) if r["status"] in ("placed", "canceled")]
        n_non = len(idx_non_approved)

        def pick_k(rate: float) -> int:
            return int(min(n_non, max(0, round(rate * n_non))))

        # Duplicates
        k_dup = pick_k(er.duplicate_rate)
        dup_idx = set(py_rng.sample(idx_non_approved, k_dup)) if k_dup > 0 else set()
        for i in sorted(dup_idx):
            # exact duplicate (same sales_id)
            sales_records.append(dict(sales_records[i]))

        # Null status
        k_null_status = pick_k(er.null_status_rate)
        for i in py_rng.sample(idx_non_approved, k_null_status):
            sales_records[i]["status"] = None

        # Null event_ts
        k_null_ts = pick_k(er.null_event_ts_rate)
        for i in py_rng.sample(idx_non_approved, k_null_ts):
            sales_records[i]["event_ts"] = None

        # Out-of-order timestamps: nudge a subset by +/- up to 2 hours
        k_ooo = pick_k(er.out_of_order_rate)
        for i in py_rng.sample(idx_non_approved, k_ooo):
            et = sales_records[i]["event_ts"]
            if isinstance(et, datetime):
                delta_sec = py_rng.randint(-7200, 7200)  # +/- 2h
                sales_records[i]["event_ts"] = et + timedelta(seconds=delta_sec)

        # TZ inconsistency: shift by +/- 1 hour (simulate offset issues)
        k_tz = pick_k(er.tz_inconsistency_rate)
        for i in py_rng.sample(idx_non_approved, k_tz):
            et = sales_records[i]["event_ts"]
            if isinstance(et, datetime):
                shift = py_rng.choice([-1, 1])
                sales_records[i]["event_ts"] = et + timedelta(hours=shift)

        # Outlier quantities for placed/canceled only
        k_outlier = pick_k(er.placed_outlier_rate)
        for i in py_rng.sample(idx_non_approved, k_outlier):
            # large plausible outlier that still doesn't affect inventory because not approved
            sales_records[i]["quantity"] = int(min(500, max(1, int(rng.pareto(2.0) * 50))))

    # --- Build Spark DataFrames ---
    sales_schema = StructType([
        StructField("sales_id", StringType(), nullable=False),
        StructField("event_ts", TimestampType(), nullable=True),
        StructField("quantity", IntegerType(), nullable=False),
        StructField("status", StringType(), nullable=True),
    ])
    acq_schema = StructType([
        StructField("acquisition_id", StringType(), nullable=False),
        StructField("event_ts", TimestampType(), nullable=False),
        StructField("reason", StringType(), nullable=False),
        StructField("quantity_received", IntegerType(), nullable=False),
    ])

    sales_df = spark.createDataFrame(
        [Row(**r) for r in sales_records],
        schema=sales_schema
    )
    acquisitions_df = spark.createDataFrame(
        [Row(**r) for r in acq_records],
        schema=acq_schema
    )

    # --- Optional write ---
    full_sales_name = None
    full_acq_name = None
    if write_mode in ("table", "view"):
        if not (catalog and schema and sales_table and acquisitions_table):
            raise ValueError("catalog, schema, sales_table and acquisitions_table are required for write_mode != 'none'.")

        full_sales_name = f"{catalog}.{schema}.{sales_table}"
        full_acq_name = f"{catalog}.{schema}.{acquisitions_table}"

        # Ensure schema exists (best-effort)
        try:
            spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
        except Exception:
            pass
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")

        if write_mode == "table":
            sales_df.write.mode(save_mode).saveAsTable(full_sales_name)
            acquisitions_df.write.mode(save_mode).saveAsTable(full_acq_name)
        elif write_mode == "view":
            # Create or replace views backed by current dataframes
            tmp_sales = f"tmp_sales_{abs(hash((full_sales_name, seed))) % 10_000_000}"
            tmp_acq = f"tmp_acq_{abs(hash((full_acq_name, seed))) % 10_000_000}"
            sales_df.createOrReplaceTempView(tmp_sales)
            acquisitions_df.createOrReplaceTempView(tmp_acq)
            spark.sql(f"CREATE OR REPLACE VIEW {full_sales_name} AS SELECT * FROM {tmp_sales}")
            spark.sql(f"CREATE OR REPLACE VIEW {full_acq_name} AS SELECT * FROM {tmp_acq}")

    # Final safety validations (types ok, approved never > stock at gen time was ensured in-loop)
    # Check timestamps are TimestampType (Spark) - already enforced by schema.

    return sales_df, acquisitions_df, int(final_stock)

# COMMAND ----------

# Exemplo de uso no Databricks

from pyspark.sql import functions as F

spark.conf.set("spark.sql.session.timeZone", "UTC")  # Spark armazena instantes; geramos datetimes 'aware' no tz local

sales_df, acquisitions_df, final_stock = generate_synthetic_tnp(
    spark=spark,
    start_date="2023-01-01",
    end_date="2025-12-20",
    tz="America/Sao_Paulo",
    initial_stock=0,
    base_daily_demand=35.0,
    demand_trend_rate_monthly=0.03,
    weekly_seasonality=[1.0, 1.0, 1.05, 1.05, 1.2, 1.4, 0.8],  # Mon..Sun
    holiday_multipliers={
        "natal": {"multiplier": 2.2, "pre_days": 3, "post_days": 0},
        "dia_das_criancas": {"multiplier": 1.6, "pre_days": 2, "post_days": 0},
        "dia_das_maes": {"multiplier": 1.9, "pre_days": 2, "post_days": 0},
        "dia_dos_pais": {"multiplier": 1.7, "pre_days": 2, "post_days": 0},
        "dia_dos_namorados": {"multiplier": 1.8, "pre_days": 2, "post_days": 0},
        "black_friday": {"multiplier": 3.2, "pre_days": 4, "post_days": 3},
        "cyber_monday": {"multiplier": 2.2, "pre_days": 0, "post_days": 0},
    },
    sunday_target_weeks_coverage=3,
    emergency_restock_coverage_days=5,
    approval_share=0.72,
    cancel_rate=0.25,
    placed_rate=0.5,  # usado apenas se cancel_rate não for fornecido; aqui cancel_rate prevalece
    seed=123,
    catalog="workspace",   # ajuste conforme seu ambiente (Unity Catalog: nome_do_catalogo)
    schema="raw",
    sales_table="sales_events_sku1",
    acquisitions_table="stock_acquisitions_sku1",
    write_mode="table",         # 'none' | 'table' | 'view'
    save_mode="overwrite",
    inject_errors=True,
    error_rates={
        "duplicate_rate": 0.01,
        "null_status_rate": 0.002,
        "null_event_ts_rate": 0.001,
        "out_of_order_rate": 0.005,
        "tz_inconsistency_rate": 0.002,
        "placed_outlier_rate": 0.001,
    }
)

# COMMAND ----------

# MAGIC %sql
# MAGIC select 
# MAGIC event_ts::date,
# MAGIC sum(quantity)
# MAGIC from workspace.raw.sales_events_sku1
# MAGIC where status = 'approved'
# MAGIC and sales_id not in (select sales_id from workspace.raw.sales_events_sku1 where status = 'canceled')
# MAGIC group by 1
# MAGIC order by event_ts

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC with params as (
# MAGIC   select
# MAGIC     cast('2023-01-01' as date) as d_start,
# MAGIC     cast('2025-12-20'   as date) as d_end,
# MAGIC     cast(0 as int) as initial_stock
# MAGIC ),
# MAGIC dates as (
# MAGIC   select explode(sequence(p.d_start, p.d_end, interval 1 day)) as dt
# MAGIC   from params p
# MAGIC ),
# MAGIC acq as (
# MAGIC   select
# MAGIC     to_date(from_utc_timestamp(event_ts,'America/Sao_Paulo')) as dt,
# MAGIC     sum(quantity_received) as qty_received
# MAGIC   from workspace.raw.stock_acquisitions_sku1
# MAGIC   group by 1
# MAGIC ),
# MAGIC sales_approved as (
# MAGIC   select
# MAGIC     to_date(from_utc_timestamp(event_ts,'America/Sao_Paulo')) as dt,
# MAGIC     sum(quantity) as qty_approved
# MAGIC   from workspace.raw.sales_events_sku1
# MAGIC   where status = 'approved'
# MAGIC   group by 1
# MAGIC ),
# MAGIC daily as (
# MAGIC   select
# MAGIC     d.dt,
# MAGIC     coalesce(a.qty_received, 0) as qty_received,
# MAGIC     coalesce(s.qty_approved, 0) as qty_approved
# MAGIC   from dates d
# MAGIC   left join acq a on a.dt = d.dt
# MAGIC   left join sales_approved s on s.dt = d.dt
# MAGIC )
# MAGIC select
# MAGIC   dt as date,
# MAGIC   (select initial_stock from params) 
# MAGIC   + sum(qty_received) over (order by dt rows between unbounded preceding and current row)
# MAGIC   - sum(qty_approved) over (order by dt rows between unbounded preceding and current row) as estoque_final
# MAGIC from daily
# MAGIC order by dt;
