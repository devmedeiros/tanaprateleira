create or replace view `catalog-impacta-capstone`.gold.inventory_forecast as 
with inv as (
  select inventory_eod as initial_inventory
  from `catalog-impacta-capstone`.gold.inventory_daily_sku1
  where event_date = (
    select max(event_date) from `catalog-impacta-capstone`.gold.inventory_daily_sku1
  )
),
weekly as (
  select
    cast(date_trunc('week', event_date) as date) as week_monday,
    ceiling(sum(demand_prediction)) as weekly_demand
  from `catalog-impacta-capstone`.gold.demand_prediction_28days
  group by 1
),
agg as (
  select
    w.week_monday,
    w.weekly_demand,
    sum(w.weekly_demand) over (
      order by w.week_monday
      rows between unbounded preceding and current row
    ) as weekly_demand_cum
  from weekly w
)
select
  a.week_monday,
  a.weekly_demand,
  a.weekly_demand_cum,
  (i.initial_inventory - a.weekly_demand_cum) as remaining_inventory
from agg a
cross join inv i
order by a.week_monday;