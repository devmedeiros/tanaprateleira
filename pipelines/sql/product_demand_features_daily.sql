create table if not exists workspace.gold.product_demand_features_daily (
  order_id string comment 'Id único do pedido',
  order_approved_at timestamp comment 'Data de aprovação do pedido',
  product_id string comment 'Id único do produto',
  product_category_name string comment 'Categoria do produto',
  product_name_lenght int comment 'Tamanho do nome do produto',
  product_description_lenght int comment 'Tamanho da descrição do produto',
  product_weight_g int comment 'Peso do produto',
  product_length_cm int comment 'Comprimento do produto',
  product_height_cm int comment 'Altura do produto',
  product_width_cm int comment 'Largura do produto',
  quantity int comment 'Quantidade',
  total_price double comment 'Valor total',
  total_freight_value double comment 'Valor total do frete'
) comment 'A tabela `product_demand_features_daily` reune _features_ dos produtos e a data dos pedidos que foram feitos dos mesmos.';

insert overwrite table workspace.gold.product_demand_features_daily
select 
    o.order_id, 
    o.order_approved_at,
    i.product_id,
    p.product_category_name,
    p.product_name_lenght,
    p.product_description_lenght,
    p.product_weight_g,
    p.product_length_cm,
    p.product_height_cm,
    p.product_width_cm,
    max(i.order_item_id) as quantity,
    sum(i.price) as total_price,
    sum(i.freight_value) as total_freight_value
from 
  workspace.silver.orders as o
  inner join workspace.silver.order_items i on i.order_id = o.order_id
  inner join workspace.silver.products p on p.product_id = i.product_id
where 
  o.order_status not in ('unavailable', 'canceled')
  and o.order_approved_at is not null
group by all;