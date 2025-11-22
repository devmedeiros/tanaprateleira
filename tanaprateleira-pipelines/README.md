# tanaprateleira-pipelines

Este diretório contém o código-fonte do pipeline 'tanaprateleira-pipelines':

- `explorations`: Notebooks ad-hoc para explorar os dados processados pelo pipeline.
- `transformations`: Definições de datasets e transformações.

## Como começar

Para começar, acesse a pasta `transformations`, onde está a maior parte do código relevante:

* Por convenção, cada dataset em `transformations` está em um arquivo separado.
* Veja o exemplo em "sample_users_tanaprateleira_pipelines.sql" para se familiarizar com a sintaxe.
  Mais detalhes sobre a sintaxe: https://docs.databricks.com/dlt/sql-ref.html.
* Use `Run file` para executar e visualizar uma transformação individual.
* Use `Run pipeline` para executar todas as transformações do pipeline.
* Use `+ Add` no navegador de arquivos para adicionar uma nova definição de dataset.
* Use `Schedule` para agendar execuções do pipeline.

Para tutoriais e referências, acesse https://docs.databricks.com/dlt.
