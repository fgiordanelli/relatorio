# Relatório Financeiro - De/Para

Aplicação Streamlit para análise financeira com mapeamento automático de categorias.

## Como usar

1. Faça upload do arquivo de extrato bancário (CSV, XLSX ou XLS)
2. Faça upload do arquivo De→Para (CSV com colunas: de, para)
3. Visualize os relatórios gerados automaticamente

## Funcionalidades

- Categorização automática de transações
- Análise de vendas por dia da semana
- Gráfico de faturamento semanal
- Detecção de estornos
- Cálculo de CMV e custos com funcionários
- Exportação para Excel

## Instalação local

```bash
pip install -r requirements.txt
streamlit run app_streamlit_depara.py
```

