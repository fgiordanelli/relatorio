# app.py
# ------------------------------------------------------------
# Relatório financeiro com De→Para aplicado na coluna "Destino"
# ------------------------------------------------------------
import io
import re
import csv
import unicodedata
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt


# -------------------- Utils --------------------
def normalize_text(s: str) -> str:
    """lowercase + sem acentos + espaços normalizados; mantém letras/números/._-"""
    # Import regex locally to avoid any accidental name shadowing
    import re as _re
    if pd.isna(s):
        return ""
    if not isinstance(s, str):
        s = str(s)
    try:
        s = s.strip().lower()
        s = unicodedata.normalize("NFKD", s)
        s = "".join(ch for ch in s if not unicodedata.combining(ch))
        s = _re.sub(r"[^a-z0-9\s\-\_\.]", " ", s)  # mantém letras, números, -, _, .
        s = _re.sub(r"\s+", " ", s).strip()
        return s
    except Exception:
        # Fallback extremamente conservador caso algo estranho ocorra
        cleaned = []
        for ch in str(s):
            if ("a" <= ch <= "z") or ("0" <= ch <= "9") or ch in " .-_":
                cleaned.append(ch)
        out = "".join(cleaned)
        return " ".join(out.split())


def to_numeric(series, decimal_hint=","):
    """Converte valores para float, suportando formato BR e US."""
    def parse_one(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip()
        if s == "":
            return np.nan
        # remove símbolos e espaços (mantém dígitos e sinais)
        s = re.sub(r"[^\d\-\.,]", "", s)
        # heurística: se tiver '.' e ',', assume BR (milhar '.' e decimal ',')
        if "." in s and "," in s:
            s = s.replace(".", "").replace(",", ".")
        else:
            if "," in s and "." not in s:  # só vírgula => decimal ','
                s = s.replace(",", ".")
        try:
            return float(s)
        except Exception:
            return np.nan
    return series.apply(parse_one)


def detect_csv_sep_encoding(file_bytes: bytes):
    """Tenta detectar separador usando csv.Sniffer; encoding já vem do Streamlit (bytes)."""
    sample = file_bytes[:4096].decode("utf-8", errors="ignore")
    try:
        dialect = csv.Sniffer().sniff(sample)
        sep = dialect.delimiter
    except Exception:
        sep = ";" if sample.count(";") >= sample.count(",") else ","
    return sep


def format_currency_br(x):
    try:
        return f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return x


def format_percent_br(x):
    try:
        return f"{x*100:.2f}%".replace(".", ",")
    except Exception:
        return x


def render_centered_table(df: pd.DataFrame):
    """Renderiza uma tabela HTML centralizada e com texto centralizado."""
    html = df.to_html(index=False)
    style = (
        "<style>"
        ".center-table{overflow-x:auto; overflow-y:auto; max-width:100%; max-height:480px;}"
        ".center-table table{margin-left:auto;margin-right:auto; border-collapse:collapse;}"
        ".center-table th,.center-table td{text-align:center !important; padding:4px 8px;}"
        "</style>"
    )
    st.markdown(style + f'<div class="center-table">{html}</div>', unsafe_allow_html=True)


def format_currency_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = out[col].map(format_currency_br)
    return out


def render_monthly_linear_scroller(monthly_linear: pd.DataFrame):
    """Renderiza N tabelas (uma por mês) lado a lado com rolagem horizontal."""
    months = list(dict.fromkeys(monthly_linear["month"].sort_values(ascending=False)))
    cards_html = []
    for m in months:
        sub = monthly_linear.loc[monthly_linear["month"] == m, ["Categoria", "Saidas"]].copy()
        sub = sub.sort_values("Saidas")
        sub_fmt = format_currency_columns(sub, ["Saidas"])  # R$
        html_table = sub_fmt.to_html(index=False)
        title = pd.to_datetime(m).strftime("%Y-%m")
        card = (
            f'<div class="month-card">'
            f'<div class="month-title">{title}</div>'
            f'{html_table}'
            f'</div>'
        )
        cards_html.append(card)
    style = (
        "<style>"
        ".monthly-scroller{display:flex; gap:16px; overflow-x:auto; padding:6px 2px;}"
        ".monthly-scroller .month-card{min-width:320px; max-width:320px;}"
        ".monthly-scroller table{margin-left:auto;margin-right:auto; border-collapse:collapse; width:100%;}"
        ".monthly-scroller th,.monthly-scroller td{text-align:center !important; padding:4px 8px;}"
        ".monthly-scroller .month-title{font-weight:600; text-align:center; margin:4px 0 8px;}"
        "</style>"
    )
    scroller = f"<div class='monthly-scroller'>{''.join(cards_html)}</div>"
    st.markdown(style + scroller, unsafe_allow_html=True)

# -------------------- De→Para --------------------
def load_depara(depara_file) -> list[dict]:
    """Lê o CSV de de→para e gera lista de regras: [{'keywords': [...], 'category': '...'}]."""
    # tentar ler como utf-8; se falhar, latin-1
    try:
        df = pd.read_csv(depara_file, encoding="utf-8")
    except Exception:
        depara_file.seek(0)
        df = pd.read_csv(depara_file, encoding="latin-1")

    if df.shape[1] < 2:
        raise ValueError("O CSV de 'de→para' precisa ter ao menos 2 colunas: 'de' e 'para'.")

    # detectar nomes das colunas (flexível)
    cols_norm = {c: normalize_text(c) for c in df.columns}
    de_col = None
    para_col = None
    for c, n in cols_norm.items():
        if n in ("de", "origem", "fornecedor", "palavra", "texto", "termo"):
            de_col = c
        if n in ("para", "categoria", "destino", "classe", "tag"):
            para_col = c
    if not de_col:
        de_col = df.columns[0]
    if not para_col:
        para_col = df.columns[1]

    # expandir termos múltiplos separados por , ; / |
    rules = []
    for _, row in df.iterrows():
        raw_de = str(row[de_col])
        category = str(row[para_col]).strip()
        if not category:
            continue
        keywords = [normalize_text(k) for k in re.split(r"[;,/|]", raw_de) if str(k).strip()]
        if not keywords:
            continue
        rules.append({"keywords": keywords, "category": category})

    # ordenar regras por comprimento do termo (prioriza match mais específico)
    expanded = []
    for r in rules:
        for kw in r["keywords"]:
            expanded.append((kw, r["category"]))
    expanded.sort(key=lambda t: len(t[0]), reverse=True)  # mais longo primeiro

    # volta para estrutura agrupada mantendo ordem
    ordered_rules = []
    for kw, cat in expanded:
        ordered_rules.append({"keywords": [kw], "category": cat})
    return ordered_rules


def apply_depara_on_destino(df_dest: pd.DataFrame, rules: list[dict], default_category="Outros"):
    """Aplica regras de de→para sobre a coluna normalizada do Destino."""
    cats = []
    for desc in df_dest["_dest_norm"]:
        found = None
        for rule in rules:
            for kw in rule["keywords"]:
                if kw and kw in desc:
                    found = rule["category"]
                    break
            if found:
                break
        cats.append(found if found else default_category)
    df = df_dest.copy()
    df["Categoria"] = cats
    return df


# -------------------- Extrato --------------------
def load_statement_use_destino_only(file, sep_input=None, encoding_input=None, decimal_hint=",", date_format_hint=None):
    """
    Lê o extrato (CSV/XLS/XLSX), identifica:
      - coluna Destino (prioriza exatamente 'Destino'; senão tenta variações),
      - coluna de Data,
      - coluna(s) de valor (débito/crédito ou valor único).
    Retorna DataFrame padronizado com: date, Destino, amount.
    """
    step = "início"
    try:
        name = file.name.lower()
        step = "ler-bytes"
        # Ler conteudo bruto (para detectar sep quando CSV)
        file_bytes = file.read()
        file.seek(0)

        if name.endswith((".xlsx", ".xls")):
            step = "read_excel"
            df = pd.read_excel(file)
        else:
            step = "detectar-separador"
            sep = sep_input or detect_csv_sep_encoding(file_bytes)
            encoding = encoding_input or "utf-8"
            step = f"read_csv utf-8 engine=python sep={sep!r}"
            try:
                df = pd.read_csv(file, sep=sep, encoding=encoding, engine="python")
            except Exception:
                file.seek(0)
                step = f"read_csv utf-8 engine=c sep={sep!r}"
                try:
                    df = pd.read_csv(file, sep=sep, encoding=encoding, engine="c", on_bad_lines="skip")
                except Exception:
                    file.seek(0)
                    step = f"read_csv latin-1 engine=python sep={sep!r}"
                    df = pd.read_csv(file, sep=sep, encoding="latin-1", engine="python")

        step = "remover-unnamed"
        # Remover colunas "Unnamed"
        df = df.loc[:, ~df.columns.astype(str).str.fullmatch(r"\s*Unnamed:?\s*\d+\s*", na=False)]

        step = "encontrar-destino"
        # Encontrar coluna "Destino"
        norm_cols = {c: normalize_text(c) for c in df.columns}
        dest_col = None
        # prioridade: exatamente "destino"
        for c in df.columns:
            if normalize_text(c) == "destino":
                dest_col = c
                break
        # senão, procurar similares
        if dest_col is None:
            for target in ["destino", "favorecido", "estabelecimento", "descricao", "descrição", "historico", "histórico"]:
                for c, n in norm_cols.items():
                    if target in n:
                        dest_col = c
                        break
                if dest_col:
                    break
        if dest_col is None:
            # fallback: primeira coluna de texto
            text_cols = df.select_dtypes(include=["object"]).columns.tolist()
            dest_col = text_cols[0] if text_cols else df.columns[0]

        # Colunas opcionais usadas para flag de Reserva Stone
        def find_col(targets: list[str]):
            for c, n in norm_cols.items():
                if n in targets:
                    return c
            return None
        origem_col = find_col(["origem"]) or "Origem" if "Origem" in df.columns else None
        origem_ag_col = find_col(["origem agencia", "origem agência"]) or next((c for c in df.columns if normalize_text(c) == "origem agencia"), None)
        origem_conta_col = find_col(["origem conta"]) or next((c for c in df.columns if normalize_text(c) == "origem conta"), None)
        dest_inst_col = find_col(["destino instituicao", "destino instituição"]) or next((c for c in df.columns if normalize_text(c) == "destino instituicao"), None)
        origem_inst_col = find_col(["origem instituicao", "origem instituição"]) or next((c for c in df.columns if normalize_text(c) == "origem instituicao"), None)

        step = "encontrar-data"
        # Encontrar data
        date_col = None
        for c, n in norm_cols.items():
            if any(t in n for t in ["data", "emissao", "emissão", "competencia", "lançamento", "lancamento", "post date", "trans date"]):
                date_col = c
                break
        if date_col is None:
            for c in df.columns:
                if "data" in normalize_text(c):
                    date_col = c
                    break

        step = "parse-data"
        if date_col:
            if date_format_hint:
                try:
                    date_series = pd.to_datetime(df[date_col], format=date_format_hint, errors="coerce")
                except Exception:
                    date_series = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
            else:
                date_series = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
        else:
            # se não achar data, usa data de hoje (não ideal, mas evita travar)
            date_series = pd.to_datetime("today").normalize()

        step = "encontrar-valor"
        # Encontrar valor
        amount_col = None
        debit_col = None
        credit_col = None
        for c, n in norm_cols.items():
            if "debito" in n or "débito" in n:
                debit_col = c
            if "credito" in n or "crédito" in n:
                credit_col = c
        for c, n in norm_cols.items():
            if any(t in n for t in ["valor", "amount", "total", "transacao", "transação"]):
                amount_col = amount_col or c

        step = "calcular-amount"
        if debit_col and credit_col:
            amt = to_numeric(df[credit_col], decimal_hint).fillna(0) - to_numeric(df[debit_col], decimal_hint).fillna(0)
        elif amount_col:
            amt = to_numeric(df[amount_col], decimal_hint)
        else:
            # fallback: última numérica
            num_cols = df.select_dtypes(include=["number"]).columns.tolist()
            amt = df[num_cols[-1]] if num_cols else pd.Series([np.nan] * len(df), dtype=float)

        step = "montar-output"
        out = pd.DataFrame({
            "date": date_series,
            "Destino": df[dest_col].astype(str),
            "amount": amt
        })
        # Colunas auxiliares para flags (se existirem)
        out["Origem"] = df[origem_col].astype(str) if origem_col else ""
        out["Origem Agência"] = df[origem_ag_col].astype(str) if origem_ag_col else ""
        out["Origem Conta"] = df[origem_conta_col].astype(str) if origem_conta_col else ""
        out["Destino Instituição"] = df[dest_inst_col].astype(str) if dest_inst_col else ""
        out["Origem Instituição"] = df[origem_inst_col].astype(str) if origem_inst_col else ""

        out["_dest_norm"] = out["Destino"].map(normalize_text)
        out["_origem_norm"] = out["Origem"].map(normalize_text)
        out["_origem_ag_norm"] = out["Origem Agência"].map(normalize_text)
        out["_origem_conta_norm"] = out["Origem Conta"].map(normalize_text)
        out["_dest_inst_norm"] = out["Destino Instituição"].map(normalize_text)
        out["_origem_inst_norm"] = out["Origem Instituição"].map(normalize_text)

        # Flag de Reserva Stone (Entrou/Saiu) conforme regras fornecidas
        mask_saiu = (
            (out["_origem_norm"] == "desconhecido") &
            (out["_origem_ag_norm"] == "desconhecido") &
            (out["_origem_conta_norm"] == "desconhecido") &
            (out["_origem_inst_norm"] == normalize_text("STONE INSTITUIÇÃO DE PAGAMENTO S.A."))
        )
        mask_entrou = (
            (out["_dest_norm"] == "desconhecido") &
            (out["_dest_inst_norm"] == normalize_text("STONE INSTITUIÇÃO DE PAGAMENTO S.A.")) &
            (out["_origem_norm"] == normalize_text("SALVATORE ALIMENTOS LTDA"))
        )
        out["ReservaStoneFlag"] = ""
        out.loc[mask_saiu, "ReservaStoneFlag"] = "Saiu"
        out.loc[mask_entrou, "ReservaStoneFlag"] = "Entrou"
        # Valor assinado: Saiu = positivo; Entrou = negativo
        signed = np.zeros(len(out))
        signed[mask_saiu.values] = np.abs(amt[mask_saiu].values)
        signed[mask_entrou.values] = -np.abs(amt[mask_entrou].values)
        out["ReservaStoneSigned"] = signed

        # Flag IFOOD: Origem Instituição = ZOOP TECNOLOGIA & INSTITUICAO DE PAGAMENTO S.A.
        ifood_norm = normalize_text("ZOOP TECNOLOGIA & INSTITUICAO DE PAGAMENTO S.A.")
        out["IfoodFlag"] = np.where(out["_origem_inst_norm"] == ifood_norm, "IFOOD", "")

        # Flag Investimento Empresa: PIX do Fabrício Giordanelli
        fabricio_norm = normalize_text("Fabrício Giordanelli")
        out["InvestimentoEmpresaFlag"] = np.where(
            out["_origem_norm"].str.contains("fabricio", na=False) & 
            out["_origem_norm"].str.contains("giordanelli", na=False),
            "Investimento Empresa",
            ""
        )

        # Flag Funcionários: Lista de funcionários conhecidos
        funcionarios_patterns = [
            ("patrick", "andrews"),
            ("maressa", "coelho"),
            ("vamberto", "barbosa"),
            ("alessandro", "silva", "barbosa"),
            ("joaldo", "gomes"),
            ("antonio", "orlando", "sousa")
        ]
        
        funcionario_mask = pd.Series([False] * len(out), dtype=bool)
        for pattern in funcionarios_patterns:
            if len(pattern) == 2:
                funcionario_mask |= (
                    out["_dest_norm"].str.contains(pattern[0], na=False) & 
                    out["_dest_norm"].str.contains(pattern[1], na=False)
                )
            elif len(pattern) == 3:
                funcionario_mask |= (
                    out["_dest_norm"].str.contains(pattern[0], na=False) & 
                    out["_dest_norm"].str.contains(pattern[1], na=False) &
                    out["_dest_norm"].str.contains(pattern[2], na=False)
                )
        
        out["FuncionarioFlag"] = np.where(funcionario_mask, "Funcionário", "")

        # Flag de Estorno: detectar transações com mesmo valor absoluto, sinais opostos, em janela temporal próxima
        out = out.sort_values("date").reset_index(drop=True)
        out["EstornoFlag"] = ""
        out["EstornoParId"] = -1
        
        # Parâmetros de detecção
        janela_minutos = 10  # janela de até 10 minutos
        tolerancia = 0.0  # estorno sempre é valor idêntico
        
        matched_indices = set()
        estorno_pairs = []
        
        for i in range(len(out)):
            if i in matched_indices:
                continue
            
            row_i = out.iloc[i]
            if pd.isna(row_i["date"]) or pd.isna(row_i["amount"]):
                continue
            
            date_i = pd.to_datetime(row_i["date"])
            amt_i = float(row_i["amount"])
            abs_amt_i = abs(amt_i)
            
            # Buscar par dentro da janela temporal
            for j in range(i + 1, len(out)):
                if j in matched_indices:
                    continue
                
                row_j = out.iloc[j]
                if pd.isna(row_j["date"]) or pd.isna(row_j["amount"]):
                    continue
                
                date_j = pd.to_datetime(row_j["date"])
                amt_j = float(row_j["amount"])
                abs_amt_j = abs(amt_j)
                
                # Verificar se está dentro da janela temporal (10 minutos)
                diff_minutos = abs((date_j - date_i).total_seconds() / 60)
                if diff_minutos > janela_minutos:
                    # Pode ter outras transações próximas, então continua (não break)
                    continue
                
                # Verificar se tem mesmo valor absoluto (com tolerância) e sinais opostos
                if (abs(abs_amt_i - abs_amt_j) <= tolerancia and 
                    np.sign(amt_i) != np.sign(amt_j)):
                    # Par de estorno encontrado!
                    pair_id = len(estorno_pairs)
                    estorno_pairs.append((i, j))
                    matched_indices.add(i)
                    matched_indices.add(j)
                    
                    # Original = transação que veio primeiro
                    # Estorno = transação que veio depois
                    out.at[i, "EstornoFlag"] = "Original"
                    out.at[i, "EstornoParId"] = pair_id
                    out.at[j, "EstornoFlag"] = "Estorno"
                    out.at[j, "EstornoParId"] = pair_id
                    break

        out = out[~out["amount"].isna()].reset_index(drop=True)
        return out
    except Exception as e:
        raise RuntimeError(f"Falha ao processar extrato na etapa '{step}': {e}")


def make_excel_report(df_cat: pd.DataFrame) -> bytes:
    """Gera um XLSX (bytes) com 3 abas: Lançamentos, Por Categoria, Mensal x Categoria."""
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        df_export = df_cat.copy()
        df_export["date"] = pd.to_datetime(df_export["date"]).dt.date
        df_export.to_excel(writer, sheet_name="Lançamentos", index=False)

        cat_sum = df_cat.groupby("Categoria", dropna=False)["amount"].sum().sort_values(ascending=True)
        cat_sum_df = cat_sum.reset_index()
        cat_sum_df.to_excel(writer, sheet_name="Por Categoria", index=False)

        monthly = df_cat.copy()
        monthly["month"] = pd.to_datetime(monthly["date"]).dt.to_period("M").dt.to_timestamp()
        piv = monthly.pivot_table(index="month", columns="Categoria", values="amount", aggfunc="sum").fillna(0.0)
        piv.reset_index().to_excel(writer, sheet_name="Mensal x Categoria", index=False)
    out.seek(0)
    return out.read()


# -------------------- UI --------------------
st.set_page_config(page_title="Relatório financeiro - De/Para (Destino)", layout="wide")
st.title("Relatório de Gastos • De→Para na coluna **Destino**")

with st.sidebar:
    st.markdown("### Passo a passo")
    st.markdown("1) Envie **Extrato** (CSV/XLSX)")
    st.markdown("2) Envie **De→Para corrigida** (CSV com 2 colunas: *de*, *para*)")
    st.markdown("3) Ajuste as opções e gere os downloads")

    st.divider()
    st.markdown("#### Opções de importação")
    sep_txt = st.text_input("Separador CSV do extrato (vazio = auto)", value="")
    enc_txt = st.text_input("Encoding do extrato (ex: utf-8, latin-1) (vazio = auto)", value="")
    decimal_hint = st.selectbox("Decimal do extrato", options=[",", "."], index=0)
    date_fmt_hint = st.text_input("Formato de data (opcional)", placeholder="Ex: %d/%m/%Y")

    st.divider()
    st.markdown("#### Mapeamento")
    default_category = st.text_input("Categoria padrão (sem match)", "Outros")

    st.divider()
    extrato_file = st.file_uploader("Extrato bancário (CSV, XLSX, XLS)", type=["csv", "xlsx", "xls"], key="extrato")
    depara_file = st.file_uploader("De→Para corrigida (CSV)", type=["csv"], key="depara")

if not extrato_file or not depara_file:
    st.info("Envie os dois arquivos para começar.")
    st.stop()

# Carregar arquivos
try:
    df_base = load_statement_use_destino_only(
        extrato_file,
        sep_input=(sep_txt or None),
        encoding_input=(enc_txt or None),
        decimal_hint=decimal_hint,
        date_format_hint=(date_fmt_hint or None),
    )
except Exception as e:
    st.error(f"Erro ao ler o **extrato**: {e}")
    st.stop()

try:
    rules = load_depara(depara_file)
except Exception as e:
    st.error(f"Erro ao ler o **de→para**: {e}")
    st.stop()

# Aplicar mapeamento sobre a coluna "Destino"
df_cat = apply_depara_on_destino(df_base, rules, default_category=default_category)

# Regra adicional: valores positivos viram "Entradas", exceto quando é o fornecedor específico
special_payee_norm = normalize_text("RIO QUALITY COMÉRCIO DE ALIMENTOS S/A")
mask_positive_not_special = (df_cat["amount"] > 0) & (~df_cat["_dest_norm"].str.contains(special_payee_norm, regex=False))
df_cat.loc[mask_positive_not_special, "Categoria"] = "Entradas"

# Regra para Investimento Empresa: PIX do Fabrício Giordanelli
mask_investimento = df_cat["InvestimentoEmpresaFlag"] == "Investimento Empresa"
df_cat.loc[mask_investimento, "Categoria"] = "Investimento Empresa"

# Regra para Funcionários: Pagamentos para funcionários conhecidos
mask_funcionarios = df_cat["FuncionarioFlag"] == "Funcionário"
df_cat.loc[mask_funcionarios, "Categoria"] = "Funcionários"

# (Removido mapeamento antigo de Reserva Stone por Destino)
# df_cat.loc[mask_reserva_stone, "Categoria"] = "Reserva Stone"

# ---- KPIs globais (primeiro bloco do relatório) ----
# Excluir linhas com flag de Reserva Stone, Estornos e Investimento Empresa das entradas/saídas
no_flag = df_cat.loc[
    ~df_cat["ReservaStoneFlag"].isin(["Saiu", "Entrou"]) &
    ~df_cat["EstornoFlag"].isin(["Original", "Estorno"]) &
    (df_cat["InvestimentoEmpresaFlag"] != "Investimento Empresa")
]
flagged = df_cat.loc[df_cat["ReservaStoneFlag"].isin(["Saiu", "Entrou"])]

# Totais sem as linhas flagadas
total_out = no_flag.loc[no_flag["amount"] < 0, "amount"].sum()
total_in = no_flag.loc[no_flag["amount"] > 0, "amount"].sum()
entradas_abs = abs(float(total_in))
saidas_abs = abs(float(total_out))
# Reserva Stone: soma dos valores assinados (Saiu positivo, Entrou negativo)
saiu_signed = flagged.loc[flagged["ReservaStoneFlag"] == "Saiu", "ReservaStoneSigned"].sum()
entrou_signed = flagged.loc[flagged["ReservaStoneFlag"] == "Entrou", "ReservaStoneSigned"].sum()
stone_total = saiu_signed + entrou_signed
saldo_abs = entradas_abs - saidas_abs - abs(float(stone_total))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Saídas (despesas)", format_currency_br(total_out))
c2.metric("Entradas (receitas)", format_currency_br(total_in))
c3.metric("Saldo líquido", format_currency_br(saldo_abs))
c4.metric("Reserva Stone", format_currency_br(stone_total))

num_in = int((flagged["ReservaStoneFlag"] == "Entrou").sum())
num_out = int((flagged["ReservaStoneFlag"] == "Saiu").sum())
st.caption(f"Reserva Stone detalhado — Entrou: {format_currency_br(entrou_signed)} (n={num_in}) • Saiu: {format_currency_br(saiu_signed)} (n={num_out}) • Líquido: {format_currency_br(stone_total)}")

# Investimento Empresa
investimento_df = df_cat.loc[df_cat["InvestimentoEmpresaFlag"] == "Investimento Empresa"]
if not investimento_df.empty:
    total_investimento = investimento_df["amount"].sum()
    num_investimento = len(investimento_df)
    st.caption(f"💰 Investimento Empresa (Fabrício Giordanelli): {format_currency_br(total_investimento)} (n={num_investimento}) • Excluído dos cálculos de entrada/vendas")

# Funcionários
funcionarios_df = df_cat.loc[df_cat["FuncionarioFlag"] == "Funcionário"]
if not funcionarios_df.empty:
    total_funcionarios = funcionarios_df["amount"].sum()
    num_funcionarios = len(funcionarios_df)
    st.caption(f"👥 Funcionários: {format_currency_br(total_funcionarios)} (n={num_funcionarios}) • Patrick, Maressa, Vamberto, Alessandro, Joaldo, Antonio Orlando")

# -------------------- Relatório mensal (Prévia) --------------------
# (apenas cálculo; a renderização virá depois do Resumo mensal)
monthly_preview = no_flag.copy()
monthly_preview["month"] = pd.to_datetime(monthly_preview["date"]).dt.to_period("M").dt.to_timestamp()

# Agrupado linear: mês x categoria, entradas e saídas
monthly_linear = (
    monthly_preview
    .groupby(["month", "Categoria"], dropna=False)["amount"]
    .agg(
        Entradas=lambda s: float(s[s > 0].sum()),
        Saidas=lambda s: float(s[s < 0].sum()),
    )
    .reset_index()
    .sort_values(["month", "Saidas"], ascending=[False, True])
)
# (renderização virá depois do Resumo mensal)

# -------------------- KPIs (REMOVIDO lugar antigo) --------------------
# (o bloco antigo de KPIs e KPIs por mês foi removido para seguir a nova ordem)

st.divider()

# -------------------- Totais por Categoria (despesas) --------------------
# (Removido a pedido: sem gráficos nem seção de total por categoria)

# -------------------- Resumo mensal: Entradas, Saídas, Saldo --------------------
st.subheader("Resumo mensal (Entradas, Saídas, Saldo)")
monthly_df = no_flag.copy()
monthly_df["month"] = pd.to_datetime(monthly_df["date"]).dt.to_period("M").dt.to_timestamp()
monthly_summary = (
    monthly_df
    .groupby("month", dropna=False)["amount"]
    .agg(
        Entradas=lambda s: float(s[s > 0].sum()),
        Saidas=lambda s: float(s[s < 0].sum()),
        Saldo=lambda s: float(s.sum()),
    )
    .reset_index()
)
# Reserva Stone por mês via flags (soma assinada)
flagged_month = df_cat.copy()
flagged_month["month"] = pd.to_datetime(flagged_month["date"]).dt.to_period("M").dt.to_timestamp()
stone_signed_m = (
    flagged_month.loc[flagged_month["ReservaStoneFlag"].isin(["Saiu", "Entrou"])]
                 .groupby("month")["ReservaStoneSigned"].sum()
                 .reset_index(name="Reserva Stone")
)
monthly_summary = monthly_summary.merge(stone_signed_m, on="month", how="left")
monthly_summary["Reserva Stone"] = monthly_summary["Reserva Stone"].fillna(0.0)
# Recalcula Saldo no formato absoluto solicitado: |entradas| - |saídas| - |reserva stone|
monthly_summary["Saldo"] = monthly_summary.apply(
    lambda r: abs(float(r["Entradas"])) - abs(float(r["Saidas"])) - abs(float(r["Reserva Stone"])), axis=1
)

# CMV mensal por mês (alinhado por merge) baseado em monthly_df
cat_norm_series = monthly_df["Categoria"].map(normalize_text)
cmv_mask_all = (
    cat_norm_series.str.contains("mercado", regex=False, na=False) |
    cat_norm_series.str.contains("parma", regex=False, na=False) |
    cat_norm_series.str.contains("bufala", regex=False, na=False) |
    cat_norm_series.str.contains("salame", regex=False, na=False) |
    cat_norm_series.str.contains("pepporoni", regex=False, na=False) |
    cat_norm_series.str.contains("parmesao", regex=False, na=False) |
    cat_norm_series.str.contains("farinha", regex=False, na=False) |
    cat_norm_series.str.contains("molho", regex=False, na=False) |
    cat_norm_series.str.contains("azeite", regex=False, na=False) |
    cat_norm_series.str.contains("gelo", regex=False, na=False)
)
cmv_by_month = (
    monthly_df.loc[cmv_mask_all]
              .groupby("month")["amount"].sum()
              .reset_index(name="CMV")
)
monthly_summary = monthly_summary.merge(cmv_by_month, on="month", how="left")
monthly_summary["CMV"] = monthly_summary["CMV"].fillna(0.0)

# Gastos com funcionários por mês
# Inclui tokens antigos + funcionários detectados pela flag automática
func_tokens = ["cesar", "raimundo", "cris", "joaldo", "marresa"]
func_mask_tokens = monthly_df["_dest_norm"].str.contains("|".join(func_tokens), regex=True, na=False)
func_mask_flag = monthly_df["FuncionarioFlag"] == "Funcionário"
func_mask = func_mask_tokens | func_mask_flag

func_spend_m = (
    monthly_df.loc[func_mask & (monthly_df["amount"] < 0)]
              .assign(_spend=lambda d: d["amount"].abs())
              .groupby("month")["_spend"].sum()
              .reset_index(name="__func_spend_raw")
)
monthly_summary = monthly_summary.merge(func_spend_m, on="month", how="left")
monthly_summary["__func_spend_raw"] = monthly_summary["__func_spend_raw"].fillna(0.0)

# Entradas IFOOD por mês
ifood_entries_m = (
    monthly_df.loc[monthly_df["IfoodFlag"] == "IFOOD"]
              .loc[lambda d: d["amount"] > 0]
              .groupby("month")["amount"].sum()
              .reset_index(name="Entradas IFOOD")
)
monthly_summary = monthly_summary.merge(ifood_entries_m, on="month", how="left")
monthly_summary["Entradas IFOOD"] = monthly_summary["Entradas IFOOD"].fillna(0.0)

# Vendas = Entradas - IFOOD
monthly_summary["Vendas"] = monthly_summary["Entradas"] - monthly_summary["Entradas IFOOD"]

# Gorjeta = 91% * 10% das Vendas
monthly_summary["Gorjeta"] = monthly_summary["Vendas"] * 0.91 * 0.10

# Entrada líquida = Vendas + IFOOD - Gorjeta
monthly_summary["Entrada líquida"] = monthly_summary["Vendas"] + monthly_summary["Entradas IFOOD"] - monthly_summary["Gorjeta"]

# %CMV = CMV / Entrada líquida (evita divisão por zero)
monthly_summary["%CMV"] = monthly_summary.apply(lambda r: (r["CMV"] / r["Entrada líquida"]) if r["Entrada líquida"] != 0 else 0.0, axis=1)

# Func = Total gasto com funcionários - Gorjeta
monthly_summary["Func"] = monthly_summary["__func_spend_raw"] - monthly_summary["Gorjeta"]

# %Func = Func / Entrada líquida (evita divisão por zero)
monthly_summary["%Func"] = monthly_summary.apply(lambda r: (r["Func"] / r["Entrada líquida"]) if r["Entrada líquida"] != 0 else 0.0, axis=1)

# Remover coluna temporária
monthly_summary = monthly_summary.drop(columns=["__func_spend_raw"])

# Reordenar colunas para que "Vendas", "Gorjeta", "Entradas IFOOD" e "Entrada líquida" fiquem ao lado de "Entradas"
monthly_summary = monthly_summary[["month", "Entradas", "Vendas", "Gorjeta", "Entradas IFOOD", "Entrada líquida", "Saidas", "Saldo", "Reserva Stone", "CMV", "%CMV", "Func", "%Func"]]

render_centered_table(
    format_currency_columns(monthly_summary, ["Entradas", "Vendas", "Gorjeta", "Entradas IFOOD", "Entrada líquida", "Saidas", "Saldo", "Reserva Stone", "CMV", "Func"])\
    .assign(**{"%CMV": lambda d: d["%CMV"].map(format_percent_br), "%Func": lambda d: d["%Func"].map(format_percent_br)})
)

# Após o resumo, renderizar a Prévia mensal por Categoria (linear)
st.subheader("Prévia mensal por Categoria (linear)")
render_monthly_linear_scroller(monthly_linear)

# -------------------- Vendas por Dia da Semana --------------------
st.divider()
st.subheader("📊 Vendas por Dia da Semana")

# Criar dataframe com vendas (Entradas excluindo IFOOD)
vendas_df = no_flag.copy()
vendas_df = vendas_df.loc[
    (vendas_df["amount"] > 0) & 
    (vendas_df["IfoodFlag"] != "IFOOD")
].copy()

# Adicionar dia da semana
vendas_df["dia_semana"] = pd.to_datetime(vendas_df["date"]).dt.day_name()

# Mapear para português
dias_map = {
    "Monday": "Segunda",
    "Tuesday": "Terça",
    "Wednesday": "Quarta",
    "Thursday": "Quinta",
    "Friday": "Sexta",
    "Saturday": "Sábado",
    "Sunday": "Domingo"
}
vendas_df["dia_semana"] = vendas_df["dia_semana"].map(dias_map)

# Contar quantos dias únicos de cada dia da semana existem no período (com ou sem vendas)
# Pegar todos os dias do período
all_dates_df = no_flag.copy()
all_dates_df["date_only"] = pd.to_datetime(all_dates_df["date"]).dt.date
unique_dates = pd.DataFrame({"date_only": all_dates_df["date_only"].unique()})
unique_dates["dia_semana"] = pd.to_datetime(unique_dates["date_only"]).dt.day_name().map(dias_map)
dias_unicos_count = unique_dates.groupby("dia_semana").size().reset_index(name="Qtd_Dias")

# Agrupar vendas por dia da semana e somar por data
vendas_df["date_only"] = pd.to_datetime(vendas_df["date"]).dt.date
vendas_por_data = vendas_df.groupby(["date_only", "dia_semana"])["amount"].sum().reset_index()

# Calcular estatísticas por dia da semana
vendas_por_dia = (
    vendas_por_data
    .groupby("dia_semana")["amount"]
    .agg(
        Total=lambda s: float(s.sum()),
        Media=lambda s: float(s.mean()),
        Desvio_Padrao=lambda s: float(s.std()),
        Dias_Com_Venda=lambda s: int(len(s))
    )
    .reset_index()
)

# Merge com contagem de dias únicos
vendas_por_dia = vendas_por_dia.merge(dias_unicos_count, on="dia_semana", how="left")

# Ordenar dias da semana corretamente
ordem_dias = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]
vendas_por_dia["dia_semana"] = pd.Categorical(vendas_por_dia["dia_semana"], categories=ordem_dias, ordered=True)
vendas_por_dia = vendas_por_dia.sort_values("dia_semana").reset_index(drop=True)
vendas_por_dia.columns = ["Dia da Semana", "Total de Vendas", "Média por Dia", "Desvio Padrão", "Dias com Venda", "Total de Dias"]

# Adicionar linha com média geral semanal e desvio padrão geral
# Agrupar por semana para calcular totais semanais
vendas_por_data_copy = vendas_por_data.copy()
vendas_por_data_copy["semana"] = pd.to_datetime(vendas_por_data_copy["date_only"]).dt.to_period("W").dt.to_timestamp()
vendas_por_semana = vendas_por_data_copy.groupby("semana")["amount"].sum().reset_index()

# Calcular média e desvio padrão dos totais semanais
media_semanal = vendas_por_semana["amount"].mean()
desvio_semanal = vendas_por_semana["amount"].std()
total_geral = vendas_por_data["amount"].sum()
dias_com_venda_geral = len(vendas_por_data)
total_dias_geral = len(unique_dates)

linha_media_geral = pd.DataFrame({
    "Dia da Semana": ["Média Geral Semanal"],
    "Total de Vendas": [total_geral],
    "Média por Dia": [media_semanal],
    "Desvio Padrão": [desvio_semanal],
    "Dias com Venda": [dias_com_venda_geral],
    "Total de Dias": [total_dias_geral]
})

vendas_por_dia = pd.concat([vendas_por_dia, linha_media_geral], ignore_index=True)

# Renderizar tabela
render_centered_table(format_currency_columns(vendas_por_dia, ["Total de Vendas", "Média por Dia", "Desvio Padrão"]))

# Gráfico de faturamento por semana
st.markdown("### 📈 Faturamento por Semana")

# Preparar dados para o gráfico
vendas_por_semana_sorted = vendas_por_semana.sort_values("semana").reset_index(drop=True)
vendas_por_semana_sorted["numero_semana"] = range(1, len(vendas_por_semana_sorted) + 1)
vendas_por_semana_sorted["semana_label"] = "Semana " + vendas_por_semana_sorted["numero_semana"].astype(str)

# Criar gráfico de linha com Altair
chart = alt.Chart(vendas_por_semana_sorted).mark_line(point=True, color="#1f77b4", strokeWidth=3).encode(
    x=alt.X("numero_semana:Q", title="Semana", axis=alt.Axis(format="d", tickMinStep=1)),
    y=alt.Y("amount:Q", title="Faturamento (R$)", axis=alt.Axis(format=",.2f")),
    tooltip=[
        alt.Tooltip("semana_label:N", title="Semana"),
        alt.Tooltip("amount:Q", title="Faturamento", format=",.2f")
    ]
).properties(
    width=800,
    height=400
).configure_point(
    size=100
)

st.altair_chart(chart, use_container_width=True)

# -------------------- Estornos Detectados --------------------
st.divider()
st.subheader("🔄 Estornos Detectados")
estornos_df = df_cat.loc[df_cat["EstornoFlag"].isin(["Original", "Estorno"])].copy()
if not estornos_df.empty:
    estornos_df = estornos_df.sort_values(["EstornoParId", "date"])
    estornos_display = estornos_df[["date", "Destino", "amount", "EstornoFlag", "EstornoParId", "Categoria"]].copy()
    estornos_display.columns = ["Data", "Destino", "Valor", "Tipo", "Par ID", "Categoria"]
    
    st.caption(f"Total de transações em pares de estorno: {len(estornos_df)} ({len(estornos_df[estornos_df['EstornoFlag'] == 'Estorno'])} estornos detectados)")
    render_centered_table(format_currency_columns(estornos_display, ["Valor"]))
else:
    st.info("Nenhum estorno detectado no período.")

# -------------------- Top Fornecedores (removido a pedido) --------------------
# (Seção excluída)

st.divider()
# -------------------- Downloads --------------------
# Excel
excel_bytes = make_excel_report(df_cat)
st.download_button(
    "⬇️ Baixar relatório completo (Excel)",
    data=excel_bytes,
    file_name="relatorio_gastos_depara.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

# CSV simples com lançamentos categorizados (sem matched_keyword)
csv_buffer = io.StringIO()
(df_cat[["date", "Destino", "amount", "Categoria"]]
 .to_csv(csv_buffer, index=False, encoding="utf-8"))
st.download_button(
    "⬇️ Baixar lançamentos categorizados (CSV)",
    data=csv_buffer.getvalue().encode("utf-8"),
    file_name="lancamentos_categorizados.csv",
    mime="text/csv",
)

st.success("Pronto! De→Para aplicado sobre a coluna **Destino**. Ajuste opções na barra lateral se algo não bater (separador, encoding, formato de data).")
st.caption("Dica: no CSV de De→Para, a coluna **de** pode ter vários termos separados por vírgula/;//| (ex.: 'AMERICANAS, COMPANHIA BRASILEIRA DE DISTRIBUICAO').")
