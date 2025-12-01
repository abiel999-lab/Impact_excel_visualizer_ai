import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import json
from fpdf import FPDF
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import torch  # <--- TAMBAHAN

# =========================
# 0. OPSIONAL: AI (Phi-3 Mini)
# =========================
AI_ENABLED = False  # kalau mau matikan AI, ubah ke False
AI_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"

if AI_ENABLED:
    try:
        from transformers import pipeline
    except ImportError:
        AI_ENABLED = False


@st.cache_resource(show_spinner=False)
def load_local_llm():
    """
    Load model Phi-3 sekali (cache).
    - Coba dulu pakai GPU (device_map="auto", float16)
    - Kalau gagal, fallback ke CPU (device=-1, float32)
    """
    if not AI_ENABLED:
        return None

    try:
        # Mode cepat: pakai GPU kalau tersedia
        gen = pipeline(
            "text-generation",
            model=AI_MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    except Exception:
        # Fallback aman: CPU biasa
        gen = pipeline(
            "text-generation",
            model=AI_MODEL_ID,
            torch_dtype=torch.float32,
            device=-1,
        )
    return gen


def ai_generate_insight(df: pd.DataFrame, template_name: str) -> str:
    """
    Generate insight dengan Phi-3 Mini.
    """
    if not AI_ENABLED:
        return "AI belum diaktifkan atau modul transformers belum terinstall."

    pipe = load_local_llm()
    if pipe is None:
        return "Model AI belum berhasil di-load."

    # ---- Ringkasan data untuk prompt ----
    sample_rows = min(len(df), 50)
    sample_df = df.head(sample_rows)
    # GANTI: to_markdown() -> to_string() supaya tidak butuh 'tabulate'
    sample_table = sample_df.to_string(index=False)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

    prompt = f"""
Kamu adalah analis data yang sangat ahli.

Tugasmu:
1. Lihat informasi dataset di bawah.
2. Berikan insight utama dalam bahasa Indonesia yang jelas.
3. Jelaskan tren penting, anomali, dan insight bisnis yang menarik.
4. Berikan juga saran tindakan (actionable insight).
5. Gaya bahasa: singkat, rapi, pakai poin-poin dan paragraf pendek.
6. Jenis dashboard yang dipilih user: "{template_name}".

Info dataset:
- Jumlah baris: {len(df)}
- Jumlah kolom: {len(df.columns)}
- Kolom numerik: {numeric_cols}
- Kolom non-numerik: {non_numeric_cols}

Contoh {sample_rows} baris pertama (tabel teks):

{sample_table}

Tuliskan analisis dan insightmu di bawah ini:
""".strip()

    out = pipe(
        prompt,
        max_new_tokens=512,    # biar cepet
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=pipe.tokenizer.eos_token_id,
    )

    text = out[0]["generated_text"]
    # Banyak model mengembalikan prompt + jawaban → potong prefix prompt
    if text.startswith(prompt):
        text = text[len(prompt):]

    return text.strip()




# =========================
# 1. CONFIG APP
# =========================
st.set_page_config(
    page_title="Impact Excel Visualizer",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Impact Excel Visualizer")
st.caption(
    "Excel & Data Visualizer + AI Insight - powered by Streamlit, Pandas, Plotly & gpt-oss-7b"
)

# =========================
# 2. FUNGSI LOAD DATA
# =========================
def load_excel(file, sheet_name=None):
    return pd.read_excel(file, sheet_name=sheet_name)


def load_csv_like(file, sep=","):
    return pd.read_csv(file, sep=sep)


def load_json(file):
    data_bytes = file.read()
    try:
        data_str = data_bytes.decode("utf-8")
    except UnicodeDecodeError:
        data_str = data_bytes.decode("latin-1")
    obj = json.loads(data_str)
    return pd.DataFrame(obj)


def load_data(uploaded_file, file_type, **kwargs):
    if file_type in ["xlsx", "xls"]:
        return load_excel(uploaded_file, sheet_name=kwargs.get("sheet_name"))
    elif file_type in ["csv", "tsv", "txt"]:
        sep = kwargs.get("sep", ",")
        return load_csv_like(uploaded_file, sep=sep)
    elif file_type == "json":
        uploaded_file.seek(0)
        return load_json(uploaded_file)
    else:
        raise ValueError("Tipe file belum didukung.")


# =========================
# 3. SIDEBAR - UPLOAD & OPTIONS
# =========================
st.sidebar.header("⚙️ Pengaturan Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload file (Excel / CSV / TSV / TXT / JSON)",
    type=["xlsx", "xls", "csv", "tsv", "txt", "json"],
)

sheet_name = None
sep = ","
df = None
file_type = None

template_dashboard = st.sidebar.selectbox(
    "Template Dashboard",
    [
        "📌 Overview Umum",
        "📈 Time Series (kalau ada tanggal)",
        "📦 Analisis Distribusi",
    ],
)

if uploaded_file is not None:
    file_type = Path(uploaded_file.name).suffix.lower().replace(".", "")

    if file_type in ["xlsx", "xls"]:
        try:
            xls = pd.ExcelFile(uploaded_file)
            sheet_options = xls.sheet_names
            sheet_name = st.sidebar.selectbox("Pilih sheet", sheet_options)
            uploaded_file.seek(0)
        except Exception as e:
            st.sidebar.error(f"Gagal membaca sheet: {e}")

    if file_type in ["csv", "tsv", "txt"]:
        default_sep = "," if file_type == "csv" else "\t"
        sep = st.sidebar.selectbox(
            "Delimiter",
            options=[default_sep, ",", ";", "|", "\t"],
            index=0,
            format_func=lambda x: "\\t (tab)" if x == "\t" else x,
        )

    try:
        df = load_data(uploaded_file, file_type, sheet_name=sheet_name, sep=sep)
    except Exception as e:
        st.error(f"❌ Gagal load data: {e}")
        df = None


# =========================
# 4. UTIL
# =========================
def auto_parse_datetime(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().sum() > 0 and parsed.notna().sum() >= len(df) * 0.3:
                    df[col] = parsed
            except Exception:
                pass
    return df


def build_basic_summary(df: pd.DataFrame) -> dict:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    summary = {
        "rows": len(df),
        "cols": len(df.columns),
        "missing_total": int(df.isna().sum().sum()),
        "numeric_cols": numeric_cols,
        "non_numeric_cols": [c for c in df.columns if c not in numeric_cols],
    }
    return summary


def pdf_safe(text: str) -> str:
    """
    Bersihkan teks agar compatible dengan FPDF (latin-1 only).
    """
    if not isinstance(text, str):
        text = str(text)
    return text.encode("latin-1", "replace").decode("latin-1")


def pdf_write_paragraph(pdf: FPDF, text: str, line_height: float = 6, max_len: int = 100):
    """
    Tulis paragraf ke PDF tanpa multi_cell.
    Pecah per '\n', lalu potong tiap baris per max_len karakter.
    """
    for raw_line in (text or "").split("\n"):
        line = pdf_safe(raw_line)
        if not line.strip():
            line = " "
        for i in range(0, len(line), max_len):
            chunk = line[i : i + max_len]
            pdf.cell(0, line_height, chunk, ln=True)


def fig_to_pdf_image(pdf: FPDF, fig, x: float = 10, w: float = 190):
    """
    Simpan matplotlib figure ke buffer PNG lalu gambar ke PDF.
    """
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    pdf.image(buf, x=x, w=w)


def pdf_table_simple(
    pdf: FPDF,
    headers,
    rows,
    col_widths=None,
    header_font=("Arial", "B", 9),
    row_font=("Arial", "", 8),
    line_height: float = 5,
    max_chars: int = 20,
):
    """
    Gambar tabel sederhana (tanpa multi_cell).
    - header: list judul kolom
    - rows: list of list/tuple
    - col_widths: list lebar kolom (optional). Kalau None → bagi rata.
    """
    usable_width = pdf.w - pdf.l_margin - pdf.r_margin

    n_cols = len(headers)
    if col_widths is None:
        col_widths = [usable_width / n_cols] * n_cols
    else:
        # Scale kalau total > usable_width
        total_w = sum(col_widths)
        if total_w > usable_width:
            factor = usable_width / total_w
            col_widths = [w * factor for w in col_widths]

    # Header
    pdf.set_font(*header_font)
    for h, w in zip(headers, col_widths):
        pdf.cell(w, line_height, pdf_safe(str(h))[:max_chars], border=1, align="C")
    pdf.ln(line_height)

    # Rows
    pdf.set_font(*row_font)
    for row in rows:
        for val, w in zip(row, col_widths):
            txt = pdf_safe(str(val))[:max_chars]
            pdf.cell(w, line_height, txt, border=1)
        pdf.ln(line_height)


def create_pdf_report(df: pd.DataFrame, summary: dict, ai_text: str) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ========= HEADER =========
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, pdf_safe("Laporan Analisis Data"), ln=True)

    pdf.set_font("Arial", "", 12)
    pdf.ln(4)
    pdf_write_paragraph(pdf, "Aplikasi: impact_excel_visualizer", line_height=8)
    pdf.ln(2)

    # ========= 1. Ringkasan Dataset =========
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, pdf_safe("1. Ringkasan Dataset"), ln=True)

    pdf.set_font("Arial", "", 11)
    ringkasan = (
        f"- Jumlah baris: {summary['rows']}\n"
        f"- Jumlah kolom: {summary['cols']}\n"
        f"- Total missing value: {summary['missing_total']}\n"
        f"- Kolom numerik: {', '.join(summary['numeric_cols']) if summary['numeric_cols'] else '-'}\n"
        f"- Kolom non-numerik: {', '.join(summary['non_numeric_cols']) if summary['non_numeric_cols'] else '-'}"
    )
    pdf_write_paragraph(pdf, ringkasan, line_height=6)
    pdf.ln(2)

    # ========= 2. Statistik Numerik =========
    numeric_cols = summary["numeric_cols"]
    if numeric_cols:
        pdf.set_font("Arial", "B", 13)
        pdf.cell(0, 8, pdf_safe("2. Statistik Numerik (ringkas)"), ln=True)
        pdf.set_font("Arial", "", 10)

        desc = df[numeric_cols].describe().T.reset_index()
        desc.rename(columns={"index": "kolom"}, inplace=True)

        for _, row in desc.iterrows():
            line = (
                f"- {row['kolom']}: mean={row['mean']:.3f}, "
                f"min={row['min']:.3f}, max={row['max']:.3f}, std={row['std']:.3f}"
            )
            pdf_write_paragraph(pdf, line, line_height=5)
        pdf.ln(2)

    # ========= 3. Struktur Kolom & Missing Value (TABLE) =========
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, pdf_safe("3. Struktur Kolom & Missing Value"), ln=True)
    pdf.ln(1)

    dtypes_df = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": df.dtypes.astype(str),
            "missing": df.isna().sum().values,
        }
    )

    pdf.set_font("Arial", "", 9)
    pdf_write_paragraph(pdf, "Struktur kolom (maks 30 baris):", line_height=5)

    # Tabel struktur kolom
    headers = ["column", "dtype", "missing"]
    rows = dtypes_df.head(30).values.tolist()
    pdf_table_simple(
        pdf,
        headers,
        rows,
        col_widths=[80, 60, 20],
        line_height=5,
        max_chars=25,
    )
    pdf.ln(1)

    # Ringkasan missing value
    missing_series = df.isna().sum()
    missing_df = missing_series[missing_series > 0].sort_values(ascending=False)
    if not missing_df.empty:
        pdf_write_paragraph(pdf, "Ringkasan missing value:", line_height=5)
        miss_rows = [[idx, val] for idx, val in missing_df.items()]
        pdf_table_simple(
            pdf,
            ["column", "missing"],
            miss_rows,
            col_widths=[100, 30],
            line_height=5,
            max_chars=25,
        )
    else:
        pdf_write_paragraph(pdf, "Tidak ada missing value.", line_height=5)

    pdf.ln(2)

    # ========= 4. Visualisasi Utama =========
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, pdf_safe("4. Visualisasi Utama"), ln=True)
    pdf.ln(2)

    # 4.1 Histogram kolom numerik pertama
    if numeric_cols:
        col = numeric_cols[0]
        fig, ax = plt.subplots(figsize=(5, 2.7))
        ax.hist(df[col].dropna(), bins=15, color="steelblue", edgecolor="black")
        ax.set_title(f"Distribusi {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        fig_to_pdf_image(pdf, fig)
        pdf.ln(2)

    # 4.2 Time Series (line) + total per bulan
    datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
    if datetime_cols and numeric_cols:
        tcol = datetime_cols[0]
        vcol = numeric_cols[0]
        df_ts = df.dropna(subset=[tcol, vcol]).sort_values(tcol)

        # line chart
        fig, ax = plt.subplots(figsize=(5, 2.7))
        ax.plot(df_ts[tcol], df_ts[vcol], marker="o", linestyle="-")
        ax.set_title(f"Time Series {vcol} vs {tcol}")
        ax.set_xlabel(str(tcol))
        ax.set_ylabel(str(vcol))
        fig.autofmt_xdate()
        fig_to_pdf_image(pdf, fig)
        pdf.ln(2)

        # total per bulan
        df_ts["__month__"] = df_ts[tcol].dt.to_period("M").dt.to_timestamp()
        df_month = df_ts.groupby("__month__")[vcol].sum().reset_index()

        fig, ax = plt.subplots(figsize=(5, 2.7))
        ax.bar(df_month["__month__"].dt.strftime("%b %Y"), df_month[vcol])
        ax.set_title(f"Total {vcol} per Bulan")
        ax.set_xlabel("Bulan")
        ax.set_ylabel(str(vcol))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        fig_to_pdf_image(pdf, fig)
        pdf.ln(2)

    # 4.3 Boxplot beberapa kolom numerik (analisis distribusi)
    if numeric_cols:
        for col in numeric_cols[:3]:
            fig, ax = plt.subplots(figsize=(3.5, 2.7))
            ax.boxplot(df[col].dropna(), vert=True)
            ax.set_title(f"Boxplot {col}")
            fig_to_pdf_image(pdf, fig, x=40, w=130)
            pdf.ln(2)

    # 4.4 Heatmap korelasi
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr().values
        labels = numeric_cols

        fig, ax = plt.subplots(figsize=(5, 3))
        im = ax.imshow(corr, cmap="Blues", vmin=-1, vmax=1)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_title("Correlation Matrix")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig_to_pdf_image(pdf, fig)
        pdf.ln(2)

    # ========= 5. Sample Data Mentah (TABLE) =========
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, pdf_safe("5. Sample Data Mentah (10 baris pertama)"), ln=True)
    pdf.ln(1)

    sample_df = df.head(10)

    # Biar tabel tidak super kecil, batasi maksimal 10 kolom pertama
    max_cols = 10
    display_cols = sample_df.columns[:max_cols]
    sample_subset = sample_df[display_cols]

    headers = list(display_cols)
    rows = sample_subset.values.tolist()

    pdf_table_simple(
        pdf,
        headers,
        rows,
        col_widths=None,  # auto rata
        line_height=5,
        max_chars=18,
    )
    pdf.ln(2)

    # ========= 6. Insight AI =========
    if ai_text:
        pdf.set_font("Arial", "B", 13)
        pdf.cell(0, 8, pdf_safe("6. Insight & Narasi (AI)"), ln=True)
        pdf.set_font("Arial", "", 11)
        pdf_write_paragraph(pdf, ai_text, line_height=6)
        pdf.ln(2)

    # output ke bytes
    result = pdf.output(dest="S")
    if isinstance(result, (bytes, bytearray)):
        pdf_bytes = bytes(result)
    else:
        pdf_bytes = str(result).encode("latin1", "replace")
    return pdf_bytes


# =========================
# 5. MAIN UI
# =========================
if df is not None:
    df = auto_parse_datetime(df)
    summary = build_basic_summary(df)

    # 5.1 Ringkasan Data
    st.subheader("1️⃣ Ringkasan Data")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Jumlah Baris", summary["rows"])
    with c2:
        st.metric("Jumlah Kolom", summary["cols"])
    with c3:
        st.metric("Total Missing", summary["missing_total"])

    dtypes_df = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": df.dtypes.astype(str),
            "missing_count": df.isna().sum().values,
        }
    )
    st.markdown("**Info Kolom:**")
    st.dataframe(dtypes_df, use_container_width=True)

    st.markdown("---")

    # 5.2 Statistik & Summary
    st.subheader("2️⃣ Statistik & Summary")

    tab1, tab2, tab3 = st.tabs(["📈 Statistik Numerik", "🕳 Missing Value", "🔗 Korelasi"])

    numeric_cols = summary["numeric_cols"]

    with tab1:
        if numeric_cols:
            st.markdown("**Deskripsi Statistik (kolom numerik)**")
            st.dataframe(df[numeric_cols].describe().T, use_container_width=True)
        else:
            st.info("Tidak ada kolom numerik.")

    with tab2:
        missing_series = df.isna().sum()
        missing_df = missing_series[missing_series > 0].sort_values(ascending=False)
        if not missing_df.empty:
            st.markdown("**Missing value per kolom:**")
            st.dataframe(
                missing_df.rename("missing_count").to_frame(),
                use_container_width=True,
            )
        else:
            st.success("Tidak ada missing value 🎉")

    with tab3:
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()
            st.markdown("**Matriks Korelasi:**")
            fig_corr = px.imshow(
                corr,
                text_auto=True,
                aspect="auto",
                title="Correlation Matrix",
            )
            st.plotly_chart(fig_corr, use_container_width=True, key="corr_matrix")
        else:
            st.info("Perlu ≥ 2 kolom numerik untuk korelasi.")

    st.markdown("---")

    # 5.3 Visualisasi Interaktif (Custom)
    st.subheader("3️⃣ Visualisasi Interaktif (Custom)")

    chart_type = st.selectbox(
        "Tipe chart",
        ["Line", "Bar", "Scatter", "Histogram", "Box", "Pie"],
        key="chart_type_manual",
    )

    all_cols = df.columns.tolist()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    fig_custom = None

    if chart_type in ["Line", "Bar", "Scatter", "Box"]:
        x_col = st.selectbox("Kolom X", options=all_cols, key="x_manual")
        y_candidates = numeric_cols if numeric_cols else all_cols
        y_col = st.selectbox("Kolom Y", options=y_candidates, key="y_manual")
        color_col = st.selectbox(
            "Kolom Color (opsional)",
            options=[None] + all_cols,
            key="color_manual",
        )

        if chart_type == "Line":
            fig_custom = px.line(df, x=x_col, y=y_col, color=color_col)
        elif chart_type == "Bar":
            fig_custom = px.bar(df, x=x_col, y=y_col, color=color_col)
        elif chart_type == "Scatter":
            fig_custom = px.scatter(df, x=x_col, y=y_col, color=color_col)
        elif chart_type == "Box":
            fig_custom = px.box(df, x=x_col, y=y_col, color=color_col)

    elif chart_type == "Histogram":
        hist_cols = numeric_cols if numeric_cols else all_cols
        x_col = st.selectbox("Kolom untuk Histogram", options=hist_cols, key="x_hist")
        color_col = st.selectbox(
            "Kolom Color (opsional)",
            options=[None] + all_cols,
            key="color_hist",
        )
        fig_custom = px.histogram(df, x=x_col, color=color_col)

    elif chart_type == "Pie":
        names_col = st.selectbox("Kolom kategori (names)", options=all_cols, key="names_pie")
        value_cols = numeric_cols if numeric_cols else all_cols
        values_col = st.selectbox("Kolom nilai (values)", options=value_cols, key="values_pie")
        fig_custom = px.pie(df, names=names_col, values=values_col)

    if fig_custom is not None:
        st.plotly_chart(fig_custom, use_container_width=True, key="custom_chart")

    st.markdown("---")

    # 5.4 TEMPLATE DASHBOARD
    st.subheader("4️⃣ Template Dashboard")

    datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if template_dashboard == "📌 Overview Umum":
        st.markdown("**Template: Overview Umum**")

        if numeric_cols:
            fig1 = px.histogram(df, x=numeric_cols[0], title=f"Distribusi {numeric_cols[0]}")
            st.plotly_chart(fig1, use_container_width=True, key="overview_hist")

        if numeric_cols and cat_cols:
            fig2 = px.bar(
                df.groupby(cat_cols[0])[numeric_cols[0]].mean().reset_index(),
                x=cat_cols[0],
                y=numeric_cols[0],
                title=f"Rata-rata {numeric_cols[0]} per {cat_cols[0]}",
            )
            st.plotly_chart(fig2, use_container_width=True, key="overview_bar")

        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()
            fig3 = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Matrix (Overview)")
            st.plotly_chart(fig3, use_container_width=True, key="overview_corr")

    elif template_dashboard == "📈 Time Series (kalau ada tanggal)":
        st.markdown("**Template: Time Series**")

        if datetime_cols and numeric_cols:
            time_col = datetime_cols[0]
            val_col = numeric_cols[0]

            df_ts = df.sort_values(time_col)
            fig_ts = px.line(df_ts, x=time_col, y=val_col, title=f"Time Series {val_col} vs {time_col}")
            st.plotly_chart(fig_ts, use_container_width=True, key="ts_line")

            df_ts["__month__"] = df_ts[time_col].dt.to_period("M").dt.to_timestamp()
            df_month = df_ts.groupby("__month__")[val_col].sum().reset_index()
            fig_month = px.bar(df_month, x="__month__", y=val_col, title=f"Total {val_col} per Bulan")
            st.plotly_chart(fig_month, use_container_width=True, key="ts_month")
        else:
            st.info("Tidak ditemukan kolom tanggal + numerik yang cocok untuk time series.")

    elif template_dashboard == "📦 Analisis Distribusi":
        st.markdown("**Template: Analisis Distribusi**")
        if numeric_cols:
            for i, col_name in enumerate(numeric_cols[:3]):
                fig = px.box(df, y=col_name, title=f"Boxplot {col_name}")
                st.plotly_chart(fig, use_container_width=True, key=f"dist_box_{i}")
        else:
            st.info("Tidak ada kolom numerik untuk distribusi.")

    st.markdown("---")

    # 5.5 AI Insight
    st.subheader("5️⃣ AI Insight (Phi-3 Mini)")

    ai_text = ""

    if AI_ENABLED:
        with st.expander("Klik untuk menghasilkan insight AI"):
            st.markdown("Model: **microsoft/Phi-3-mini-4k-instruct** (jalan lokal via HuggingFace Transformers).")
            st.markdown("⚠️ Pertama kali jalan bisa lama karena harus download model & load ke RAM/VRAM.")
            if st.button("Generate Insight dengan AI"):
                with st.spinner("AI sedang menganalisis data..."):
                    ai_text = ai_generate_insight(df, template_dashboard)
                st.markdown("---")
                st.markdown("### Hasil Insight AI")
                st.write(ai_text)
    else:
        st.info("Fitur AI belum aktif. Pastikan `transformers` terinstall dan `AI_ENABLED = True` di app.py.")

    st.markdown("---")

    # 5.6 Export Laporan (PDF)
    st.subheader("6️⃣ Export Laporan")

    st.markdown(
        "Laporan berisi ringkasan dataset, statistik numerik, struktur kolom, visualisasi utama, sample data, dan (opsional) insight AI."
    )

    if "last_ai_text" not in st.session_state:
        st.session_state["last_ai_text"] = ""

    if ai_text:
        st.session_state["last_ai_text"] = ai_text

    use_ai_in_report = st.checkbox(
        "Sertakan insight AI (kalau tersedia)",
        value=True,
    )

    if st.button("📝 Generate & Download PDF"):
        text_for_report = st.session_state["last_ai_text"] if use_ai_in_report else ""
        pdf_bytes = create_pdf_report(df, summary, text_for_report)
        st.success("Laporan berhasil dibuat. Silakan download di bawah:")

        st.download_button(
            label="⬇️ Download Laporan PDF",
            data=pdf_bytes,
            file_name="laporan_impact_excel_visualizer.pdf",
            mime="application/pdf",
        )

    st.markdown("---")

    # 5.7 Data Mentah
    st.subheader("7️⃣ Data Mentah")
    with st.expander("Tampilkan tabel data"):
        st.dataframe(df, use_container_width=True)

else:
    st.info("Silakan upload file di sidebar untuk mulai analisis.")


