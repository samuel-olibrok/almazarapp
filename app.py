# app.py
import streamlit as st
import pandas as pd
import sqlite3
from datetime import date, datetime
import numpy as np
from collections import Counter

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Condiciones Óptimas de Molturación", page_icon="🫒", layout="wide")
DB_FILE = "moliendas.db"

# =========================
# DB FUNCTIONS
# =========================
@st.cache_resource
def get_connection():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS moliendas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fecha TEXT,
            variedad TEXT,
            contenido_graso REAL,
            contenido_graso_seco REAL,
            humedad REAL,
            maquinaria TEXT,
            criba REAL,
            temperatura_masa REAL,
            talco REAL,
            bomba_masa REAL,
            agua_martillo TEXT,
            agua_batidora REAL,
            agua_bomba REAL,
            chapa TEXT,
            graso_orujo REAL,
            observaciones TEXT,
            operario TEXT
        )
    """)
    return conn

conn = get_connection()

def insertar_molienda(data_tuple):
    with conn:
        conn.execute("""
            INSERT INTO moliendas (
                fecha, variedad, contenido_graso, contenido_graso_seco, humedad,
                maquinaria, criba, temperatura_masa, talco, bomba_masa,
                agua_martillo, agua_batidora, agua_bomba, chapa,
                graso_orujo, observaciones, operario
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, data_tuple)

def leer_df():
    df = pd.read_sql_query("SELECT * FROM moliendas ORDER BY date(fecha) DESC, id DESC", conn)
    return df

def eliminar_registro(registro_id: int):
    with conn:
        conn.execute("DELETE FROM moliendas WHERE id = ?", (registro_id,))

# =========================
# UTILS
# =========================
def grasa_sobre_seco(graso, humedad):
    try:
        graso = float(graso)
        humedad = float(humedad)
        if humedad >= 100:
            return np.nan
        return graso / (1 - humedad / 100)
    except Exception:
        return np.nan

def unique_sorted(series):
    vals = sorted({v for v in series.dropna().astype(str) if v.strip()})
    return vals

def moda_o_vacio(vals):
    c = Counter(vals)
    if not c:
        return None
    moda, freq = c.most_common(1)[0]
    return moda

# =========================
# PREDICTIVA
# =========================
def knn_recommendation(df, variedad, graso, humedad, maquinaria=None, k=5):
    base = df.copy()

    if variedad:
        subset = base[base["variedad"].fillna("").str.contains(variedad, case=False, na=False)]
        base = subset if not subset.empty else base
    if maquinaria:
        same_mach = base[base["maquinaria"].fillna("").str.contains(maquinaria, case=False, na=False)]
        if not same_mach.empty:
            base = same_mach

    base = base[pd.notna(base["contenido_graso"]) & pd.notna(base["humedad"])]
    if base.empty:
        return None, None

    base = base.copy()
    base["contenido_graso_seco"] = base.apply(
        lambda r: r["contenido_graso_seco"]
        if pd.notna(r["contenido_graso_seco"])
        else grasa_sobre_seco(r["contenido_graso"], r["humedad"]),
        axis=1
    )

    X = np.c_[
        pd.to_numeric(base["contenido_graso"], errors="coerce").astype(float),
        pd.to_numeric(base["humedad"], errors="coerce").astype(float),
        pd.to_numeric(base["contenido_graso_seco"], errors="coerce").astype(float)
    ]

    gss_input = grasa_sobre_seco(graso, humedad)
    x0 = np.array([float(graso), float(humedad), float(gss_input)], dtype=float)

    Xmin = np.nanmin(X, axis=0)
    Xrng = np.nanmax(X, axis=0) - Xmin
    Xrng[Xrng == 0] = 1.0
    Xn = (X - Xmin) / Xrng
    x0n = (x0 - Xmin) / Xrng

    def dist_row(v):
        m = ~np.isnan(v) & ~np.isnan(x0n)
        if not m.any():
            return np.inf
        return np.sqrt(np.sum((v[m] - x0n[m]) ** 2))

    dists = np.apply_along_axis(dist_row, 1, Xn)
    base = base.assign(_dist=dists).sort_values("_dist").head(int(k))

    rec = {}
    # --- variables de media
    rec["temperatura_masa"] = base["temperatura_masa"].mean(skipna=True)

    # --- variables de horquilla
    for col in ["talco", "bomba_masa", "agua_batidora", "agua_bomba"]:
        vals = pd.to_numeric(base[col], errors="coerce").dropna()
        if len(vals) > 0:
            min_v, max_v = vals.min(), vals.max()
            moda = moda_o_vacio(vals)
            rec[col] = f"{min_v:.1f} – {max_v:.1f}"
            if moda is not None and (vals == moda).sum() >= 2:
                rec[col] += f" (más repetido: {moda:.1f})"

    # --- variables categóricas
    for col in ["criba", "chapa", "agua_martillo"]:
        vals = [str(v).strip() for v in base[col].dropna() if str(v).strip()]
        if vals:
            moda = moda_o_vacio(vals)
            rec[col] = f"{moda} (mayoritaria)"

    info = {"n_vecinos": len(base), "gss_input": gss_input}
    return rec, info

# =========================
# APP
# =========================
st.title("🫒 Condiciones Óptimas de Molturación")

df_all = leer_df()
df_all["fecha"] = pd.to_datetime(df_all["fecha"], errors="coerce").dt.date

# ---------- PREDICTIVA ----------
st.subheader("🔎 Consulta predictiva (antes de moler)")
with st.form("predictiva"):
    c1, c2, c3, c4, c5 = st.columns([1.2,1,1,1.2,0.8])
    variedades = unique_sorted(df_all["variedad"]) if not df_all.empty else []
    maqs = unique_sorted(df_all["maquinaria"]) if not df_all.empty else []

    with c1:
        var_sel = st.selectbox("Variedad (historial)", ["(ninguna)"] + variedades)
        variedad_in = st.text_input("Variedad", value=("" if var_sel == "(ninguna)" else var_sel))
    with c2:
        graso_in = st.number_input("Contenido graso (%)", min_value=0.0, step=0.1)
    with c3:
        humedad_in = st.number_input("Humedad (%)", min_value=0.0, step=0.1)
    with c4:
        maq_sel = st.selectbox("Maquinaria (historial)", ["(ninguna)"] + maqs)
        maquinaria_in = st.text_input("Maquinaria", value=("" if maq_sel == "(ninguna)" else maq_sel))
    with c5:
        k = st.number_input("K vecinos", min_value=1, max_value=20, value=5)

    submit_pred = st.form_submit_button("Consultar")

if submit_pred:
    if df_all.empty:
        st.warning("No hay moliendas registradas aún.")
    else:
        rec, meta = knn_recommendation(df_all, variedad_in, graso_in, humedad_in, maquinaria_in, k)
        if rec is None:
            st.warning("No se encontraron moliendas comparables.")
        else:
            st.success(f"Basado en {meta['n_vecinos']} molienda(s) similar(es).")
            st.caption(f"Contenido graso sobre materia seca calculado: **{meta['gss_input']:.2f}%**")

            order = ["criba", "temperatura_masa", "talco", "bomba_masa", "agua_martillo", "agua_batidora", "agua_bomba", "chapa"]
            for k in order:
                if k in rec:
                    st.markdown(f"**{k.replace('_',' ').capitalize()}:** {rec[k]}")

# ---------- NUEVA MOLIENDA ----------
st.subheader("➕ Añadir nueva molienda exitosa")
with st.form("nueva_molienda"):
    c1, c2, c3 = st.columns(3)
    variedades = unique_sorted(df_all["variedad"]) if not df_all.empty else []
    maqs = unique_sorted(df_all["maquinaria"]) if not df_all.empty else []

    with c1:
        fecha = st.date_input("Fecha (DD/MM/AAAA)", value=date.today(), min_value=date(2000, 1, 1), max_value=date.today(), format="DD/MM/YYYY")
        var_sel2 = st.selectbox("Variedad (historial)", ["(ninguna)"] + variedades)
        variedad = st.text_input("Variedad", value=("" if var_sel2 == "(ninguna)" else var_sel2))
        contenido_graso = st.number_input("Contenido graso (%)", min_value=0.0, step=0.1)
    with c2:
        humedad = st.number_input("Humedad (%)", min_value=0.0, step=0.1)
        cgms_calc = grasa_sobre_seco(contenido_graso, humedad)
        st.info(f"💡 CGMS calculado: **{cgms_calc:.2f}%**")
        maq_sel2 = st.selectbox("Maquinaria (historial)", ["(ninguna)"] + maqs)
        maquinaria = st.text_input("Maquinaria", value=("" if maq_sel2 == "(ninguna)" else maq_sel2))
    with c3:
        criba = st.number_input("Criba (mm)", min_value=0.0, step=0.1)
        temperatura_masa = st.number_input("Temperatura masa (°C)", min_value=0.0, step=0.5)
        talco = st.number_input("Talco (%)", min_value=0.0, max_value=3.0, step=0.1)

    # Aguas en orden: Martillo → Batidora → Bomba masa
    agua_martillo = st.text_input("Agua en martillo (L/h o Sí/No)")
    agua_batidora = st.number_input("Agua en batidora (L/h)", min_value=0.0, step=1.0)
    agua_bomba = st.number_input("Agua en bomba de masa (L/h)", min_value=0.0, step=1.0)

    bomba_masa = st.number_input("Bomba de masa (Hz)", min_value=0.0, max_value=60.0, step=0.5)
    chapa = st.text_input("Chapa decanter")
    graso_orujo = st.number_input("Contenido graso en orujo (%)", min_value=0.0, step=0.1)
    observaciones = st.text_area("Observaciones")
    operario = st.text_input("Operario")

    submit_new = st.form_submit_button("Guardar molienda")

if submit_new:
    if graso_orujo > 3.0:
        st.error("❌ Esta molienda NO es exitosa (graso orujo > 3%).")
    else:
        data = (
            str(fecha), variedad.strip(), float(contenido_graso),
            float(cgms_calc), float(humedad), maquinaria.strip(), float(criba),
            float(temperatura_masa), float(talco), float(bomba_masa),
            agua_martillo.strip(), float(agua_batidora), float(agua_bomba),
            chapa.strip(), float(graso_orujo), observaciones.strip(), operario.strip()
        )
        insertar_molienda(data)
        st.success("✅ Molienda guardada correctamente.")

# ---------- REGISTROS ----------
st.subheader("📋 Moliendas registradas")
df = leer_df()
if df.empty:
    st.info("Aún no hay registros.")
else:
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce").dt.date
    st.dataframe(df, use_container_width=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar CSV", csv, "moliendas.csv", "text/csv")

    st.subheader("🗑️ Eliminar registro")
    rid = st.number_input("ID a eliminar", min_value=0, step=1)
    if st.button("Eliminar"):
        eliminar_registro(int(rid))
        st.warning(f"Registro {int(rid)} eliminado.")

