# app.py (versión PostgreSQL Render)
import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from datetime import date
from collections import Counter
import os

# =========================
# CONFIGURACIÓN
# =========================
st.set_page_config(page_title="Condiciones Óptimas de Molturación", page_icon="🫒", layout="wide")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///moliendas.db")
engine = create_engine(DATABASE_URL, echo=False)

# =========================
# CREAR TABLA SI NO EXISTE
# =========================
def crear_tabla_si_no_existe():
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS moliendas (
                id SERIAL PRIMARY KEY,
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
            );
        """))
crear_tabla_si_no_existe()

# =========================
# FUNCIONES BÁSICAS
# =========================
def insertar_molienda(data_tuple):
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO moliendas (
                fecha, variedad, contenido_graso, contenido_graso_seco, humedad,
                maquinaria, criba, temperatura_masa, talco, bomba_masa,
                agua_martillo, agua_batidora, agua_bomba, chapa,
                graso_orujo, observaciones, operario
            ) VALUES (
                :fecha, :variedad, :contenido_graso, :contenido_graso_seco, :humedad,
                :maquinaria, :criba, :temperatura_masa, :talco, :bomba_masa,
                :agua_martillo, :agua_batidora, :agua_bomba, :chapa,
                :graso_orujo, :observaciones, :operario
            )
        """), dict(
            fecha=data_tuple[0],
            variedad=data_tuple[1],
            contenido_graso=data_tuple[2],
            contenido_graso_seco=data_tuple[3],
            humedad=data_tuple[4],
            maquinaria=data_tuple[5],
            criba=data_tuple[6],
            temperatura_masa=data_tuple[7],
            talco=data_tuple[8],
            bomba_masa=data_tuple[9],
            agua_martillo=data_tuple[10],
            agua_batidora=data_tuple[11],
            agua_bomba=data_tuple[12],
            chapa=data_tuple[13],
            graso_orujo=data_tuple[14],
            observaciones=data_tuple[15],
            operario=data_tuple[16]
        ))

def leer_df():
    with engine.begin() as conn:
        df = pd.read_sql("SELECT * FROM moliendas ORDER BY id DESC", conn)
    return df

def eliminar_registro(registro_id):
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM moliendas WHERE id = :id"), {"id": registro_id})

def actualizar_molienda(id_, data):
    with engine.begin() as conn:
        conn.execute(text("""
            UPDATE moliendas SET
                variedad=:variedad, contenido_graso=:contenido_graso,
                contenido_graso_seco=:contenido_graso_seco, humedad=:humedad,
                maquinaria=:maquinaria, criba=:criba, temperatura_masa=:temperatura_masa,
                talco=:talco, bomba_masa=:bomba_masa, agua_martillo=:agua_martillo,
                agua_batidora=:agua_batidora, agua_bomba=:agua_bomba, chapa=:chapa,
                graso_orujo=:graso_orujo, observaciones=:observaciones, operario=:operario
            WHERE id=:id
        """), dict(
            id=id_,
            variedad=data[0], contenido_graso=data[1], contenido_graso_seco=data[2], humedad=data[3],
            maquinaria=data[4], criba=data[5], temperatura_masa=data[6], talco=data[7],
            bomba_masa=data[8], agua_martillo=data[9], agua_batidora=data[10],
            agua_bomba=data[11], chapa=data[12], graso_orujo=data[13],
            observaciones=data[14], operario=data[15]
        ))

# =========================
# AUXILIARES
# =========================
def grasa_sobre_seco(graso, humedad):
    try:
        return float(graso) / (1 - float(humedad) / 100)
    except Exception:
        return np.nan

def unique_sorted(series):
    vals = sorted({v for v in series.dropna().astype(str) if v.strip()})
    return vals

def moda_o_vacio(vals):
    c = Counter(vals)
    return c.most_common(1)[0][0] if c else None

# =========================
# APP
# =========================
st.title("🫒 Condiciones Óptimas de Molturación")

df_all = leer_df()
if not df_all.empty:
    df_all["fecha"] = pd.to_datetime(df_all["fecha"], errors="coerce").dt.date

# 🔎 Consulta predictiva
st.subheader("🔎 Consulta predictiva (antes de moler)")
with st.form("predictiva"):
    c1, c2, c3 = st.columns(3)
    variedades = unique_sorted(df_all["variedad"]) if not df_all.empty else []
    var_sel = st.selectbox("Variedad (historial)", ["(ninguna)"] + variedades)
    variedad = st.text_input("Variedad", value=("" if var_sel == "(ninguna)" else var_sel))
    graso_in = st.number_input("Contenido graso (%)", min_value=0.0, step=0.1)
    humedad_in = st.number_input("Humedad (%)", min_value=0.0, step=0.1)
    submit_pred = st.form_submit_button("Consultar")

if submit_pred:
    st.info("Funcionalidad predictiva en preparación para versión PostgreSQL.")

# ➕ Nueva molienda
st.subheader("➕ Añadir nueva molienda exitosa")
with st.form("nueva_molienda"):
    fecha = st.date_input("Fecha (DD/MM/AAAA)", value=date.today(), min_value=date(2000, 1, 1), max_value=date.today(), format="DD/MM/YYYY")
    variedad = st.text_input("Variedad")
    contenido_graso = st.number_input("Contenido graso (%)", min_value=0.0, step=0.1)
    humedad = st.number_input("Humedad (%)", min_value=0.0, step=0.1)
    cgms_calc = grasa_sobre_seco(contenido_graso, humedad)
    st.info(f"CGMS calculado: {cgms_calc:.2f}%")
    maquinaria = st.text_input("Maquinaria")
    criba = st.number_input("Criba (mm)", min_value=0.0, step=0.1)
    temperatura_masa = st.number_input("Temperatura masa (°C)", min_value=0.0, step=0.5)
    talco = st.number_input("Talco (%)", min_value=0.0, step=0.1)
    bomba_masa = st.number_input("Bomba de masa (Hz)", min_value=0.0, step=0.5)
    agua_martillo = st.text_input("Agua en martillo")
    agua_batidora = st.number_input("Agua en batidora (L/h)", min_value=0.0, step=1.0)
    agua_bomba = st.number_input("Agua en bomba de masa (L/h)", min_value=0.0, step=1.0)
    chapa = st.text_input("Chapa decanter")
    graso_orujo = st.number_input("Contenido graso en orujo (%)", min_value=0.0, step=0.1)
    observaciones = st.text_area("Observaciones")
    operario = st.text_input("Operario")
    submit_new = st.form_submit_button("Guardar molienda")

if submit_new:
    data = (str(fecha), variedad, contenido_graso, cgms_calc, humedad, maquinaria, criba,
            temperatura_masa, talco, bomba_masa, agua_martillo, agua_batidora,
            agua_bomba, chapa, graso_orujo, observaciones, operario)
    insertar_molienda(data)
    st.success("✅ Molienda guardada correctamente en la base de datos de Render.")

# 📋 Registros
st.subheader("📋 Moliendas registradas")
df = leer_df()
if df.empty:
    st.info("No hay registros todavía.")
else:
    st.dataframe(df, use_container_width=True)
