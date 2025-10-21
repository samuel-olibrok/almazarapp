# app.py (versión definitiva PostgreSQL)
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
def leer_df():
    with engine.begin() as conn:
        return pd.read_sql("SELECT * FROM moliendas ORDER BY id DESC", conn)

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
            fecha=data_tuple[0], variedad=data_tuple[1], contenido_graso=data_tuple[2],
            contenido_graso_seco=data_tuple[3], humedad=data_tuple[4], maquinaria=data_tuple[5],
            criba=data_tuple[6], temperatura_masa=data_tuple[7], talco=data_tuple[8],
            bomba_masa=data_tuple[9], agua_martillo=data_tuple[10], agua_batidora=data_tuple[11],
            agua_bomba=data_tuple[12], chapa=data_tuple[13], graso_orujo=data_tuple[14],
            observaciones=data_tuple[15], operario=data_tuple[16]
        ))

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
            id=id_, variedad=data[0], contenido_graso=data[1], contenido_graso_seco=data[2],
            humedad=data[3], maquinaria=data[4], criba=data[5], temperatura_masa=data[6],
            talco=data[7], bomba_masa=data[8], agua_martillo=data[9], agua_batidora=data[10],
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

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return 0.0

# =========================
# APP
# =========================
st.title("🫒 Condiciones Óptimas de Molturación")

df = leer_df()

# ➕ NUEVA MOLIENDA
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
    st.success("✅ Molienda guardada correctamente en PostgreSQL.")
    st.rerun()

# 📋 LISTADO DE MOLIENDAS
st.subheader("📋 Moliendas registradas")
if df.empty:
    st.info("No hay registros todavía.")
else:
    st.dataframe(df, use_container_width=True)

# ✏️ EDITAR MOLIENDA
st.subheader("✏️ Editar molienda existente")

if "edit_id" not in st.session_state:
    st.session_state.edit_id = None

edit_id_input = st.number_input("ID a editar", min_value=1, step=1, value=st.session_state.edit_id or 1)
if st.button("Cargar molienda"):
    fila = df[df["id"] == edit_id_input]
    if fila.empty:
        st.error("No se encontró una molienda con ese ID.")
    else:
        st.session_state.edit_id = int(edit_id_input)
        st.session_state.edit_data = fila.iloc[0].to_dict()
        st.rerun()

if st.session_state.get("edit_data"):
    fila = st.session_state.edit_data
    st.write(f"**Editando molienda ID {st.session_state.edit_id} - Fecha:** {fila['fecha']} (no editable)")
    st.caption("⚠️ Al guardar se sobreescribirá el registro existente.")

    with st.form("editar_molienda"):
        variedad = st.text_input("Variedad", value=fila["variedad"])
        contenido_graso = st.number_input("Contenido graso (%)", value=safe_float(fila["contenido_graso"]), step=0.1)
        humedad = st.number_input("Humedad (%)", value=safe_float(fila["humedad"]), step=0.1)
        cgms_calc = grasa_sobre_seco(contenido_graso, humedad)
        maquinaria = st.text_input("Maquinaria", value=fila["maquinaria"])
        criba = st.number_input("Criba (mm)", value=safe_float(fila["criba"]), step=0.1)
        temperatura_masa = st.number_input("Temperatura masa (°C)", value=safe_float(fila["temperatura_masa"]), step=0.5)
        talco = st.number_input("Talco (%)", value=safe_float(fila["talco"]), step=0.1)
        bomba_masa = st.number_input("Bomba de masa (Hz)", value=safe_float(fila["bomba_masa"]), step=0.5)
        agua_martillo = st.text_input("Agua en martillo", value=str(fila["agua_martillo"]))
        agua_batidora = st.number_input("Agua en batidora (L/h)", value=safe_float(fila["agua_batidora"]), step=1.0)
        agua_bomba = st.number_input("Agua en bomba de masa (L/h)", value=safe_float(fila["agua_bomba"]), step=1.0)
        chapa = st.text_input("Chapa decanter", value=fila["chapa"])
        graso_orujo = st.number_input("Graso orujo (%)", value=safe_float(fila["graso_orujo"]), step=0.1)
        observaciones = st.text_area("Observaciones", value=fila["observaciones"])
        operario = st.text_input("Operario", value=fila["operario"])
        submit_edit = st.form_submit_button("Guardar cambios")

    if submit_edit:
        data = (variedad, contenido_graso, cgms_calc, humedad, maquinaria, criba,
                temperatura_masa, talco, bomba_masa, agua_martillo,
                agua_batidora, agua_bomba, chapa, graso_orujo,
                observaciones, operario)
        actualizar_molienda(st.session_state.edit_id, data)
        st.success("✅ Cambios guardados correctamente.")
        del st.session_state["edit_data"]
        del st.session_state["edit_id"]
        st.rerun()

# 🗑️ ELIMINAR MOLIENDA
st.subheader("🗑️ Eliminar molienda")
delete_id = st.number_input("ID a eliminar", min_value=1, step=1)
if st.button("Eliminar registro"):
    eliminar_registro(delete_id)
    st.warning(f"Registro {delete_id} eliminado correctamente.")
    st.rerun()

