# ============================================
# APP: Condiciones Óptimas de Molturación
# Versión: PostgreSQL + Predictiva + Edición
# Con autocompletado de Variedad y Maquinaria
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from datetime import date
from collections import Counter
import os

# -----------------------------------------------------
# CONFIGURACIÓN GENERAL DE LA APP
# -----------------------------------------------------
st.set_page_config(page_title="Condiciones Óptimas de Molturación", page_icon="🫒", layout="wide")

# Conexión a PostgreSQL vía variable de entorno DATABASE_URL (Render)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///moliendas.db")
engine = create_engine(DATABASE_URL, echo=False)

# -----------------------------------------------------
# CREAR TABLA SI NO EXISTE
# -----------------------------------------------------
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

# -----------------------------------------------------
# FUNCIONES DE BASE DE DATOS
# -----------------------------------------------------

def leer_df():
    """Lee todas las moliendas (orden descendente por id)."""
    with engine.begin() as conn:
        return pd.read_sql("SELECT * FROM moliendas ORDER BY id DESC", conn)

def insertar_molienda(data_tuple):
    """Inserta una molienda."""
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
    """Elimina una molienda por ID."""
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM moliendas WHERE id = :id"), {"id": registro_id})

def actualizar_molienda(id_, data):
    """Actualiza una molienda existente (fecha no editable)."""
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

# -----------------------------------------------------
# AUXILIARES DE CÁLCULO Y UI
# -----------------------------------------------------

def grasa_sobre_seco(graso, humedad):
    """CGMS = graso / (1 - humedad/100)."""
    try:
        return float(graso) / (1 - float(humedad) / 100)
    except Exception:
        return np.nan

def safe_float(x):
    """Convierte a float o devuelve 0.0 si no es convertible."""
    try:
        return float(x)
    except Exception:
        return 0.0

def moda(lista):
    """Moda (valor más repetido) o None si vacío."""
    if not lista:
        return None
    return Counter(lista).most_common(1)[0][0]

def unique_sorted(series: pd.Series):
    """Lista ordenada de valores únicos (texto) para autocompletar suave."""
    if series is None or series.empty:
        return []
    return sorted({str(v).strip() for v in series.dropna() if str(v).strip()})

def resumen_horquilla_moda(vals_num):
    """
    Dada una serie numérica, devuelve string: 'min – max (moda: x)'.
    Si hay un único valor, lo devuelve como 'x'.
    """
    vals = pd.to_numeric(pd.Series(vals_num), errors="coerce").dropna()
    if len(vals) == 0:
        return "-"
    if len(vals) == 1:
        return f"{vals.iloc[0]:.1f}"
    return f"{vals.min():.1f} – {vals.max():.1f} (moda: {moda(list(vals)):.1f})"

# -----------------------------------------------------
# INTERFAZ PRINCIPAL
# -----------------------------------------------------
st.title("🫒 Condiciones Óptimas de Molturación")

df = leer_df()
if not df.empty:
    # Normalizamos tipos básicos por si hay mezclas texto/número
    for col in ["contenido_graso", "contenido_graso_seco", "humedad", "criba",
                "temperatura_masa", "talco", "bomba_masa", "agua_batidora",
                "agua_bomba", "graso_orujo"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

# =======================
# 🔮 CUADRO PREDICTIVO
# =======================
st.subheader("🔮 Consulta predictiva antes de moler")

if df.empty:
    st.info("Aún no hay datos históricos suficientes para hacer predicciones.")
else:
    # --- Autocompletado suave de Variedad y Maquinaria ---
    colp1, colp2, colp3, colp4 = st.columns([1.2, 1, 1.2, 1])
    with colp1:
        variedades_hist = ["(ninguna)"] + unique_sorted(df["variedad"])
        var_sel = st.selectbox("Variedad (historial)", variedades_hist)
        variedad_in = st.text_input("Variedad", value="" if var_sel == "(ninguna)" else var_sel)
    with colp2:
        graso_in = st.number_input("Contenido graso (%)", min_value=0.0, step=0.1)
    with colp3:
        maqs_hist = ["(ninguna)"] + unique_sorted(df["maquinaria"])
        maq_sel = st.selectbox("Maquinaria (historial)", maqs_hist)
        maquinaria_in = st.text_input("Maquinaria", value="" if maq_sel == "(ninguna)" else maq_sel)
    with colp4:
        humedad_in = st.number_input("Humedad (%)", min_value=0.0, step=0.1)

    cgms_est = grasa_sobre_seco(graso_in, humedad_in)
    st.caption(f"CGMS estimado: **{cgms_est:.2f}%**")

    if st.button("Consultar condiciones óptimas"):
        # Filtro por variedad (obligatorio) y maquinaria (opcional),
        # y ventana ±3% en graso y humedad
        sub = df.copy()
        sub = sub[sub["variedad"].astype(str).str.lower() == str(variedad_in).strip().lower()]

        if maquinaria_in.strip():
            sub = sub[sub["maquinaria"].astype(str).str.lower() == maquinaria_in.strip().lower()]

        if sub.empty:
            st.warning("No hay moliendas con esa combinación de Variedad y/o Maquinaria.")
        else:
            sub = sub[
                (sub["contenido_graso"].between(graso_in - 3, graso_in + 3)) &
                (sub["humedad"].between(humedad_in - 3, humedad_in + 3))
            ]

            if sub.empty:
                st.warning("No hay moliendas con parámetros de graso/humedad similares (±3%).")
            else:
                # Recomendaciones:
                # - Criba y Chapa: mayoritaria (moda)
                # - Temperatura: media
                # - Talco, Bomba, Agua batidora, Agua bomba: horquilla min–max + moda
                # - Agua martillo: moda (texto)
                criba_moda = moda(list(sub["criba"].dropna()))
                chapa_moda = moda([str(x).strip() for x in sub["chapa"].dropna() if str(x).strip()])
                agua_mart_moda = moda([str(x).strip() for x in sub["agua_martillo"].dropna() if str(x).strip()])
                temp_media = sub["temperatura_masa"].mean()

                talco_desc = resumen_horquilla_moda(sub["talco"])
                bomba_desc = resumen_horquilla_moda(sub["bomba_masa"])
                agua_batidora_desc = resumen_horquilla_moda(sub["agua_batidora"])
                agua_bomba_desc = resumen_horquilla_moda(sub["agua_bomba"])

                st.success(f"✅ Basado en {len(sub)} molienda(s) similar(es).")
                # Orden de aguas: Martillo → Batidora → Bomba
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Criba (mayoritaria):** {criba_moda if criba_moda is not None else '-'}")
                    st.markdown(f"**Temperatura masa (°C):** {temp_media:.1f}" if not np.isnan(temp_media) else "**Temperatura masa (°C):** -")
                    st.markdown(f"**Talco (%):** {talco_desc}")
                    st.markdown(f"**Bomba de masa (Hz):** {bomba_desc}")
                with c2:
                    st.markdown(f"**Agua en martillo:** {agua_mart_moda if agua_mart_moda else '-'}")
                    st.markdown(f"**Agua en batidora (L/h):** {agua_batidora_desc}")
                    st.markdown(f"**Agua en bomba de masa (L/h):** {agua_bomba_desc}")
                    st.markdown(f"**Chapa (mayoritaria):** {chapa_moda if chapa_moda else '-'}")

# =======================
# ➕ NUEVA MOLIENDA
# =======================
st.subheader("➕ Registrar molienda exitosa")

with st.form("nueva_molienda"):
    c1, c2, c3 = st.columns(3)
    with c1:
        fecha = st.date_input("Fecha", value=date.today(), min_value=date(2000, 1, 1),
                              max_value=date.today(), format="DD/MM/YYYY")
        # Autocompletado suave de Variedad
        variedades_hist = ["(ninguna)"] + (unique_sorted(df["variedad"]) if not df.empty else [])
        var_sel2 = st.selectbox("Variedad (historial)", variedades_hist, key="var_sel2")
        variedad = st.text_input("Variedad", value="" if var_sel2 == "(ninguna)" else var_sel2, key="variedad_new")
        graso = st.number_input("Contenido graso (%)", min_value=0.0, step=0.1)
    with c2:
        humedad = st.number_input("Humedad (%)", min_value=0.0, step=0.1)
        cgms_calc = grasa_sobre_seco(graso, humedad)
        st.info(f"CGMS calculado: **{cgms_calc:.2f}%**")
        # Autocompletado suave de Maquinaria
        maqs_hist = ["(ninguna)"] + (unique_sorted(df["maquinaria"]) if not df.empty else [])
        maq_sel2 = st.selectbox("Maquinaria (historial)", maqs_hist, key="maq_sel2")
        maquinaria = st.text_input("Maquinaria", value="" if maq_sel2 == "(ninguna)" else maq_sel2, key="maquinaria_new")
    with c3:
        criba = st.number_input("Criba (mm)", min_value=0.0, step=0.1)
        temperatura = st.number_input("Temperatura masa (°C)", min_value=0.0, step=0.5)
        talco = st.number_input("Talco (%)", min_value=0.0, step=0.1)

    # Orden de aguas: Martillo → Batidora → Bomba
    agua_martillo = st.text_input("Agua en martillo (texto o L/h)")
    agua_batidora = st.number_input("Agua en batidora (L/h)", min_value=0.0, step=1.0)
    agua_bomba = st.number_input("Agua en bomba de masa (L/h)", min_value=0.0, step=1.0)

    bomba = st.number_input("Bomba masa (Hz)", min_value=0.0, step=0.5)
    chapa = st.text_input("Chapa decanter")
    graso_orujo = st.number_input("Graso orujo (%)", min_value=0.0, step=0.1)
    observaciones = st.text_area("Observaciones")
    operario = st.text_input("Operario")

    submit_new = st.form_submit_button("Guardar molienda")

if submit_new:
    data = (str(fecha), variedad.strip(), safe_float(graso), safe_float(cgms_calc), safe_float(humedad),
            maquinaria.strip(), safe_float(criba), safe_float(temperatura), safe_float(talco),
            safe_float(bomba), agua_martillo.strip(), safe_float(agua_batidora),
            safe_float(agua_bomba), chapa.strip(), safe_float(graso_orujo),
            observaciones.strip(), operario.strip())
    insertar_molienda(data)
    st.success("✅ Nueva molienda guardada correctamente.")
    st.rerun()

# =======================
# ✏️ EDITAR MOLIENDA
# =======================
st.subheader("✏️ Editar molienda")

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
    st.write(f"**Editando ID {st.session_state.edit_id} — Fecha:** {fila['fecha']} (no editable)")
    st.caption("⚠️ Al guardar se sobreescribirá el registro seleccionado.")

    with st.form("editar_molienda"):
        c1e, c2e, c3e = st.columns(3)
        with c1e:
            # Autocompletado suave de Variedad (edición)
            variedades_hist = ["(ninguna)"] + (unique_sorted(df["variedad"]) if not df.empty else [])
            var_sel3 = st.selectbox("Variedad (historial)", variedades_hist, index=variedades_hist.index("(ninguna)") if fila["variedad"] not in variedades_hist else variedades_hist.index(str(fila["variedad"])))
            variedad = st.text_input("Variedad", value=fila["variedad"] if var_sel3 == "(ninguna)" else var_sel3)
            graso = st.number_input("Contenido graso (%)", value=safe_float(fila["contenido_graso"]), step=0.1)
            humedad = st.number_input("Humedad (%)", value=safe_float(fila["humedad"]), step=0.1)
        with c2e:
            cgms_calc = grasa_sobre_seco(graso, humedad)
            st.caption(f"CGMS recalculado: **{cgms_calc:.2f}%**")
            # Autocompletado suave de Maquinaria (edición)
            maqs_hist = ["(ninguna)"] + (unique_sorted(df["maquinaria"]) if not df.empty else [])
            pre_idx = maqs_hist.index("(ninguna)") if str(fila["maquinaria"]) not in maqs_hist else maqs_hist.index(str(fila["maquinaria"]))
            maq_sel3 = st.selectbox("Maquinaria (historial)", maqs_hist, index=pre_idx)
            maquinaria = st.text_input("Maquinaria", value=fila["maquinaria"] if maq_sel3 == "(ninguna)" else maq_sel3)
            criba = st.number_input("Criba (mm)", value=safe_float(fila["criba"]), step=0.1)
            temperatura = st.number_input("Temperatura masa (°C)", value=safe_float(fila["temperatura_masa"]), step=0.5)
        with c3e:
            talco = st.number_input("Talco (%)", value=safe_float(fila["talco"]), step=0.1)
            bomba = st.number_input("Bomba masa (Hz)", value=safe_float(fila["bomba_masa"]), step=0.5)
            # Orden de aguas: Martillo → Batidora → Bomba
            agua_martillo = st.text_input("Agua martillo", value=str(fila["agua_martillo"]))
            agua_batidora = st.number_input("Agua batidora (L/h)", value=safe_float(fila["agua_batidora"]), step=1.0)
            agua_bomba = st.number_input("Agua bomba masa (L/h)", value=safe_float(fila["agua_bomba"]), step=1.0)
            chapa = st.text_input("Chapa decanter", value=fila["chapa"])

        graso_orujo = st.number_input("Graso orujo (%)", value=safe_float(fila["graso_orujo"]), step=0.1)
        observaciones = st.text_area("Observaciones", value=fila["observaciones"])
        operario = st.text_input("Operario", value=fila["operario"])
        submit_edit = st.form_submit_button("Guardar cambios")

    if submit_edit:
        data = (variedad.strip(), safe_float(graso), safe_float(cgms_calc), safe_float(humedad),
                maquinaria.strip(), safe_float(criba), safe_float(temperatura), safe_float(talco),
                safe_float(bomba), agua_martillo.strip(), safe_float(agua_batidora),
                safe_float(agua_bomba), chapa.strip(), safe_float(graso_orujo),
                observaciones.strip(), operario.strip())
        actualizar_molienda(st.session_state.edit_id, data)
        st.success("✅ Cambios guardados correctamente.")
        del st.session_state["edit_data"]
        del st.session_state["edit_id"]
        st.rerun()

# =======================
# 🗑️ ELIMINAR MOLIENDA
# =======================
st.subheader("🗑️ Eliminar molienda")
delete_id = st.number_input("ID a eliminar", min_value=1, step=1)
if st.button("Eliminar registro"):
    eliminar_registro(delete_id)
    st.warning(f"Registro {delete_id} eliminado correctamente.")
    st.rerun()

# =======================
# 📋 LISTADO FINAL
# =======================
st.subheader("📋 Moliendas registradas")
if df.empty:
    st.info("No hay registros todavía.")
else:
    st.dataframe(df, use_container_width=True)

