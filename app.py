import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
from sqlalchemy import create_engine, text
import os

st.set_page_config(page_title="Condiciones Óptimas de Molturación", layout="wide")

# ==========================================================
# 🔧 Conexión a base de datos
# ==========================================================
db_url = os.getenv("DATABASE_URL")
if db_url:
    engine = create_engine(db_url)
    st.sidebar.success("🟢 Conectado a PostgreSQL (Render)")
else:
    engine = create_engine("sqlite:///moliendas.db")
    st.sidebar.info("📦 Modo local (SQLite)")

# ==========================================================
# 🧮 FUNCIONES AUXILIARES
# ==========================================================
def grasa_sobre_seco(graso, humedad):
    try:
        return graso / (100 - humedad) * 100
    except ZeroDivisionError:
        return 0

def moda(lista):
    if not lista:
        return None
    c = Counter(lista)
    return c.most_common(1)[0][0]

def resumen_horquilla_moda(serie):
    datos = serie.dropna().tolist()
    if not datos:
        return "-"
    minimo, maximo = min(datos), max(datos)
    m = moda(datos)
    if minimo == maximo:
        return f"{minimo}"
    return f"{minimo} – {maximo} (más frecuente: {m})"

def predecir_chapa_ponderada(df, maquinaria_in, graso_in, humedad_in, k=10, peso_h=0.8, peso_g=0.2):
    """Predice la 'chapa' ponderando HUMEDAD como el factor dominante."""
    if not maquinaria_in.strip():
        return None, "⚠️ Falta maquinaria."
    dfm = df[df["maquinaria"].astype(str).str.lower() == maquinaria_in.strip().lower()].copy()
    dfm = dfm.dropna(subset=["humedad", "contenido_graso", "chapa"])
    if dfm.empty:
        return None, "⚠️ No hay datos en esa maquinaria."
    h = pd.to_numeric(dfm["humedad"], errors="coerce")
    g = pd.to_numeric(dfm["contenido_graso"], errors="coerce")
    x_h = (h - h.min()) / (h.max() - h.min() + 1e-9)
    x_g = (g - g.min()) / (g.max() - g.min() + 1e-9)
    h0 = float(humedad_in)
    g0 = float(graso_in)
    xh0 = (h0 - h.min()) / (h.max() - h.min() + 1e-9)
    xg0 = (g0 - g.min()) / (g.max() - g.min() + 1e-9)
    dist = np.sqrt(peso_h * (x_h - xh0)**2 + peso_g * (x_g - xg0)**2)
    dfm = dfm.assign(_dist=dist).sort_values("_dist").head(int(k))
    eps = 1e-6
    dfm["_w"] = 1.0 / (dfm["_dist"] + eps)
    pesos = {}
    for _, r in dfm.iterrows():
        chapa = str(r["chapa"]).strip()
        if not chapa:
            continue
        pesos[chapa] = pesos.get(chapa, 0.0) + r["_w"]
    if not pesos:
        moda_simple = Counter([str(x).strip() for x in dfm["chapa"].dropna() if str(x).strip()]).most_common(1)
        return (moda_simple[0][0], "🔁 Fallback a moda simple en maquinaria.") if moda_simple else (None, "⚠️ Sin datos de chapa.")
    chapa_pred = max(pesos.items(), key=lambda kv: kv[1])[0]
    detalle = f"🧮 Votación ponderada (k={k}, peso humedad={peso_h:.1f}, peso graso={peso_g:.1f})."
    return chapa_pred, detalle

# ==========================================================
# 📥 Cargar datos existentes
# ==========================================================
with engine.begin() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS moliendas (
    id SERIAL PRIMARY KEY,

            fecha TEXT,
            variedad TEXT,
            contenido_graso REAL,
            humedad REAL,
            cgms REAL,
            criba TEXT,
            temperatura_masa REAL,
            talco REAL,
            bomba_masa REAL,
            agua_martillo TEXT,
            agua_batidora REAL,
            agua_bomba REAL,
            chapa TEXT,
            maquinaria TEXT,
            grasa_orujo REAL
        )
    """))

df = pd.read_sql("SELECT * FROM moliendas", engine)

# ==========================================================
# 🔮 CUADRO PREDICTIVO
# ==========================================================
st.subheader("🔮 Consulta predictiva antes de moler")

if df.empty:
    st.info("Aún no hay datos históricos suficientes para hacer predicciones.")
else:
    colp1, colp2, colp3, colp4 = st.columns([1.2, 1, 1.2, 1])
    with colp1:
        variedades_hist = ["(ninguna)"] + sorted(df["variedad"].dropna().unique().tolist())
        var_sel = st.selectbox("Variedad (historial)", variedades_hist)
        variedad_in = st.text_input("Variedad", value="" if var_sel == "(ninguna)" else var_sel)
    with colp2:
        graso_in = st.number_input("Contenido graso (%)", min_value=0.0, step=0.1)
    with colp3:
        maqs_hist = ["(ninguna)"] + sorted(df["maquinaria"].dropna().unique().tolist())
        maq_sel = st.selectbox("Maquinaria (historial)", maqs_hist)
        maquinaria_in = st.text_input("Maquinaria", value="" if maq_sel == "(ninguna)" else maq_sel)
    with colp4:
        humedad_in = st.number_input("Humedad (%)", min_value=0.0, step=0.1)

    cgms_est = grasa_sobre_seco(graso_in, humedad_in)
    st.caption(f"CGMS estimado: **{cgms_est:.2f}%**")

if st.button("Consultar condiciones óptimas"):
    if not maquinaria_in.strip():
        st.warning("Debes indicar la maquinaria para obtener una predicción.")
    else:
        # Criba por variedad
        if variedad_in.strip():
            criba_var = df[df["variedad"].astype(str).str.lower() == variedad_in.strip().lower()]
            criba_moda = moda(list(criba_var["criba"].dropna()))
        else:
            criba_moda = None

        # Filtramos por maquinaria
        sub = df[df["maquinaria"].astype(str).str.lower() == maquinaria_in.strip().lower()]

        if sub.empty:
            st.warning("No hay moliendas registradas para esa maquinaria.")
        else:
            # Intento 1: rango ±3 %
            filtrado = sub[
                (sub["contenido_graso"].between(graso_in - 3, graso_in + 3)) &
                (sub["humedad"].between(humedad_in - 3, humedad_in + 3))
            ]

            # Si no hay nada, buscar humedad más cercana registrada
            if filtrado.empty:
                sub["dif_humedad"] = abs(sub["humedad"] - humedad_in)
                humedad_mas_cercana = sub.loc[sub["dif_humedad"].idxmin(), "humedad"]
                filtrado = sub[sub["humedad"] == humedad_mas_cercana]
                st.warning(f"No había moliendas en ±3 %. Usando la humedad más cercana registrada: {humedad_mas_cercana:.1f} %.")
            else:
                st.success(f"✅ Basado en {len(filtrado)} molienda(s) similar(es) dentro de ±3 %.")

            # Cálculo de chapa ponderando humedad
            chapa_pred, _ = predecir_chapa_ponderada(df, maquinaria_in, graso_in, humedad_in)

            agua_mart_moda = moda([str(x).strip() for x in filtrado["agua_martillo"].dropna() if str(x).strip()])
            temp_media = filtrado["temperatura_masa"].mean()
            talco_desc = resumen_horquilla_moda(filtrado["talco"])
            bomba_desc = resumen_horquilla_moda(filtrado["bomba_masa"])
            agua_batidora_desc = resumen_horquilla_moda(filtrado["agua_batidora"])
            agua_bomba_desc = resumen_horquilla_moda(filtrado["agua_bomba"])

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Criba (según variedad):** {criba_moda if criba_moda else '-'}")
                st.markdown(f"**Temperatura masa (°C):** {temp_media:.1f}" if not np.isnan(temp_media) else "**Temperatura masa (°C):** -")
                st.markdown(f"**Talco (%):** {talco_desc}")
                st.markdown(f"**Bomba de masa (Hz):** {bomba_desc}")
            with c2:
                st.markdown(f"**Agua en martillo:** {agua_mart_moda if agua_mart_moda else '-'}")
                st.markdown(f"**Agua en batidora (L/h):** {agua_batidora_desc}")
                st.markdown(f"**Agua en bomba de masa (L/h):** {agua_bomba_desc}")
                st.markdown(f"**Chapa (pondera HUMEDAD):** {chapa_pred if chapa_pred else '-'}")

            st.caption("ℹ️ Si no hay moliendas en ±3 %, se usa la humedad más cercana registrada en esa maquinaria. "
                       "La **chapa** pondera la humedad; la **criba** se calcula por variedad.")

# ==========================================================
# ➕ REGISTRO DE NUEVA MOLIENDA
# ==========================================================
st.subheader("➕ Registrar nueva molienda")
with st.form("registro_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        fecha = st.date_input("Fecha")
        variedad = st.text_input("Variedad")
        maquinaria = st.text_input("Maquinaria")
    with col2:
        graso = st.number_input("Contenido graso (%)", min_value=0.0, step=0.1)
        humedad = st.number_input("Humedad (%)", min_value=0.0, step=0.1)
        cgms = grasa_sobre_seco(graso, humedad)
    with col3:
        temperatura = st.number_input("Temperatura masa (°C)", min_value=0.0, step=0.1)
        grasa_orujo = st.number_input("Grasa en orujo (%)", min_value=0.0, step=0.1)
        criba = st.text_input("Criba")

    col4, col5, col6 = st.columns(3)
    with col4:
        talco = st.number_input("Talco (%)", min_value=0.0, step=0.1)
    with col5:
        bomba = st.number_input("Bomba de masa (Hz)", min_value=0.0, step=0.1)
    with col6:
        agua_martillo = st.text_input("Agua en martillo")
        agua_batidora = st.number_input("Agua en batidora (L/h)", min_value=0.0, step=0.1)
        agua_bomba = st.number_input("Agua en bomba (L/h)", min_value=0.0, step=0.1)

    chapa = st.text_input("Chapa")

    if st.form_submit_button("Guardar molienda"):
        nueva = pd.DataFrame([{
            "fecha": fecha,
            "variedad": variedad,
            "contenido_graso": graso,
            "humedad": humedad,
            "cgms": cgms,
            "criba": criba,
            "temperatura_masa": temperatura,
            "talco": talco,
            "bomba_masa": bomba,
            "agua_martillo": agua_martillo,
            "agua_batidora": agua_batidora,
            "agua_bomba": agua_bomba,
            "chapa": chapa,
            "maquinaria": maquinaria,
            "grasa_orujo": grasa_orujo
        }])
        nueva.to_sql("moliendas", engine, if_exists="append", index=False)
        st.success("✅ Molienda guardada correctamente.")

# ==========================================================
# 🧾 LISTADO, EDICIÓN Y ELIMINACIÓN
# ==========================================================
st.subheader("📋 Registros existentes")
if not df.empty:
    st.dataframe(df.sort_values("fecha", ascending=False))

    id_sel = st.number_input("ID de molienda a editar o eliminar", min_value=0, step=1)
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Eliminar registro"):
            with engine.begin() as conn:
                conn.execute(text("DELETE FROM moliendas WHERE id = :id"), {"id": id_sel})
            st.warning("🗑️ Registro eliminado.")
    with col_b:
        if st.button("Editar molienda"):
            st.info("✏️ Introduce los nuevos valores en el formulario superior y guarda con el mismo ID.")
else:
    st.info("No hay registros todavía.")
