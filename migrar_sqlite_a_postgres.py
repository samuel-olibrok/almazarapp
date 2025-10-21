import sqlite3
import pandas as pd
from sqlalchemy import create_engine

# 1️⃣ Ruta de tu base local
sqlite_file = "moliendas.db"

# 2️⃣ URL de conexión a PostgreSQL Render
pg_url = "postgresql://almazara_user:Umm8GTrdHYvtbBsT5rcUv5sO6OFznfog@dpg-d3rmd7gdl3ps73fhuvo0-a.frankfurt-postgres.render.com/almazara"

# 3️⃣ Conexiones
sqlite_conn = sqlite3.connect(sqlite_file)
pg_engine = create_engine(pg_url)

# 4️⃣ Leer datos locales
df = pd.read_sql("SELECT * FROM moliendas", sqlite_conn)
print(f"📦 Registros encontrados en SQLite: {len(df)}")

# 5️⃣ Escribirlos en PostgreSQL
df.to_sql("moliendas", pg_engine, if_exists="append", index=False)
print("✅ Migración completada con éxito. Todos los registros se han copiado a Render.")

sqlite_conn.close()
