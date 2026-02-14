import streamlit as st
import pandas as pd
import requests
import os
from datetime import datetime

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="GDI: Mendoza Ops", layout="wide", page_icon="🧥")

FILE_INV = 'inventory.csv'

# --- FUNCIONES AUXILIARES SNA (INGENIERÍA) ---
def decodificar_sna(codigo):
    """
    Parsea el código SNA manejando la longitud variable de 'CS' vs 'R'.
    """
    try:
        season = codigo[0]
        # Detectar si es Camisa (CS) o el resto (1 letra)
        if len(codigo) > 2 and codigo[1:3] == 'CS':
            tipo = 'CS'
            idx_start_attr = 3
        else:
            tipo = codigo[1]
            idx_start_attr = 2
            
        attr = codigo[idx_start_attr : idx_start_attr + 2]
        idx_occ = idx_start_attr + 2
        occasion = codigo[idx_occ]
        
        return {"tipo": tipo, "attr": attr, "occasion": occasion}
    except:
        return None

def load_data():
    if not os.path.exists(FILE_INV):
        return pd.DataFrame(columns=['Code', 'Category', 'Season', 'Occasion', 'ImageURL', 'Status', 'LastWorn'])
    df = pd.read_csv(FILE_INV)
    df['Code'] = df['Code'].astype(str) 
    return df

def save_data(df):
    df.to_csv(FILE_INV, index=False)

def get_weather(api_key):
    lat, lon = -32.8908, -68.8272 
    if not api_key:
        return {"temp": 24, "min": 18, "max": 30, "desc": "Modo Demo (Zonda)"}
        
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric&lang=es"
        res = requests.get(url).json()
        return {
            "temp": res['main']['temp'], 
            "min": res['main']['temp_min'], 
            "max": res['main']['temp_max'], 
            "desc": res['weather'][0]['description'].capitalize()
        }
    except:
        st.error("Error de conexión con OpenWeather.")
        return {"temp": 15, "min": 10, "max": 20, "desc": "Error API"}

# --- LÓGICA DE RECOMENDACIÓN TÉRMICA ---
def recommend_outfit(df, weather, occasion):
    clean_df = df[df['Status'] == 'Limpio'].copy()
    if clean_df.empty: return pd.DataFrame()

    temp_real = weather['temp']
    temp_percibida = temp_real + 3 # Ajuste usuario caluroso
    
    recs = []
    
    for index, row in clean_df.iterrows():
        sna = decodificar_sna(row['Code'])
        if not sna: continue
        
        if sna['occasion'] != occasion: continue

        # Lógica térmica simplificada para brevedad
        if row['Category'] == 'Pantalón':
            tipo_pan = sna['attr']
            if temp_percibida > 26 and tipo_pan in ['Sh', 'DC']: recs.append(row)
            elif temp_percibida > 20 and tipo_pan in ['Je', 'Ve', 'DL', 'Sh']: recs.append(row)
            elif temp_percibida <= 20 and tipo_pan in ['Je', 'Ve', 'DL']: recs.append(row)

        elif row['Category'] in ['Remera', 'Camisa']:
            manga = sna['attr']
            if temp_percibida > 25 and manga in ['00', '01']: recs.append(row)
            elif temp_percibida < 15 and manga in ['02', '01']: recs.append(row)
            else: recs.append(row)

        elif row['Category'] in ['Campera', 'Buzo']:
            nivel = int(sna['attr'])
            if temp_percibida < 12 and nivel >= 4: recs.append(row)
            elif temp_percibida < 18 and nivel in [2, 3]: recs.append(row)
            elif temp_percibida < 22 and nivel == 1: recs.append(row)

    return pd.DataFrame(recs)

# --- INTERFAZ DE USUARIO ---
st.sidebar.title("🛠️ GDI: Mendoza Ops")
api_key = st.sidebar.text_input("API Key (OpenWeather)", type="password")
user_occ = st.sidebar.selectbox("Ocasión Actual", ["U (Universidad)", "D (Deporte)", "C (Casa)", "F (Formal)"])
code_occ = user_occ[0]

# Cargar datos en Session State
if 'inventory' not in st.session_state:
    st.session_state['inventory'] = load_data()
df = st.session_state['inventory']

weather = get_weather(api_key)

# AHORA SON 4 PESTAÑAS
tab1, tab2, tab3, tab4 = st.tabs(["🔥 Sugerencia", "🧺 Lavadero", "📋 Inventario General", "➕ Carga Manual"])

# --- TAB 1: SUGERENCIA ---
with tab1:
    st.markdown(f"### 🌡️ Clima: {weather['temp']}°C (Sensación: {weather['temp']+3}°C)")
    st.caption(f"Condición: {weather['desc']}")
    
    if (weather['max'] - weather['min']) > 15:
        st.warning("⚠️ Alerta Zonda/Amplitud! Usá capas.")

    recs_df = recommend_outfit(df, weather, code_occ)
    
    if not recs_df.empty:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Base**")
            base = recs_df[recs_df['Category'].isin(['Remera', 'Camisa'])]
            if not base.empty:
                item = base.sample(1).iloc[0]
                st.info(f"{item['Category']} ({item['Code']})")
                if pd.notna(item['ImageURL']) and item['ImageURL']: st.image(item['ImageURL'])
        with col2:
            st.markdown("**Piernas**")
            legs = recs_df[recs_df['Category'] == 'Pantalón']
            if not legs.empty:
                item = legs.sample(1).iloc[0]
                st.success(f"{item['Category']} ({item['Code']})")
                if pd.notna(item['ImageURL']) and item['ImageURL']: st.image(item['ImageURL'])
        with col3:
            st.markdown("**Abrigo**")
            outer = recs_df[recs_df['Category'].isin(['Campera', 'Buzo'])]
            if not outer.empty:
                item = outer.sample(1).iloc[0]
                st.warning(f"{item['Category']} ({item['Code']})")
                if pd.notna(item['ImageURL']) and item['ImageURL']: st.image(item['ImageURL'])
            else:
                st.write("No necesitás abrigo.")
    else:
        st.error("No hay ropa limpia para este clima.")

# --- TAB 2: LAVADERO (Solo Status) ---
with tab2:
    st.subheader("Operaciones de Lavado")
    st.info("Aquí solo cambiamos el estado de las prendas (Limpio ↔ Sucio).")
    
    # Mostramos solo columnas relevantes para el lavado
    columnas_lavado = ['Code', 'Category', 'Status', 'LastWorn']
    
    edited_laundry = st.data_editor(
        df[columnas_lavado], 
        key="editor_lavadero",
        column_config={
            "Status": st.column_config.SelectboxColumn("Estado", options=["Limpio", "Sucio", "Lavando"], required=True)
        },
        hide_index=True,
        disabled=["Code", "Category", "LastWorn"] # Bloqueamos edición de lo que no sea status
    )
    
    if st.button("Guardar Estados de Lavado"):
        # Actualizamos el DF principal con los cambios de estado
        df.update(edited_laundry)
        st.session_state['inventory'] = df
        save_data(df)
        st.success("¡Canasto de ropa actualizado!")

# --- TAB 3: INVENTARIO GENERAL (Edición total + Borrar) ---
with tab3:
    st.subheader("Gestión de Inventario (Admin)")
    st.warning("⚠️ Seleccioná la casilla a la izquierda de la fila y presioná 'Supr' (Delete) en tu teclado para borrar una prenda.")
    
    # num_rows="dynamic" PERMITE AGREGAR Y BORRAR FILAS
    edited_inventory = st.data_editor(
        df, 
        key="editor_inventario",
        num_rows="dynamic", # ¡ESTA ES LA CLAVE PARA BORRAR!
        hide_index=False
    )
    
    col_d1, col_d2 = st.columns([1, 4])
    with col_d1:
        if st.button("💾 Guardar Cambios en Inventario"):
            st.session_state['inventory'] = edited_inventory
            save_data(edited_inventory)
            st.success("Inventario guardado.")
    with col_d2:
        st.download_button("📥 Descargar CSV de Respaldo", df.to_csv(index=False).encode('utf-8'), "gdi_backup.csv")

# --- TAB 4: CARGA MANUAL ---
with tab4:
    st.subheader("Alta de Nueva Prenda (SNA)")
    col_a, col_b = st.columns(2)
    with col_a:
        c_temp = st.selectbox("Temporada", ["V", "W", "M"])
        c_type_full = st.selectbox("Tipo", ["R - Remera", "CS - Camisa", "P - Pantalón", "C - Campera", "B - Buzo"])
        
        type_map = {"R - Remera": "R", "CS - Camisa": "CS", "P - Pantalón": "P", "C - Campera": "C", "B - Buzo": "B"}
        type_code = type_map[c_type_full]
        category_name = c_type_full.split(" - ")[1]

        if type_code == "P":
            c_attr = st.selectbox("Tipo Pantalón", ["Je", "Sh", "DL", "DC", "Ve"])
        elif type_code in ["C", "B"]:
            c_attr_raw = st.selectbox("Nivel Abrigo", ["1 - Rompevientos", "2 - Fina", "3 - Común", "4 - Gruesa", "5 - Muy Gruesa"])
            c_attr = f"0{c_attr_raw[0]}"
        else: 
            c_attr_raw = st.selectbox("Manga", ["00 - Musculosa", "01 - Corta", "02 - Larga"])
            c_attr = c_attr_raw[:2]

    with col_b:
        c_occ_full = st.selectbox("Ocasión", ["U - Universidad", "D - Deporte", "C - Casa", "F - Formal"])
        c_occ = c_occ_full[0]
        c_col_full = st.selectbox("Color", ["01-Blanco", "02-Negro", "03-Rojo", "04-Azul", "05-Gris", "06-Verde", "07-Amarillo", "08-Marrón", "09-Estampado", "10-Denim"])
        c_col = c_col_full[:2]
        c_url = st.text_input("URL de la Foto (Opcional)")

    prefix = f"{c_temp}{type_code}{c_attr}{c_occ}{c_col}"
    # Contar existentes para ID
    count = len([c for c in df['Code'] if str(c).startswith(prefix)])
    new_id = f"{count + 1:02d}"
    final_code = f"{prefix}{new_id}"
    
    st.code(f"Código Generado: {final_code}")
    
    if st.button("Agregar al Armario"):
        new_row = pd.DataFrame([{
            'Code': final_code, 'Category': category_name, 'Season': c_temp, 'Occasion': c_occ, 
            'ImageURL': c_url, 'Status': 'Limpio', 'LastWorn': datetime.now().strftime("%Y-%m-%d")
        }])
        updated_df = pd.concat([df, new_row], ignore_index=True)
        st.session_state['inventory'] = updated_df
        save_data(updated_df)
        st.success(f"Prenda {final_code} guardada.")
