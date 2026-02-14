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
    Retorna un diccionario con los atributos.
    """
    try:
        season = codigo[0]
        
        # Detectar si es Camisa (CS) o el resto (1 letra)
        if codigo[1:3] == 'CS':
            tipo = 'CS'
            idx_start_attr = 3
        else:
            tipo = codigo[1]
            idx_start_attr = 2
            
        # El atributo (Nivel abrigo o Tipo Pantalon) son siempre 2 caracteres
        attr = codigo[idx_start_attr : idx_start_attr + 2]
        
        # Ocasión
        idx_occ = idx_start_attr + 2
        occasion = codigo[idx_occ]
        
        return {
            "tipo": tipo,
            "attr": attr, # Puede ser '04' (abrigo) o 'Je' (pantalón)
            "occasion": occasion
        }
    except:
        return None

def load_data():
    if not os.path.exists(FILE_INV):
        # Estructura inicial vacía
        return pd.DataFrame(columns=['Code', 'Category', 'Season', 'Occasion', 'ImageURL', 'Status', 'LastWorn'])
    df = pd.read_csv(FILE_INV)
    # Convertir a string para evitar problemas
    df['Code'] = df['Code'].astype(str) 
    return df

def save_data(df):
    df.to_csv(FILE_INV, index=False)

def get_weather(api_key):
    # Coordenadas Mendoza
    lat, lon = -32.8908, -68.8272 
    
    # MODO DEMO (Si no hay API Key)
    if not api_key:
        return {"temp": 24, "min": 18, "max": 30, "desc": "Modo Demo (Zonda)"} # Puse calor para probar shorts
        
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
    # AJUSTE PERSONAL: Usuario caluroso (+3°C de sensación)
    temp_percibida = temp_real + 3 
    
    recs = []
    
    for index, row in clean_df.iterrows():
        sna = decodificar_sna(row['Code'])
        if not sna: continue # Saltar códigos mal formados
        
        # Filtro de Ocasión
        if sna['occasion'] != occasion: continue

        # --- LÓGICA DE FILTRADO TÉRMICO ---
        
        # 1. PARTE INFERIOR (Pantalones)
        if row['Category'] == 'Pantalón':
            tipo_pan = sna['attr'] # Sh, Je, DL, etc.
            
            if temp_percibida > 26: # Mucho calor
                if tipo_pan in ['Sh', 'DC']: recs.append(row) # Short o Dep Corto
            elif temp_percibida > 20: # Calor moderado
                if tipo_pan in ['Je', 'Ve', 'DL', 'Sh']: recs.append(row) # Todo vale, prefiere fresco
            else: # Frío o templado (< 20)
                if tipo_pan in ['Je', 'Ve', 'DL']: recs.append(row) # Pantalón largo

        # 2. PARTE SUPERIOR (Remeras/Camisas)
        elif row['Category'] in ['Remera', 'Camisa']:
            manga = sna['attr'] # 00, 01, 02
            
            if temp_percibida > 25:
                if manga in ['00', '01']: recs.append(row) # Musculosa o corta
            elif temp_percibida < 15:
                if manga in ['02', '01']: recs.append(row) # Larga o corta (con abrigo arriba)
            else:
                recs.append(row) # Todo sirve en media estación

        # 3. ABRIGOS (Camperas/Buzos) - Lógica de Capas
        elif row['Category'] in ['Campera', 'Buzo']:
            nivel_abrigo = int(sna['attr']) # 1 a 5
            
            if temp_percibida < 12: # Frío fuerte
                if nivel_abrigo >= 4: recs.append(row)
            elif temp_percibida < 18: # Fresco
                if nivel_abrigo in [2, 3]: recs.append(row)
            elif temp_percibida < 22: # Ventoso / Noche de verano
                if nivel_abrigo == 1: recs.append(row)
            # Si hace más de 22 (percibido), no recomienda abrigo.

    return pd.DataFrame(recs)

# --- INTERFAZ DE USUARIO ---
st.sidebar.title("🛠️ GDI: Mendoza Ops")
api_key = st.sidebar.text_input("API Key (OpenWeather)", type="password")
user_occ = st.sidebar.selectbox("Ocasión Actual", ["U (Universidad)", "D (Deporte)", "C (Casa)", "F (Formal)"])
code_occ = user_occ[0] # Extraer solo la letra

# Cargar datos
if 'inventory' not in st.session_state:
    st.session_state['inventory'] = load_data()
df = st.session_state['inventory']

weather = get_weather(api_key)

tab1, tab2, tab3 = st.tabs(["🔥 Sugerencia del Día", "🧺 Inventario & Lavado", "➕ Carga Manual"])

with tab1:
    st.markdown(f"### 🌡️ Clima: {weather['temp']}°C (Sensación tuya: {weather['temp']+3}°C)")
    st.caption(f"Condición: {weather['desc']}")
    
    if (weather['max'] - weather['min']) > 15:
        st.warning("⚠️ ¡Alerta Zonda/Amplitud! Llevá capas (cebolla).")

    recs_df = recommend_outfit(df, weather, code_occ)
    
    if not recs_df.empty:
        col1, col2, col3 = st.columns(3)
        
        # Mostrar sugerencias por capa
        with col1:
            st.markdown("**Base**")
            base = recs_df[recs_df['Category'].isin(['Remera', 'Camisa'])]
            if not base.empty:
                item = base.sample(1).iloc[0]
                st.info(f"{item['Category']} ({item['Code']})")
                if pd.notna(item['ImageURL']) and item['ImageURL']: st.image(item['ImageURL'])
            else:
                st.write("No hay prendas base limpias para este clima.")

        with col2:
            st.markdown("**Piernas**")
            legs = recs_df[recs_df['Category'] == 'Pantalón']
            if not legs.empty:
                item = legs.sample(1).iloc[0]
                st.success(f"{item['Category']} ({item['Code']})")
                if pd.notna(item['ImageURL']) and item['ImageURL']: st.image(item['ImageURL'])
            else:
                st.write("No hay pantalones limpios aptos.")

        with col3:
            st.markdown("**Abrigo**")
            outer = recs_df[recs_df['Category'].isin(['Campera', 'Buzo'])]
            if not outer.empty:
                item = outer.sample(1).iloc[0]
                st.warning(f"{item['Category']} ({item['Code']})")
                if pd.notna(item['ImageURL']) and item['ImageURL']: st.image(item['ImageURL'])
            else:
                st.write("¡No necesitás abrigo o no hay limpios!")
    else:
        st.error("No encontré ropa limpia que coincida con el clima y la ocasión. ¡A lavar!")

with tab2:
    st.subheader("Gestión de Lavadero")
    # Data Editor permite editar el estado directamente
    edited_df = st.data_editor(
        df, 
        column_config={
            "ImageURL": st.column_config.LinkColumn("Foto"),
            "Status": st.column_config.SelectboxColumn("Estado", options=["Limpio", "Sucio", "Lavando"])
        },
        hide_index=True
    )
    
    if st.button("Guardar Cambios de Estado"):
        st.session_state['inventory'] = edited_df
        save_data(edited_df)
        st.success("Inventario actualizado.")

    st.download_button("📥 Descargar Backup (CSV)", df.to_csv(index=False).encode('utf-8'), "gdi_backup.csv")

with tab3:
    st.subheader("Alta de Nueva Prenda (SNA)")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        c_temp = st.selectbox("Temporada", ["V", "W", "M"])
        c_type_full = st.selectbox("Tipo", ["R - Remera", "CS - Camisa", "P - Pantalón", "C - Campera", "B - Buzo"])
        
        # Mapeo de códigos
        type_map = {"R - Remera": "R", "CS - Camisa": "CS", "P - Pantalón": "P", "C - Campera": "C", "B - Buzo": "B"}
        type_code = type_map[c_type_full]
        category_name = c_type_full.split(" - ")[1]

        # Lógica dinámica de Atributos
        if type_code == "P":
            c_attr = st.selectbox("Tipo Pantalón", ["Je", "Sh", "DL", "DC", "Ve"])
        elif type_code in ["C", "B"]:
            c_attr_raw = st.selectbox("Nivel Abrigo", ["1 - Rompevientos", "2 - Fina", "3 - Común", "4 - Gruesa", "5 - Muy Gruesa"])
            c_attr = f"0{c_attr_raw[0]}" # Extrae el número y le pone un 0 delante
        else: # R o CS
            c_attr_raw = st.selectbox("Manga", ["00 - Musculosa", "01 - Corta", "02 - Larga"])
            c_attr = c_attr_raw[:2]

    with col_b:
        c_occ_full = st.selectbox("Ocasión", ["U - Universidad", "D - Deporte", "C - Casa", "F - Formal"])
        c_occ = c_occ_full[0]
        
        c_col_full = st.selectbox("Color", ["01-Blanco", "02-Negro", "03-Rojo", "04-Azul", "05-Gris", "06-Verde", "07-Amarillo", "08-Marrón", "09-Estampado", "10-Denim"])
        c_col = c_col_full[:2]
        
        c_url = st.text_input("URL de la Foto (Opcional)")

    # Generación de ID automática (simple)
    # Busca cuántas prendas iguales existen para asignar ID
    prefix = f"{c_temp}{type_code}{c_attr}{c_occ}{c_col}"
    existing_count = len([c for c in df['Code'] if c.startswith(prefix)])
    new_id = f"{existing_count + 1:02d}" # Formato 01, 02...
    
    final_code = f"{prefix}{new_id}"
    
    st.success(f"Código Generado: **{final_code}**")
    
    if st.button("Agregar al Armario"):
        new_row = pd.DataFrame([{
            'Code': final_code, 
            'Category': category_name, 
            'Season': c_temp, 
            'Occasion': c_occ, 
            'ImageURL': c_url, 
            'Status': 'Limpio', 
            'LastWorn': datetime.now().strftime("%Y-%m-%d")
        }])
        
        updated_df = pd.concat([df, new_row], ignore_index=True)
        st.session_state['inventory'] = updated_df
        save_data(updated_df)
        st.balloons()
