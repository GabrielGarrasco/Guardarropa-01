import streamlit as st
import pandas as pd
import requests
import os
from datetime import datetime

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="GDI: Mendoza Ops", layout="wide", page_icon="🧥")

FILE_INV = 'inventory.csv'
FILE_FEEDBACK = 'feedback.csv'

# --- FUNCIONES AUXILIARES SNA (INGENIERÍA) ---
def decodificar_sna(codigo):
    try:
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

def save_feedback_entry(entry):
    """Guarda la calificación del usuario para entrenar el sistema a futuro"""
    if not os.path.exists(FILE_FEEDBACK):
        df = pd.DataFrame(columns=['Date', 'Temp', 'Occasion', 'Top', 'Bottom', 'Outer', 'Rating_Abrigo', 'Rating_Comodidad', 'Rating_Seguridad'])
    else:
        df = pd.read_csv(FILE_FEEDBACK)
    
    new_row = pd.DataFrame([entry])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(FILE_FEEDBACK, index=False)

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

        # Lógica térmica
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

if 'inventory' not in st.session_state:
    st.session_state['inventory'] = load_data()
df = st.session_state['inventory']

weather = get_weather(api_key)

tab1, tab2, tab3, tab4 = st.tabs(["🔥 Sugerencia", "🧺 Lavadero", "📋 Inventario General", "➕ Carga Manual"])

# --- TAB 1: SUGERENCIA + FEEDBACK ---
with tab1:
    st.markdown(f"### 🌡️ Clima: {weather['temp']}°C (Sensación: {weather['temp']+3}°C)")
    st.caption(f"Condición: {weather['desc']}")
    
    recs_df = recommend_outfit(df, weather, code_occ)
    
    # Variables para guardar qué se recomendó
    rec_top = None
    rec_bot = None
    rec_out = None

    if not recs_df.empty:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Base**")
            base = recs_df[recs_df['Category'].isin(['Remera', 'Camisa'])]
            if not base.empty:
                item = base.sample(1).iloc[0]
                rec_top = item['Code'] # Guardamos el código
                st.info(f"{item['Category']} ({item['Code']})")
                if pd.notna(item['ImageURL']) and item['ImageURL']: st.image(item['ImageURL'])
        with col2:
            st.markdown("**Piernas**")
            legs = recs_df[recs_df['Category'] == 'Pantalón']
            if not legs.empty:
                item = legs.sample(1).iloc[0]
                rec_bot = item['Code']
                st.success(f"{item['Category']} ({item['Code']})")
                if pd.notna(item['ImageURL']) and item['ImageURL']: st.image(item['ImageURL'])
        with col3:
            st.markdown("**Abrigo**")
            outer = recs_df[recs_df['Category'].isin(['Campera', 'Buzo'])]
            if not outer.empty:
                item = outer.sample(1).iloc[0]
                rec_out = item['Code']
                st.warning(f"{item['Category']} ({item['Code']})")
                if pd.notna(item['ImageURL']) and item['ImageURL']: st.image(item['ImageURL'])
            else:
                rec_out = "N/A"
                st.write("No necesitás abrigo.")
        
        # --- SECCIÓN DE FEEDBACK ---
        st.divider()
        st.subheader("⭐ Calificalo para que aprenda")
        with st.form("feedback_form"):
            st.write("¿Qué tal funcionó esta combinación?")
            c1, c2, c3 = st.columns(3)
            with c1:
                r_abrigo = st.slider("🌡️ Nivel de Abrigo", 1, 5, 3, help="1: Me congelé/Asé - 5: Temperatura perfecta")
            with c2:
                r_comodidad = st.slider("😌 Nivel de Comodidad", 1, 5, 3)
            with c3:
                r_seguridad = st.slider("😎 Flow / Seguridad", 1, 5, 3)
            
            submit_feedback = st.form_submit_button("Guardar Calificación")
            
            if submit_feedback:
                entry = {
                    'Date': datetime.now().strftime("%Y-%m-%d %H:%M"),
                    'Temp': weather['temp'],
                    'Occasion': code_occ,
                    'Top': rec_top,
                    'Bottom': rec_bot,
                    'Outer': rec_out,
                    'Rating_Abrigo': r_abrigo,
                    'Rating_Comodidad': r_comodidad,
                    'Rating_Seguridad': r_seguridad
                }
                save_feedback_entry(entry)
                st.success("✅ ¡Datos guardados! El algoritmo será más inteligente mañana.")

    else:
        st.error("No hay ropa limpia para este clima.")

# --- TAB 2: LAVADERO ---
with tab2:
    st.subheader("Operaciones de Lavado")
    columnas_lavado = ['Code', 'Category', 'Status', 'LastWorn']
    edited_laundry = st.data_editor(
        df[columnas_lavado], 
        key="editor_lavadero",
        column_config={"Status": st.column_config.SelectboxColumn("Estado", options=["Limpio", "Sucio", "Lavando"], required=True)},
        hide_index=True,
        disabled=["Code", "Category", "LastWorn"]
    )
    if st.button("Guardar Estados de Lavado"):
        df.update(edited_laundry)
        st.session_state['inventory'] = df
        save_data(df)
        st.success("¡Canasto de ropa actualizado!")

# --- TAB 3: INVENTARIO GENERAL ---
with tab3:
    st.subheader("Gestión de Inventario (Admin)")
    edited_inventory = st.data_editor(df, key="editor_inventario", num_rows="dynamic", hide_index=False)
    if st.button("💾 Guardar Cambios"):
        st.session_state['inventory'] = edited_inventory
        save_data(edited_inventory)
        st.success("Inventario guardado.")
    st.download_button("📥 Descargar CSV", df.to_csv(index=False).encode('utf-8'), "gdi_backup.csv")

# --- TAB 4: CARGA MANUAL ---
with tab4:
    st.subheader("Alta de Nueva Prenda")
    col_a, col_b = st.columns(2)
    with col_a:
        c_temp = st.selectbox("Temporada", ["V", "W", "M"])
        c_type_full = st.selectbox("Tipo", ["R - Remera", "CS - Camisa", "P - Pantalón", "C - Campera", "B - Buzo"])
        type_map = {"R - Remera": "R", "CS - Camisa": "CS", "P - Pantalón": "P", "C - Campera": "C", "B - Buzo": "B"}
        type_code = type_map[c_type_full]
        category_name = c_type_full.split(" - ")[1]
        
        if type_code == "P": c_attr = st.selectbox("Tipo", ["Je", "Sh", "DL", "DC", "Ve"])
        elif type_code in ["C", "B"]: c_attr = f"0{st.selectbox('Abrigo', ['1', '2', '3', '4', '5'])}"
        else: c_attr = st.selectbox("Manga", ["00", "01", "02"])[:2]

    with col_b:
        c_occ = st.selectbox("Ocasión", ["U", "D", "C", "F"])
        c_col = st.selectbox("Color", ["01-Blanco", "02-Negro", "03-Rojo", "04-Azul", "10-Denim"])[:2]
        c_url = st.text_input("URL Foto")

    prefix = f"{c_temp}{type_code}{c_attr}{c_occ}{c_col}"
    count = len([c for c in df['Code'] if str(c).startswith(prefix)])
    final_code = f"{prefix}{count + 1:02d}"
    
    st.code(f"Código: {final_code}")
    if st.button("Agregar"):
        new_row = pd.DataFrame([{'Code': final_code, 'Category': category_name, 'Season': c_temp, 'Occasion': c_occ, 'ImageURL': c_url, 'Status': 'Limpio', 'LastWorn': datetime.now().strftime("%Y-%m-%d")}])
        updated_df = pd.concat([df, new_row], ignore_index=True)
        st.session_state['inventory'] = updated_df
        save_data(updated_df)
        st.success(f"Guardado: {final_code}")
