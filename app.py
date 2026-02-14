import streamlit as st
import pandas as pd
import requests
import os
from datetime import datetime

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="GDI: Mendoza Ops", layout="wide", page_icon="🧥")

FILE_INV = 'inventory.csv'
FILE_FEEDBACK = 'feedback.csv'

# Colores Completos (Restaurados)
COLORS_LIST = [
    "01-Blanco", "02-Negro", "03-Rojo", "04-Azul", "05-Gris", 
    "06-Verde", "07-Amarillo/Naranja", "08-Marrón/Beige", "09-Estampado", "10-Denim"
]

# Límites de Uso
MAX_USOS = {
    'Remera': 2, 'Camisa': 2, 'Pantalón': 6, 
    'Buzo': 5, 'Campera': 10, 'Short': 1    
}

# --- FUNCIONES ---
def decodificar_sna(codigo):
    try:
        if len(codigo) > 2 and codigo[1:3] == 'CS':
            tipo = 'CS'; idx_start_attr = 3
        else:
            tipo = codigo[1]; idx_start_attr = 2
        return {"tipo": tipo, "attr": codigo[idx_start_attr:idx_start_attr+2], "occasion": codigo[idx_start_attr+2]}
    except: return None

def load_data():
    if not os.path.exists(FILE_INV):
        return pd.DataFrame(columns=['Code', 'Category', 'Season', 'Occasion', 'ImageURL', 'Status', 'LastWorn', 'Uses'])
    try:
        df = pd.read_csv(FILE_INV)
        df['Code'] = df['Code'].astype(str)
        if 'Uses' not in df.columns: df['Uses'] = 0
        return df
    except: return pd.DataFrame(columns=['Code', 'Category', 'Season', 'Occasion', 'ImageURL', 'Status', 'LastWorn', 'Uses'])

def save_data(df):
    df.to_csv(FILE_INV, index=False)

def save_feedback_entry(entry):
    if not os.path.exists(FILE_FEEDBACK):
        df = pd.DataFrame(columns=['Date', 'City', 'Temp_Real', 'Feels_Like', 'User_Adj', 'Top', 'Bottom', 'Outer', 'R_Abrigo', 'R_Comodidad', 'R_Seguridad'])
    else:
        df = pd.read_csv(FILE_FEEDBACK)
    df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    df.to_csv(FILE_FEEDBACK, index=False)

def get_weather(api_key, city="Mendoza, AR"):
    if not city: city = "Mendoza, AR"
    if not api_key: return {"temp": 24, "feels_like": 22, "min": 18, "max": 30, "desc": "Modo Demo"}
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric&lang=es"
        res = requests.get(url).json()
        if res.get("cod") != 200: return {"temp": 0, "feels_like": 0, "desc": "Error City", "min":0, "max":0}
        return {
            "temp": res['main']['temp'], "feels_like": res['main']['feels_like'],
            "min": res['main']['temp_min'], "max": res['main']['temp_max'],
            "desc": res['weather'][0]['description'].capitalize()
        }
    except: return {"temp": 15, "feels_like": 14, "desc": "Error API", "min":10, "max":20}

# --- LÓGICA RECOMENDACIÓN ---
def recommend_outfit(df, weather, occasion):
    clean_df = df[df['Status'] == 'Limpio'].copy()
    if clean_df.empty: return pd.DataFrame(), 0

    st_meteo = weather.get('feels_like', weather['temp'])
    temp_decision = st_meteo + 3 
    
    recs = []
    for index, row in clean_df.iterrows():
        sna = decodificar_sna(row['Code'])
        if not sna or sna['occasion'] != occasion: continue

        matches = False
        if row['Category'] == 'Pantalón':
            tipo = sna['attr']
            if temp_decision > 26 and tipo in ['Sh', 'DC']: matches = True
            elif temp_decision > 20 and tipo in ['Je', 'Ve', 'DL', 'Sh']: matches = True
            elif temp_decision <= 20 and tipo in ['Je', 'Ve', 'DL']: matches = True
        elif row['Category'] in ['Remera', 'Camisa']:
            manga = sna['attr']
            if temp_decision > 25 and manga in ['00', '01']: matches = True
            elif temp_decision < 15 and manga in ['02', '01']: matches = True
            else: matches = True
        elif row['Category'] in ['Campera', 'Buzo']:
            nivel = int(sna['attr'])
            if temp_decision < 12 and nivel >= 4: matches = True
            elif temp_decision < 18 and nivel in [2, 3]: matches = True
            elif temp_decision < 22 and nivel == 1: matches = True
        
        if matches: recs.append(row)

    res_df = pd.DataFrame(recs)
    if not res_df.empty: res_df = res_df.sort_values(by='Uses', ascending=False)
    return res_df, temp_decision

# --- INTERFAZ ---
st.sidebar.title("🛠️ GDI: Mendoza Ops v7")
api_key = st.sidebar.text_input("API Key", type="password")
user_city = st.sidebar.text_input("Ciudad", value="Mendoza, AR")
user_occ = st.sidebar.selectbox("Ocasión", ["U (Universidad)", "D (Deporte)", "C (Casa)", "F (Formal)"])
code_occ = user_occ[0]

if 'inventory' not in st.session_state: st.session_state['inventory'] = load_data()
df = st.session_state['inventory']
weather = get_weather(api_key, user_city)

tab1, tab2, tab3, tab4 = st.tabs(["🔥 Sugerencia & Feedback", "🧺 Lavadero", "📋 Inventario", "➕ Carga"])

# --- TAB 1: SUGERENCIA (ESTÉTICA v5 RESTAURADA) ---
with tab1:
    recs_df, temp_calc = recommend_outfit(df, weather, code_occ)
    
    # Métricas Visuales (Estilo v5)
    col_w1, col_w2, col_w3 = st.columns(3)
    col_w1.metric("🌡️ Temp Aire", f"{weather['temp']}°C")
    col_w2.metric("💨 ST Real", f"{weather['feels_like']}°C")
    col_w3.metric("🤖 Tu Ajuste (+3)", f"{temp_calc:.1f}°C", delta="Perfil Caluroso")
    st.caption(f"Condición: {weather['desc']}")

    if (weather['max'] - weather['min']) > 15: st.warning("⚠️ Alerta Zonda/Amplitud! Usá capas.")

    current_items = {}
    
    if not recs_df.empty:
        st.divider()
        col1, col2, col3 = st.columns(3)
        
        # Mostrar Sugerencias (Con chequeo seguro de imagen)
        base = recs_df[recs_df['Category'].isin(['Remera', 'Camisa'])]
        if not base.empty:
            item = base.iloc[0]
            current_items['Top'] = item
            with col1:
                st.info(f"Base: {item['Category']}")
                if pd.notna(item['ImageURL']) and item['ImageURL']: st.image(item['ImageURL'])
                st.caption(f"Usos: {item['Uses']}/{MAX_USOS.get(item['Category'], 2)}")

        legs = recs_df[recs_df['Category'] == 'Pantalón']
        if not legs.empty:
            item = legs.iloc[0]
            current_items['Bottom'] = item
            with col2:
                st.success(f"Piernas: {item['Category']}")
                if pd.notna(item['ImageURL']) and item['ImageURL']: st.image(item['ImageURL'])
                st.caption(f"Usos: {item['Uses']}/{MAX_USOS.get(item['Category'], 5)}")

        outer = recs_df[recs_df['Category'].isin(['Campera', 'Buzo'])]
        if not outer.empty:
            item = outer.iloc[0]
            current_items['Outer'] = item
            with col3:
                st.warning(f"Abrigo: {item['Category']}")
                if pd.notna(item['ImageURL']) and item['ImageURL']: st.image(item['ImageURL'])
                st.caption(f"Usos: {item['Uses']}/{MAX_USOS.get(item['Category'], 5)}")
        
        st.divider()

        # --- FLUJO DE CONFIRMACIÓN + ESTRELLAS ---
        # Si NO hemos confirmado todavía, mostramos botones de uso
        if 'last_confirmed_date' not in st.session_state or st.session_state['last_confirmed_date'] != datetime.now().strftime("%Y-%m-%d %H"):
            st.subheader("¿Vas a usar esta combinación?")
            c_yes, c_no = st.columns(2)
            if c_yes.button("✅ SÍ, usar esto"):
                # Registrar Uso
                for k, item in current_items.items():
                    idx = df[df['Code'] == item['Code']].index[0]
                    df.at[idx, 'Uses'] += 1
                    df.at[idx, 'LastWorn'] = datetime.now().strftime("%Y-%m-%d")
                save_data(df)
                st.session_state['last_confirmed_date'] = datetime.now().strftime("%Y-%m-%d %H")
                st.session_state['confirmed_items'] = current_items # Guardamos para el feedback
                st.rerun()
                
            if c_no.button("❌ NO, usé otra cosa"):
                st.session_state['manual_mode'] = True

        # MODO MANUAL
        if st.session_state.get('manual_mode'):
            st.warning("Modo Manual: Ingresá el código de lo que te pusiste.")
            m_code = st.text_input("Código Prenda:")
            if st.button("Confirmar Manual") and m_code:
                found = df[df['Code'] == m_code]
                if not found.empty:
                    idx = found.index[0]
                    df.at[idx, 'Uses'] += 1
                    df.at[idx, 'LastWorn'] = datetime.now().strftime("%Y-%m-%d")
                    save_data(df)
                    st.success(f"Uso registrado para {m_code}")
                    st.session_state['manual_mode'] = False
                    st.session_state['last_confirmed_date'] = datetime.now().strftime("%Y-%m-%d %H")
                    st.session_state['confirmed_items'] = {'Manual': found.iloc[0]}
                    st.rerun()
                else: st.error("Código no existe")

        # --- SECCIÓN DE ESTRELLAS (Solo aparece si ya confirmamos uso) ---
        if 'last_confirmed_date' in st.session_state and st.session_state['last_confirmed_date'] == datetime.now().strftime("%Y-%m-%d %H"):
            st.success("✅ ¡Uso registrado! Ahora calificá la experiencia:")
            
            with st.form("star_feedback"):
                c1, c2, c3 = st.columns(3)
                r_abrigo = c1.slider("🌡️ Nivel de Abrigo", 1, 5, 3)
                r_comodidad = c2.slider("😌 Comodidad", 1, 5, 3)
                r_seguridad = c3.slider("😎 Flow/Seguridad", 1, 5, 3)
                
                if st.form_submit_button("Guardar Calificación ⭐"):
                    # Guardar Feedback
                    items_code = {k: v['Code'] for k, v in st.session_state.get('confirmed_items', {}).items()}
                    entry = {
                        'Date': datetime.now().strftime("%Y-%m-%d %H:%M"),
                        'City': user_city,
                        'Temp_Real': weather['temp'],
                        'Feels_Like': weather['feels_like'],
                        'User_Adj': temp_calc,
                        'Top': items_code.get('Top'),
                        'Bottom': items_code.get('Bottom'),
                        'Outer
