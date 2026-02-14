import streamlit as st
import pandas as pd
import requests
import os
from datetime import datetime

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="GDI: Mendoza Ops", layout="wide", page_icon="🧥")

FILE_INV = 'inventory.csv'
FILE_FEEDBACK = 'feedback.csv'

# Definición de Límites de Uso
MAX_USOS = {
    'Remera': 2,
    'Camisa': 2,
    'Pantalón': 6, 
    'Buzo': 5,
    'Campera': 10, 
    'Short': 1,    
}

# --- FUNCIONES AUXILIARES ---
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
        return pd.DataFrame(columns=['Code', 'Category', 'Season', 'Occasion', 'ImageURL', 'Status', 'LastWorn', 'Uses'])
    
    try:
        df = pd.read_csv(FILE_INV)
        df['Code'] = df['Code'].astype(str)
        # Migración automática segura
        if 'Uses' not in df.columns:
            df['Uses'] = 0
        return df
    except Exception as e:
        st.error(f"Error leyendo el archivo: {e}")
        return pd.DataFrame(columns=['Code', 'Category', 'Season', 'Occasion', 'ImageURL', 'Status', 'LastWorn', 'Uses'])

def save_data(df):
    df.to_csv(FILE_INV, index=False)

def get_weather(api_key, city="Mendoza, AR"):
    if not city: city = "Mendoza, AR"
    if not api_key: return {"temp": 24, "feels_like": 22, "min": 18, "max": 30, "desc": "Modo Demo"}
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric&lang=es"
        res = requests.get(url).json()
        if res.get("cod") != 200: return {"temp": 0, "feels_like": 0, "desc": "Error City", "min":0, "max":0}
        return {
            "temp": res['main']['temp'],
            "feels_like": res['main']['feels_like'],
            "min": res['main']['temp_min'],
            "max": res['main']['temp_max'],
            "desc": res['weather'][0]['description'].capitalize()
        }
    except: return {"temp": 15, "feels_like": 14, "desc": "Error API", "min":10, "max":20}

# --- LÓGICA DE RECOMENDACIÓN ---
def recommend_outfit(df, weather, occasion):
    clean_df = df[df['Status'] == 'Limpio'].copy()
    if clean_df.empty: return pd.DataFrame(), 0

    st_meteo = weather.get('feels_like', weather['temp'])
    temp_decision = st_meteo + 3 
    
    recs = []
    
    for index, row in clean_df.iterrows():
        sna = decodificar_sna(row['Code'])
        if not sna: continue
        if sna['occasion'] != occasion: continue

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
        
        if matches:
            recs.append(row)

    res_df = pd.DataFrame(recs)
    if not res_df.empty:
        res_df = res_df.sort_values(by='Uses', ascending=False)
        
    return res_df, temp_decision

# --- REGISTRAR USO ---
def registrar_uso(code, df):
    idx = df[df['Code'] == code].index
    if len(idx) == 0: return "Error: Código no encontrado", df, False

    idx = idx[0]
    cat = df.at[idx, 'Category']
    current_uses = df.at[idx, 'Uses'] + 1 
    
    df.at[idx, 'Uses'] = current_uses
    df.at[idx, 'LastWorn'] = datetime.now().strftime("%Y-%m-%d")
    
    limit = MAX_USOS.get(cat, 1)
    
    msg = ""
    needs_wash = False
    
    if cat in ['Short', 'Deportivo']:
        msg = f"Usaste {cat} (Total: {current_uses}). ¿Transpiraste? ¿A lavar?"
        needs_wash = "ASK"
    elif current_uses >= limit:
        msg = f"¡Atención! {cat} llegó a {current_uses} usos. Se recomienda LAVAR."
        needs_wash = True
    else:
        msg = f"{cat} tiene {current_uses} usos. (Límite sugerido: {limit}). ¿Seguís usándolo?"
        needs_wash = "ASK"
        
    return msg, df, needs_wash

# --- INTERFAZ ---
st.sidebar.title("🛠️ GDI: Mendoza Ops v6.1")
api_key = st.sidebar.text_input("API Key", type="password")
user_city = st.sidebar.text_input("Ciudad", value="Mendoza, AR")
user_occ = st.sidebar.selectbox("Ocasión", ["U (Universidad)", "D (Deporte)", "C (Casa)", "F (Formal)"])
code_occ = user_occ[0]

if 'inventory' not in st.session_state: st.session_state['inventory'] = load_data()
df = st.session_state['inventory']
weather = get_weather(api_key, user_city)

tab1, tab2, tab3, tab4 = st.tabs(["🔥 Sugerencia & Uso", "🧺 Lavadero", "📋 Inventario", "➕ Carga"])

# --- TAB 1: SUGERENCIA ---
with tab1:
    recs_df, temp_calc = recommend_outfit(df, weather, code_occ)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("🌡️ Temp", f"{weather['temp']}°C")
    c2.metric("🤖 Tu Ajuste", f"{temp_calc:.1f}°C")
    c3.caption(weather['desc'])
    
    current_suggestion = {}

    if not recs_df.empty:
        st.divider()
        st.subheader("Tu Outfit Recomendado (Prioridad: Usado)")
        
        col1, col2, col3 = st.columns(3)
        
        # Base
        base = recs_df[recs_df['Category'].isin(['Remera', 'Camisa'])]
        if not base.empty:
            item = base.iloc[0]
            current_suggestion['Top'] = item
            with col1:
                st.info(f"Base: {item['Category']}")
                st.caption(f"Usos actuales: {item['Uses']}")
                # CORRECCIÓN DE VISUALIZACIÓN
                if pd.notna(item['ImageURL']) and item['ImageURL']:
                    st.image(item['ImageURL'])
                else:
                    st.write(f"Código: {item['Code']}")

        # Piernas
        legs = recs_df[recs_df['Category'] == 'Pantalón']
        if not legs.empty:
            item = legs.iloc[0]
            current_suggestion['Bottom'] = item
            with col2:
                st.success(f"Piernas: {item['Category']}")
                st.caption(f"Usos actuales: {item['Uses']}")
                # CORRECCIÓN DE VISUALIZACIÓN
                if pd.notna(item['ImageURL']) and item['ImageURL']:
                    st.image(item['ImageURL'])
                else:
                    st.write(f"Código: {item['Code']}")

        # Abrigo
        outer = recs_df[recs_df['Category'].isin(['Campera', 'Buzo'])]
        if not outer.empty:
            item = outer.iloc[0]
            current_suggestion['Outer'] = item
            with col3:
                st.warning(f"Abrigo: {item['Category']}")
                st.caption(f"Usos actuales: {item['Uses']}")
                # CORRECCIÓN DE VISUALIZACIÓN
                if pd.notna(item['ImageURL']) and item['ImageURL']:
                    st.image(item['ImageURL'])
                else:
                    st.write(f"Código: {item['Code']}")
        
        st.divider()
        st.write("### ¿Vas a usar esta sugerencia?")
        col_yes, col_no = st.columns(2)
        
        with col_yes:
            if st.button("✅ SÍ, voy a usar esto"):
                st.session_state['confirm_mode'] = 'YES'
                st.session_state['selected_items'] = current_suggestion
        
        with col_no:
            if st.button("❌ NO, usé otra cosa"):
                st.session_state['confirm_mode'] = 'NO'

        # LÓGICA POST-BOTÓN
        if st.session_state.get('confirm_mode') == 'YES':
            items = st.session_state['selected_items']
            st.success("Procesando prendas sugeridas...")
            
            for key, item in items.items():
                msg, df, decision = registrar_uso(item['Code'], df)
                st.write(f"**{item['Category']} ({item['Code']}):** {msg}")
                
                key_w = f"wash_{item['Code']}"
                
                if decision == True: 
                    st.error(f"⚠️ {item['Category']} debería lavarse.")
                    if st.button(f"Mandar {item['Category']} a Lavar", key=key_w):
                        df.loc[df['Code'] == item['Code'], 'Status'] = 'Sucio'
                        df.loc[df['Code'] == item['Code'], 'Uses'] = 0
                        save_data(df)
                        st.rerun()
                else: 
                    action = st.radio(f"Estado de {item['Category']}", ["Seguir Usando", "Mandar a Lavar"], key=key_w, horizontal=True)
                    if st.button(f"Confirmar {item['Category']}", key=f"conf_{item['Code']}"):
                        if action == "Mandar a Lavar":
                            df.loc[df['Code'] == item['Code'], 'Status'] = 'Sucio'
                            df.loc[df['Code'] == item['Code'], 'Uses'] = 0
                        save_data(df)
                        st.success("Actualizado.")
                        st.rerun()

        elif st.session_state.get('confirm_mode') == 'NO':
            st.markdown("#### ¿Qué te pusiste entonces?")
            manual_code = st.text_input("Ingresá el Código de la prenda que usaste:")
            
            found_item = df[df['Code'] == manual_code]
            if not found_item.empty:
                row = found_item.iloc[0]
                if pd.notna(row['ImageURL']) and row['ImageURL']:
                    st.image(row['ImageURL'], width=150)
                
                st.info(f"Prenda: {row['Category']} | Usos actuales: {row['Uses']}")
                
                if st.button("Confirmar que usé esto"):
                    msg, df, decision = registrar_uso(manual_code, df)
                    st.write(msg)
                    action_man = st.radio("¿Estado?", ["Seguir Usando", "A Lavar"], horizontal=True)
                    if st.button("Finalizar Manual"):
                        if action_man == "A Lavar":
                            df.loc[df['Code'] == manual_code, 'Status'] = 'Sucio'
                            df.loc[df['Code'] == manual_code, 'Uses'] = 0
                        save_data(df)
                        st.session_state['confirm_mode'] = None 
                        st.rerun()
            elif manual_code:
