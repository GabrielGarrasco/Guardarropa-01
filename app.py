import streamlit as st
import pandas as pd
import requests
import os
from datetime import datetime

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="GDI: Mendoza Ops v8.3", layout="centered", page_icon="🧥")

FILE_INV = 'inventory.csv'
FILE_FEEDBACK = 'feedback.csv'

# --- LÍMITES DE USO (INGENIERÍA DE CICLO DE VIDA) ---
LIMITES_USO = {
    "Je": 6, "Ve": 4, "DL": 3, "DC": 2, "Sh": 1,
    "R": 2, "CS": 3,
    "B": 5, "C": 10
}

# --- FUNCIONES AUXILIARES ---
def decodificar_sna(codigo):
    try:
        codigo = str(codigo).strip()
        if len(codigo) < 4: return None
        if codigo[1:3] == 'CS':
            tipo = 'CS'; idx_start_attr = 3
        else:
            tipo = codigo[1]; idx_start_attr = 2
        attr = codigo[idx_start_attr : idx_start_attr + 2]
        idx_occ = codigo[idx_start_attr + 2] if len(codigo) > idx_start_attr + 2 else "C"
        return {"tipo": tipo, "attr": attr, "occasion": occasion}
    except: return None

def load_data():
    if not os.path.exists(FILE_INV):
        return pd.DataFrame(columns=['Code', 'Category', 'Season', 'Occasion', 'ImageURL', 'Status', 'LastWorn', 'Uses'])
    df = pd.read_csv(FILE_INV)
    df['Code'] = df['Code'].astype(str)
    if 'Uses' not in df.columns: df['Uses'] = 0
    return df

def save_data(df): df.to_csv(FILE_INV, index=False)

def save_feedback_entry(entry):
    if not os.path.exists(FILE_FEEDBACK):
        df = pd.DataFrame(columns=['Date', 'City', 'Temp_Real', 'Feels_Like', 'User_Adj_Temp', 'Occasion', 'Top', 'Bottom', 'Outer', 'Rating_Abrigo', 'Rating_Comodidad', 'Rating_Seguridad', 'Action'])
    else:
        df = pd.read_csv(FILE_FEEDBACK)
        # Migración rápida si falta columna Action
        if 'Action' not in df.columns: df['Action'] = 'Confirm'
    
    new_row = pd.DataFrame([entry])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(FILE_FEEDBACK, index=False)

def get_weather(api_key, city):
    if not api_key: return {"temp": 24, "feels_like": 22, "min": 18, "max": 30, "desc": "Modo Demo"}
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric&lang=es"
        res = requests.get(url).json()
        if res.get("cod") != 200: return {"temp": 0, "feels_like": 0, "min": 0, "max": 0, "desc": "Error API"}
        return {
            "temp": res['main']['temp'],
            "feels_like": res['main']['feels_like'],
            "min": res['main']['temp_min'], 
            "max": res['main']['temp_max'], 
            "desc": res['weather'][0]['description'].capitalize()
        }
    except: return {"temp": 15, "feels_like": 14, "desc": "Error Conexión"}

# --- LÓGICA DE RECOMENDACIÓN CON SEMILLA ---
def recommend_outfit(df, weather, occasion, seed):
    clean_df = df[df['Status'] == 'Limpio'].copy()
    if clean_df.empty: return pd.DataFrame(), 0

    st_meteo = weather.get('feels_like', weather['temp'])
    temp_decision = st_meteo + 3 
    
    recs = []
    for index, row in clean_df.iterrows():
        sna = decodificar_sna(row['Code'])
        if not sna: continue
        if sna['occasion'] != occasion: continue

        # Reglas
        match = False
        if row['Category'] == 'Pantalón':
            t = sna['attr']
            if temp_decision > 26 and t in ['Sh', 'DC']: match = True
            elif temp_decision > 20 and t in ['Je', 'Ve', 'DL', 'Sh']: match = True
            elif temp_decision <= 20 and t in ['Je', 'Ve', 'DL']: match = True
        elif row['Category'] in ['Remera', 'Camisa']:
            m = sna['attr']
            if temp_decision > 25 and m in ['00', '01']: match = True
            elif temp_decision < 15 and m in ['02', '01']: match = True
            else: match = True
        elif row['Category'] in ['Campera', 'Buzo']:
            try:
                n = int(sna['attr'])
                if temp_decision < 12 and n >= 4: match = True
                elif temp_decision < 18 and n in [2, 3]: match = True
                elif temp_decision < 24 and n == 1: match = True
            except: pass
        
        if match: recs.append(row)

    # AQUÍ ESTÁ LA MAGIA DE LA ROTACIÓN: Usamos la 'seed' para el sampleo
    return pd.DataFrame(recs), temp_decision

# --- INTERFAZ ---
st.sidebar.title("GDI: Mendoza Ops")
st.sidebar.markdown("---")
api_key = st.sidebar.text_input("🔑 API Key", type="password")
user_city = st.sidebar.text_input("📍 Ciudad", value="Mendoza, AR")
user_occ = st.sidebar.selectbox("🎯 Ocasión", ["U (Universidad)", "D (Deporte)", "C (Casa)", "F (Formal)"])
code_occ = user_occ[0]

# --- SESSION STATE ---
if 'inventory' not in st.session_state: st.session_state['inventory'] = load_data()
if 'seed' not in st.session_state: st.session_state['seed'] = 42 # Semilla inicial
if 'change_mode' not in st.session_state: st.session_state['change_mode'] = False # Controla la visualización de la encuesta
if 'confirm_stage' not in st.session_state: st.session_state['confirm_stage'] = 0 
if 'alerts_buffer' not in st.session_state: st.session_state['alerts_buffer'] = []

df = st.session_state['inventory']
weather = get_weather(api_key, user_city)

tab1, tab2, tab3, tab4 = st.tabs(["✨ Sugerencia", "🧺 Lavadero", "📦 Inventario", "➕ Nuevo Item"])

with tab1:
    recs_df, temp_calculada = recommend_outfit(df, weather, code_occ, st.session_state['seed'])

    # HEADER CLIMA
    with st.container(border=True):
        col_w1, col_w2, col_w3 = st.columns(3)
        col_w1.metric("Clima", f"{weather['temp']}°C", weather['desc'])
        col_w2.metric("Sensación", f"{weather['feels_like']}°C")
        col_w3.metric("Tu Perfil", f"{temp_calculada:.1f}°C", "+3°C adj")

    # HEADER OUTFIT + BOTÓN CAMBIAR
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.subheader("Outfit Recomendado")
    with col_h2:
        # Si tocan este botón, activamos el modo cambio
        if st.button("🔄 Cambiar", help="Ver otra opción"):
            st.session_state['change_mode'] = not st.session_state['change_mode']

    # VARIABLES PARA SELECCIÓN
    rec_top, rec_bot, rec_out = None, None, None
    selected_items_codes = []

    if not recs_df.empty:
        # LOGICA DE SELECCIÓN CON SEMILLA (Reproducibilidad hasta que se cambie la seed)
        base = recs_df[recs_df['Category'].isin(['Remera', 'Camisa'])]
        legs = recs_df[recs_df['Category'] == 'Pantalón']
        outer = recs_df[recs_df['Category'].isin(['Campera', 'Buzo'])]

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("###### Torso")
            if not base.empty:
                item = base.sample(1, random_state=st.session_state['seed']).iloc[0]
                rec_top = item['Code']; selected_items_codes.append(item)
                st.info(f"**{item['Category']}**\n\nCode: `{item['Code']}`")
                if pd.notna(item['ImageURL']) and item['ImageURL']: st.image(item['ImageURL'], use_column_width=True)
            else: st.write("🚫 Sin opciones")

        with c2:
            st.markdown("###### Piernas")
            if not legs.empty:
                item = legs.sample(1, random_state=st.session_state['seed']).iloc[0]
                rec_bot = item['Code']; selected_items_codes.append(item)
                st.success(f"**{item['Category']}**\n\nCode: `{item['Code']}`")
                if pd.notna(item['ImageURL']) and item['ImageURL']: st.image(item['ImageURL'], use_column_width=True)
            else: st.write("🚫 Sin opciones")

        with c3:
            st.markdown("###### Abrigo")
            if not outer.empty:
                item = outer.sample(1, random_state=st.session_state['seed']).iloc[0]
                rec_out = item['Code']; selected_items_codes.append(item)
                st.warning(f"**{item['Category']}**\n\nCode: `{item['Code']}`")
                if pd.notna(item['ImageURL']) and item['ImageURL']: st.image(item['ImageURL'], use_column_width=True)
            else:
                st.write("✅ No necesario"); rec_out = "N/A"

        st.divider()

        # --- FLUJO 1: CAMBIAR SUGERENCIA (ENCUESTA DE RECHAZO) ---
        if st.session_state['change_mode']:
            st.info("Ayudame a mejorar para la próxima.")
            with st.container(border=True):
                st.markdown("##### 📝 ¿Qué aspecto no te convenció?")
                
                cf1, cf2, cf3 = st.columns(3)
                with cf1: st.write("🌡️ Abrigo"); n_abr = st.feedback("stars", key="neg_abr")
                with cf2: st.write("😌 Comodidad"); n_com = st.feedback("stars", key="neg_com")
                with cf3: st.write("😎 Estilo"); n_seg = st.feedback("stars", key="neg_seg")
                
                chk_dislike = st.checkbox("Simplemente no me gustó (Ignorar estrellas)")
                
                if st.button("🎲 Generar Nueva Sugerencia", type="secondary"):
                    # Guardamos el rechazo (feedback negativo)
                    ra = n_abr + 1 if n_abr is not None else 3
                    rc = n_com + 1 if n_com is not None else 3
                    rs = n_seg + 1 if n_seg is not None else 3
                    
                    if chk_dislike:
                        ra, rc, rs = 1, 1, 1 # Penalización máxima si es dislike puro
                        
                    entry = {
                        'Date': datetime.now().strftime("%Y-%m-%d %H:%M"),
                        'City': user_city, 'Temp_Real': weather['temp'], 'User_Adj_Temp': temp_calculada,
                        'Occasion': code_occ, 'Top': rec_top, 'Bottom': rec_bot, 'Outer': rec_out,
                        'Rating_Abrigo': ra, 'Rating_Comodidad': rc, 'Rating_Seguridad': rs,
                        'Action': 'Rejected' # Marcamos que fue rechazo
                    }
                    save_feedback_entry(entry)
                    
                    # CAMBIAMOS LA SEMILLA Y RESETEAMOS MODOS
                    st.session_state['seed'] += 1 
                    st.session_state['change_mode'] = False
                    st.session_state['confirm_stage'] = 0
                    st.rerun()

        # --- FLUJO 2: CONFIRMAR Y USAR ---
        else: # Solo mostramos esto si NO estamos cambiando
            if st.session_state['confirm_stage'] == 0:
                # Estrellas de Feedback Positivo (Siempre visibles)
                st.markdown("### ⭐ Calificación del Outfit")
                st.caption("Si te gusta, calificá y dale al botón verde para usar.")
                
                c_fb1, c_fb2, c_fb3 = st.columns(3)
                with c_fb1: st.write("🌡️ Abrigo"); r_abrigo = st.feedback("stars", key="fb_abrigo")
                with c_fb2: st.write("😌 Comodidad"); r_comodidad = st.feedback("stars", key="fb_comodidad")
                with c_fb3: st.write("😎 Estilo"); r_seguridad = st.feedback("stars", key="fb_estilo")
                
                # Botón Principal
                if st.button("✅ Registrar Uso y Feedback", type="primary", use_container_width=True):
                    # 1. Verificar Límites
                    alerts = []
                    for item in selected_items_codes:
                        idx = df[df['Code'] == item['Code']].index[0]
                        current_uses = int(df.at[idx, 'Uses'])
                        
                        # Buscar límite
                        sna = decodificar_sna(item['Code'])
                        limit = 99
                        if sna:
                            if item['Category'] == 'Pantalón': limit = LIMITES_USO.get(sna['attr'], 3)
                            elif item['Category'] in ['Remera', 'Camisa']: limit = LIMITES_USO.get(sna['tipo'], 2)
                            elif item['Category'] in ['Campera', 'Buzo']: limit = LIMITES_USO.get(sna['tipo'], 5)
                        
                        # Simulamos +1 para ver si se pasa
                        if (current_uses + 1) > limit:
                             alerts.append({'code': item['Code'], 'cat': item['Category'], 'uses': current_uses, 'limit': limit})

                    if alerts:
                        st.session_state['alerts_buffer'] = alerts
                        st.session_state['confirm_stage'] = 1 # Bloqueo por lavado
                        st.rerun()
                    else:
                        # Si no hay alertas, GUARDAMOS TODO
                        # A. Actualizar Usos
                        for item in selected_items_codes:
                            idx = df[df['Code'] == item['Code']].index[0]
                            df.at[idx, 'Uses'] = int(df.at[idx, 'Uses']) + 1
                            df.at[idx, 'LastWorn'] = datetime.now().strftime("%Y-%m-%d")
                        
                        st.session_state['inventory'] = df
                        save_data(df)

                        # B. Guardar Feedback
                        ra = r_abrigo + 1 if r_abrigo is not None else 3
                        rc = r_comodidad + 1 if r_comodidad is not None else 3
                        rs = r_seguridad + 1 if r_seguridad is not None else 3
                        
                        entry = {
                            'Date': datetime.now().strftime("%Y-%m-%d %H:%M"),
                            'City': user_city, 'Temp_Real': weather['temp'], 'User_Adj_Temp': temp_calculada,
                            'Occasion': code_occ, 'Top': rec_top, 'Bottom': rec_bot, 'Outer': rec_out,
                            'Rating_Abrigo': ra, 'Rating_Comodidad': rc, 'Rating_Seguridad': rs,
                            'Action': 'Accepted'
                        }
                        save_feedback_entry(entry)
                        st.toast("¡Outfit registrado con éxito!", icon="🔥")

            # Etapa de Conflicto (Límite Lav
