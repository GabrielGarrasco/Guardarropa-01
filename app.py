import streamlit as st
import pandas as pd
import requests
import os
from datetime import datetime

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="GDI: Mendoza Ops v8.4", layout="centered", page_icon="🧥")

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
    """Parsea el código SNA de forma segura."""
    try:
        codigo = str(codigo).strip()
        if len(codigo) < 4: return None
        if len(codigo) > 2 and codigo[1:3] == 'CS':
            tipo = 'CS'; idx_start_attr = 3
        else:
            tipo = codigo[1]; idx_start_attr = 2
        attr = codigo[idx_start_attr : idx_start_attr + 2]
        idx_letra_ocasion = idx_start_attr + 2
        occasion = codigo[idx_letra_ocasion] if len(codigo) > idx_letra_ocasion else "C"
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
    except: return {"temp": 15, "feels_like": 14, "min": 10, "max": 20, "desc": "Error Conexión"}

# --- LÓGICA DE RECOMENDACIÓN (PREVISORA FRÍO/CALOR) ---
def recommend_outfit(df, weather, occasion, seed):
    clean_df = df[df['Status'] == 'Limpio'].copy()
    if clean_df.empty: return pd.DataFrame(), 0

    temp_actual = weather.get('feels_like', weather['temp']) + 3 
    temp_maxima = weather.get('max', weather['temp']) + 3
    temp_minima = weather.get('min', weather['temp']) + 3
    
    recs = []
    for index, row in clean_df.iterrows():
        sna = decodificar_sna(row['Code'])
        if not sna: continue
        if sna['occasion'] != occasion: continue

        match = False
        # Pantalones: Prioriza cortos si la máxima > 28°C
        if row['Category'] == 'Pantalón':
            tipo_pan = sna['attr']
            if temp_maxima > 28:
                if tipo_pan in ['Sh', 'DC']: match = True
                elif temp_actual < 24 and tipo_pan in ['Je', 'DL']: match = True
            elif temp_actual > 20 and tipo_pan in ['Je', 'Ve', 'DL', 'Sh']: match = True
            elif temp_actual <= 20 and tipo_pan in ['Je', 'Ve', 'DL']: match = True

        # Tops: Solo cortos si la máxima > 30°C
        elif row['Category'] in ['Remera', 'Camisa']:
            manga = sna['attr']
            if temp_maxima > 30:
                if manga in ['00', '01']: match = True
            elif temp_actual < 18 and temp_maxima > 25: match = True
            else:
                if temp_actual > 25 and manga in ['00', '01']: match = True
                elif temp_actual < 15 and manga in ['02']: match = True
                else: match = True

        # Abrigo: Mira la mínima del día
        elif row['Category'] in ['Campera', 'Buzo']:
            try:
                nivel = int(sna['attr'])
                if temp_minima < 12 and nivel >= 4: match = True
                elif temp_minima < 18 and nivel in [2, 3]: match = True
                elif temp_minima < 22 and nivel == 1: match = True
            except: pass
        
        if match: recs.append(row)

    return pd.DataFrame(recs), temp_actual

# --- INTERFAZ ---
st.sidebar.title("GDI: Mendoza Ops")
st.sidebar.markdown("---")
api_key = st.sidebar.text_input("🔑 API Key", type="password")
user_city = st.sidebar.text_input("📍 Ciudad", value="Mendoza, AR")
user_occ = st.sidebar.selectbox("🎯 Ocasión", ["U (Universidad)", "D (Deporte)", "C (Casa)", "F (Formal)"])
code_occ = user_occ[0]

if 'inventory' not in st.session_state: st.session_state['inventory'] = load_data()
if 'seed' not in st.session_state: st.session_state['seed'] = 42
if 'change_mode' not in st.session_state: st.session_state['change_mode'] = False
if 'confirm_stage' not in st.session_state: st.session_state['confirm_stage'] = 0 
if 'alerts_buffer' not in st.session_state: st.session_state['alerts_buffer'] = []

df = st.session_state['inventory']
weather = get_weather(api_key, user_city)

tab1, tab2, tab3, tab4 = st.tabs(["✨ Sugerencia", "🧺 Lavadero", "📦 Inventario", "➕ Nuevo Item"])

with tab1:
    recs_df, temp_calculada = recommend_outfit(df, weather, code_occ, st.session_state['seed'])

    with st.container(border=True):
        col_w1, col_w2, col_w3 = st.columns(3)
        col_w1.metric("Clima", f"{weather['temp']}°C", weather['desc'])
        col_w2.metric("Sensación", f"{weather['feels_like']}°C")
        col_w3.metric("Tu Perfil", f"{temp_calculada:.1f}°C", "+3°C adj")

    col_h1, col_h2 = st.columns([3, 1])
    with col_h1: st.subheader("Outfit Recomendado")
    with col_h2: 
        if st.button("🔄 Cambiar"): st.session_state['change_mode'] = not st.session_state['change_mode']

    rec_top, rec_bot, rec_out = None, None, None
    selected_items_codes = []

    if not recs_df.empty:
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
            else: st.write("✅ No necesario"); rec_out = "N/A"

        st.divider()

        if st.session_state['change_mode']:
            st.info("¿Qué no te convenció?")
            with st.container(border=True):
                cf1, cf2, cf3 = st.columns(3)
                with cf1: n_abr = st.feedback("stars", key="neg_abr")
                with cf2: n_com = st.feedback("stars", key="neg_com")
                with cf3: n_seg = st.feedback("stars", key="neg_seg")
                chk_dislike = st.checkbox("Simplemente no me gustó")
                if st.button("🎲 Nueva Sugerencia"):
                    ra = n_abr + 1 if n_abr is not None else 3
                    entry = {'Date': datetime.now().strftime("%Y-%m-%d %H:%M"), 'City': user_city, 'Temp_Real': weather['temp'], 'User_Adj_Temp': temp_calculada, 'Occasion': code_occ, 'Top': rec_top, 'Bottom': rec_bot, 'Outer': rec_out, 'Rating_Abrigo': ra, 'Rating_Comodidad': 3, 'Rating_Seguridad': 3, 'Action': 'Rejected'}
                    save_feedback_entry(entry); st.session_state['seed'] += 1; st.session_state['change_mode'] = False; st.rerun()

        else:
            if st.session_state['confirm_stage'] == 0:
                st.markdown("### ⭐ Calificación")
                c_fb1, c_fb2, c_fb3 = st.columns(3)
                with c_fb1: r_abrigo = st.feedback("stars", key="fb_abrigo")
                with c_fb2: r_comodidad = st.feedback("stars", key="fb_comodidad")
                with c_fb3: r_seguridad = st.feedback("stars", key="fb_estilo")
                if st.button("✅ Registrar Uso y Feedback", type="primary", use_container_width=True):
                    alerts = []
                    for item in selected_items_codes:
                        idx = df[df['Code'] == item['Code']].index[0]
                        sna = decodificar_sna(item['Code'])
                        limit = 99
                        if sna:
                            if item['Category'] == 'Pantalón': limit = LIMITES_USO.get(sna['attr'], 3)
                            elif item['Category'] in ['Remera', 'Camisa']: limit = LIMITES_USO.get(sna['tipo'], 2)
                            elif item['Category'] in ['Campera', 'Buzo']: limit = LIMITES_USO.get(sna['tipo'], 5)
                        if (int(df.at[idx, 'Uses']) + 1) > limit: alerts.append({'code': item['Code'], 'cat': item['Category'], 'uses': int(df.at[idx, 'Uses']), 'limit': limit})
                    if alerts: st.session_state['alerts_buffer'] = alerts; st.session_state['confirm_stage'] = 1; st.rerun()
                    else:
                        for item in selected_items_codes:
                            idx = df[df['Code'] == item['Code']].index[0]
                            df.at[idx, 'Uses'] = int(df.at[idx, 'Uses']) + 1
                        st.session_state['inventory'] = df; save_data(df)
                        ra = r_abrigo + 1 if r_abrigo is not None else 3
                        rc = r_comodidad + 1 if r_comodidad is not None else 3
                        rs = r_seguridad + 1 if r_seguridad is not None else 3
                        entry = {'Date': datetime.now().strftime("%Y-%m-%d %H:%M"), 'City': user_city, 'Temp_Real': weather['temp'], 'User_Adj_Temp': temp_calculada, 'Occasion': code_occ, 'Top': rec_top, 'Bottom': rec_bot, 'Outer': rec_out, 'Rating_Abrigo': ra, 'Rating_Comodidad': rc, 'Rating_Seguridad': rs, 'Action': 'Accepted'}
                        save_feedback_entry(entry); st.toast("¡Outfit registrado!"); st.rerun()

            elif st.session_state['confirm_stage'] == 1:
                st.error("🚨 ¡Límite de uso alcanzado!")
                for alert in st.session_state['alerts_buffer']:
                    with st.container(border=True):
                        st.write(f"**{alert['cat']} ({alert['code']})**")
                        c_w1, c_w2 = st.columns(2)
                        if c_w1.button("🧼 Lavar", key=f"w_{alert['code']}"):
                            idx = df[df['Code'] == alert['code']].index[0]
                            df.at[idx, 'Status'] = 'Sucio'; df.at[idx, 'Uses'] = 0; save_data(df); st.rerun()
                        if c_w2.button("👟 Usar igual", key=f"k_{alert['code']}"):
                            idx = df[df['Code'] == alert['code']].index[0]
                            df.at[idx, 'Uses'] = int(df.at[idx, 'Uses']) + 1; save_data(df); st.session_state['confirm_stage'] = 0; st.session_state['alerts_buffer'] = []; st.rerun()
    else: st.error("No hay ropa limpia disponible.")

with tab2: # Lavadero
    st.subheader("🧺 Gestión de Lavado")
    edited_laundry = st.data_editor(df[['Code', 'Category', 'Status', 'Uses']], key="ed_lav", column_config={"Status": st.column_config.SelectboxColumn("Estado", options=["Limpio", "Sucio", "Lavando"], required=True)}, hide_index=True, disabled=["Code", "Category", "Uses"], use_container_width=True)
    if st.button("🔄 Actualizar Lavadero"):
        df.update(edited_laundry)
        for idx in df.index:
            if df.at[idx, 'Status'] in ['Lavando', 'Sucio']: df.at[idx, 'Uses'] = 0
        st.session_state['inventory'] = df; save_data(df); st.success("Actualizado")

with tab3: # Inventario
    st.subheader("📦 Inventario Total")
    edited_inv = st.data_editor(df, num_rows="dynamic", use_container_width=True, column_config={"Uses": st.column_config.ProgressColumn("Desgaste", min_value=0, max_value=10, format="%d")})
    if st.button("💾 Guardar Inventario"): st.session_state['inventory'] = edited_inv; save_data(edited_inv); st.toast("Guardado")

with tab4: # Carga Manual
    st.subheader("🏷️ Alta de Prenda")
    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            temp = st.selectbox("Temporada", ["V (Verano)", "W (Invierno)", "M (Media)"]).split(" ")[0]
            tipo_f = st.selectbox("Tipo", ["R - Remera", "CS - Camisa", "P - Pantalón", "C - Campera", "B - Buzo"])
            t_code = {"R - Remera":"R", "CS - Camisa":"CS", "P - Pantalón":"P", "C - Campera":"C", "B - Buzo":"B"}[tipo_f]
            if t_code == "P": attr = st.selectbox("Corte", ["Je (Jean)", "Sh (Short)", "DL (Deportivo)", "DC (Corto)", "Ve (Vestir)"]).split(" ")[0]
            elif t_code in ["C", "B"]: attr = f"0{st.selectbox('Abrigo', ['1 (Rompevientos)', '2 (Liviana)', '3 (Normal)', '4 (Gruesa)', '5 (Muy Gruesa)']).split(' ')[0]}"
            else: attr = st.selectbox("Manga", ["00 (Musculosa)", "01 (Corta)", "02 (Larga)"]).split(" ")[0]
        with c2:
            occ = st.selectbox("Ocasión", ["U", "D", "C", "F"])
            col = st.selectbox("Color", ["01-Blanco", "02-Negro", "03-Gris", "04-Azul", "05-Verde", "06-Rojo", "07-Amarillo", "08-Beige", "09-Marron", "10-Denim", "11-Naranja", "12-Violeta", "99-Estampado"])[:2]
            url = st.text_input("URL Foto")
        
        prefix = f"{temp}{t_code}{attr}{occ}{col}"
        code = f"{prefix}{len([c for c in df['Code'] if str(c).startswith(prefix)]) + 1:02d}"
        st.write(f"Code: `{code}`")
        if st.button("Agregar"):
            new = pd.DataFrame([{'Code': code, 'Category': tipo_f.split(" - ")[1], 'Season': temp, 'Occasion': occ, 'ImageURL': url, 'Status': 'Limpio', 'LastWorn': datetime.now().strftime("%Y-%m-%d"), 'Uses': 0}])
            st.session_state['inventory'] = pd.concat([df, new], ignore_index=True); save_data(st.session_state['inventory']); st.success("Agregado")
