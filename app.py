import streamlit as st

import pandas as pd

import requests

import os

import pytz

from datetime import datetime, timedelta



# --- CONFIGURACIÓN ---

st.set_page_config(page_title="GDI: Mendoza Ops v9.3", layout="centered", page_icon="🧥")



FILE_INV = 'inventory.csv'

FILE_FEEDBACK = 'feedback.csv'



# --- LÍMITES DE USO (INGENIERÍA DE CICLO DE VIDA) ---

LIMITES_USO = {

    "Je": 6, "Ve": 4, "DL": 3, "DC": 2, "Sh": 1, # Pantalones

    "R": 2, "CS": 3,                             # Tops

    "B": 5, "C": 10                              # Abrigos

}



# --- FUNCIONES AUXILIARES ---

def get_mendoza_time():

    """Obtiene la hora real de Mendoza."""

    try:

        tz = pytz.timezone('America/Argentina/Mendoza')

        return datetime.now(tz)

    except:

        return datetime.now()



def decodificar_sna(codigo):

    """Parsea el código SNA de forma robusta."""

    try:

        codigo = str(codigo).strip().upper()

        if len(codigo) < 4: return None

        

        season = codigo[0]

        

        if len(codigo) > 2 and codigo[1:3] == 'CS':

            tipo = 'CS'; idx_start_attr = 3

        else:

            tipo = codigo[1]; idx_start_attr = 2

            

        attr = codigo[idx_start_attr : idx_start_attr + 2]

        idx_occ = idx_start_attr + 2

        occasion = codigo[idx_occ] if len(codigo) > idx_occ else "C"

        

        return {"season": season, "tipo": tipo, "attr": attr, "occasion": occasion}

    except: return None



def get_limit_for_item(category, sna_dict):

    """Devuelve el límite de usos."""

    if not sna_dict: return 5

    if category == 'Pantalón': return LIMITES_USO.get(sna_dict['attr'], 3)

    elif category in ['Remera', 'Camisa']: return LIMITES_USO.get(sna_dict['tipo'], 2)

    elif category in ['Campera', 'Buzo']: return LIMITES_USO.get(sna_dict['tipo'], 5)

    return 5



def load_data():

    if not os.path.exists(FILE_INV):

        return pd.DataFrame(columns=['Code', 'Category', 'Season', 'Occasion', 'ImageURL', 'Status', 'LastWorn', 'Uses', 'LaundryStart'])

    df = pd.read_csv(FILE_INV)

    df['Code'] = df['Code'].astype(str)

    if 'Uses' not in df.columns: df['Uses'] = 0

    if 'LaundryStart' not in df.columns: df['LaundryStart'] = None 

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



# --- LÓGICA AUTOMÁTICA DE LAVADO (24 HS) ---

def check_laundry_timers(df):

    updated = False

    now = datetime.now()

    for idx, row in df.iterrows():

        if row['Status'] == 'Lavando':

            if pd.notna(row['LaundryStart']):

                try:

                    start_time = datetime.fromisoformat(str(row['LaundryStart']))

                    if (now - start_time).total_seconds() > 86400:

                        df.at[idx, 'Status'] = 'Limpio'

                        df.at[idx, 'Uses'] = 0

                        df.at[idx, 'LaundryStart'] = None

                        updated = True

                except: pass

            else:

                df.at[idx, 'LaundryStart'] = now.isoformat()

                updated = True

    return df, updated



# --- LÓGICA DE RECOMENDACIÓN ---

def recommend_outfit(df, weather, occasion, seed):

    clean_df = df[df['Status'] == 'Limpio'].copy()

    if clean_df.empty: return pd.DataFrame(), 0



    if os.path.exists(FILE_FEEDBACK):

        try:

            fb = pd.read_csv(FILE_FEEDBACK)

            today_str = get_mendoza_time().strftime("%Y-%m-%d")

            rejected_today = fb[(fb['Date'].str.contains(today_str, na=False)) & (fb['Action'] == 'Rejected')]

            blacklist = set(rejected_today['Top'].dropna().tolist() + rejected_today['Bottom'].dropna().tolist() + rejected_today['Outer'].dropna().tolist())

            clean_df = clean_df[~clean_df['Code'].isin(blacklist)]

        except: pass



    temp_actual = weather.get('feels_like', weather['temp']) + 3 

    temp_maxima = weather.get('max', weather['temp']) + 3

    temp_minima = weather.get('min', weather['temp']) + 3

    

    recs = []

    for index, row in clean_df.iterrows():

        sna = decodificar_sna(row['Code'])

        if not sna: continue

        if sna['occasion'] != occasion: continue



        match = False

        if row['Category'] == 'Pantalón':

            tipo_pan = sna['attr']

            if temp_maxima > 28:

                if tipo_pan in ['Sh', 'DC']: match = True

                elif temp_actual < 24 and tipo_pan in ['Je', 'DL']: match = True

            elif temp_actual > 20 and tipo_pan in ['Je', 'Ve', 'DL', 'Sh']: match = True

            elif temp_actual <= 20 and tipo_pan in ['Je', 'Ve', 'DL']: match = True

        elif row['Category'] in ['Remera', 'Camisa']:

            manga = sna['attr']

            if temp_maxima > 30:

                if manga in ['00', '01']: match = True

            elif temp_actual < 18 and temp_maxima > 25: match = True

            else:

                if temp_actual > 25 and manga in ['00', '01']: match = True

                elif temp_actual < 15 and manga in ['02']: match = True

                else: match = True

        elif row['Category'] in ['Campera', 'Buzo']:

            try:

                nivel = int(sna['attr'])

                if temp_minima < 12 and nivel >= 4: match = True

                elif temp_minima < 16 and nivel in [2, 3]: match = True

                elif temp_minima < 22 and nivel == 1: match = True

            except: pass

        if match: recs.append(row)



    return pd.DataFrame(recs), temp_actual



# --- INTERFAZ ---

st.sidebar.title("GDI: Mendoza Ops")

st.sidebar.caption("v9.3 - Stats Edition")

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



df_checked, updated = check_laundry_timers(st.session_state['inventory'])

if updated:

    st.session_state['inventory'] = df_checked

    save_data(df_checked)

    st.toast("🧺 Ropa limpia recuperada automáticamente")



df = st.session_state['inventory']

weather = get_weather(api_key, user_city)



# --- AQUI ESTÁ EL CAMBIO DE LOS TABS ---

tab1, tab2, tab3, tab4, tab5 = st.tabs(["✨ Sugerencia", "🧺 Lavadero", "📦 Inventario", "➕ Nuevo Item", "📊 Estadísticas"])



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

        if st.button("🔄 Cambiar"): 

            st.session_state['change_mode'] = not st.session_state['change_mode']



    rec_top, rec_bot, rec_out = None, None, None

    selected_items_codes = []



    def render_card(col, title, df_subset):

        with col:

            st.markdown(f"###### {title}")

            if not df_subset.empty:

                item = df_subset.sample(1, random_state=st.session_state['seed']).iloc[0]

                sna = decodificar_sna(item['Code'])

                limit = get_limit_for_item(item['Category'], sna)

                uses = int(item['Uses'])

                health = max(0.0, min(1.0, (limit - uses) / limit))

                

                if pd.notna(item['ImageURL']) and item['ImageURL']: st.image(item['ImageURL'], use_column_width=True)

                else: st.empty()

                

                st.markdown(f"**{item['Category']}**")

                st.caption(f"Code: `{item['Code']}`")

                st.progress(health, text=f"Vida útil: {uses}/{limit} usos")

                if health < 0.25: st.warning("⚠️ Lavar pronto")

                return item

            else:

                st.info("🤷‍♂️ N/A")

                return None



    if not recs_df.empty:

        c1, c2, c3 = st.columns(3)

        rec_top_item = render_card(c1, "Torso", recs_df[recs_df['Category'].isin(['Remera', 'Camisa'])])

        if rec_top_item is not None: rec_top = rec_top_item['Code']; selected_items_codes.append(rec_top_item)

        

        rec_bot_item = render_card(c2, "Piernas", recs_df[recs_df['Category'] == 'Pantalón'])

        if rec_bot_item is not None: rec_bot = rec_bot_item['Code']; selected_items_codes.append(rec_bot_item)

        

        rec_out_item = render_card(c3, "Abrigo", recs_df[recs_df['Category'].isin(['Campera', 'Buzo'])])

        if rec_out_item is not None: rec_out = rec_out_item['Code']; selected_items_codes.append(rec_out_item)

        else: rec_out = "N/A"



        st.divider()



        if st.session_state['change_mode']:

            st.info("¿Qué no te convenció?")

            with st.container(border=True):

                cf1, cf2, cf3 = st.columns(3)

                with cf1: n_abr = st.feedback("stars", key="neg_abr")

                with cf2: n_com = st.feedback("stars", key="neg_com")

                with cf3: n_seg = st.feedback("stars", key="neg_seg")

                if st.button("🎲 Dame otra opción"):

                    ra = n_abr + 1 if n_abr is not None else 3

                    entry = {'Date': get_mendoza_time().strftime("%Y-%m-%d %H:%M"), 'City': user_city, 'Temp_Real': weather['temp'], 'User_Adj_Temp': temp_calculada, 'Occasion': code_occ, 'Top': rec_top, 'Bottom': rec_bot, 'Outer': rec_out, 'Rating_Abrigo': ra, 'Rating_Comodidad': 3, 'Rating_Seguridad': 3, 'Action': 'Rejected'}

                    save_feedback_entry(entry); st.session_state['seed'] += 1; st.session_state['change_mode'] = False; st.rerun()

        else:

            if st.session_state['confirm_stage'] == 0:

                st.markdown("### ⭐ Calificación del día")

                c_fb1, c_fb2, c_fb3 = st.columns(3)

                with c_fb1: 

                    st.markdown("**🌡️ Nivel de Abrigo**")

                    r_abrigo = st.feedback("stars", key="fb_abrigo")

                with c_fb2: 

                    st.markdown("**☁️ Nivel de Comodidad**")

                    r_comodidad = st.feedback("stars", key="fb_comodidad")

                with c_fb3: 

                    st.markdown("**⚡ Nivel de Flow**")

                    r_seguridad = st.feedback("stars", key="fb_estilo")

                

                if st.button("✅ Registrar Uso y Feedback", type="primary", use_container_width=True):

                    alerts = []

                    for item in selected_items_codes:

                        idx = df[df['Code'] == item['Code']].index[0]

                        sna = decodificar_sna(item['Code'])

                        limit = get_limit_for_item(item['Category'], sna)

                        if (int(df.at[idx, 'Uses']) + 1) > limit: alerts.append({'code': item['Code'], 'cat': item['Category'], 'uses': int(df.at[idx, 'Uses']), 'limit': limit})

                    

                    if alerts:

                        st.session_state['alerts_buffer'] = alerts; st.session_state['confirm_stage'] = 1; st.rerun()

                    else:

                        for item in selected_items_codes:

                            idx = df[df['Code'] == item['Code']].index[0]

                            df.at[idx, 'Uses'] = int(df.at[idx, 'Uses']) + 1

                        st.session_state['inventory'] = df; save_data(df)

                        ra = r_abrigo + 1 if r_abrigo is not None else 3

                        rc = r_comodidad + 1 if r_comodidad is not None else 3

                        rs = r_seguridad + 1 if r_seguridad is not None else 3

                        entry = {'Date': get_mendoza_time().strftime("%Y-%m-%d %H:%M"), 'City': user_city, 'Temp_Real': weather['temp'], 'User_Adj_Temp': temp_calculada, 'Occasion': code_occ, 'Top': rec_top, 'Bottom': rec_bot, 'Outer': rec_out, 'Rating_Abrigo': ra, 'Rating_Comodidad': rc, 'Rating_Seguridad': rs, 'Action': 'Accepted'}

                        save_feedback_entry(entry); st.toast("¡Outfit registrado!"); st.rerun()



            elif st.session_state['confirm_stage'] == 1:

                st.error("🚨 ¡Límite de uso alcanzado!")

                for alert in st.session_state['alerts_buffer']:

                    st.write(f"**{alert['cat']} ({alert['code']})** al límite ({alert['uses']}/{alert['limit']})")

                    c_w1, c_w2 = st.columns(2)

                    if c_w1.button("🧼 Lavar", key=f"w_{alert['code']}"):

                        idx = df[df['Code'] == alert['code']].index[0]

                        df.at[idx, 'Status'] = 'Lavando'

                        df.at[idx, 'Uses'] = 0

                        df.at[idx, 'LaundryStart'] = datetime.now().isoformat()

                        save_data(df); st.rerun()

                    if c_w2.button("👟 Usar igual", key=f"k_{alert['code']}"):

                        idx = df[df['Code'] == alert['code']].index[0]

                        df.at[idx, 'Uses'] = int(df.at[idx, 'Uses']) + 1; save_data(df); st.session_state['confirm_stage'] = 0; st.session_state['alerts_buffer'] = []; st.rerun()

    else: st.error("No hay ropa limpia disponible.")



with tab2: 

    st.subheader("🚿 Ingreso Rápido al Lavadero")

    with st.container(border=True):

        col_input, col_btn = st.columns([3, 1])

        with col_input:

            with st.form("quick_wash_form", clear_on_submit=True):

                code_input = st.text_input("Ingresar Código", placeholder="Ej: VR01C0501...")

                submitted = st.form_submit_button("🧼 Mandar a Lavar", use_container_width=True)

                

                if submitted and code_input:

                    code_clean = code_input.strip().upper()

                    if code_clean in df['Code'].values:

                        idx = df[df['Code'] == code_clean].index[0]

                        df.at[idx, 'Status'] = 'Lavando'

                        df.at[idx, 'Uses'] = 0

                        df.at[idx, 'LaundryStart'] = datetime.now().isoformat()

                        st.session_state['inventory'] = df
