import streamlit as st
import pandas as pd
import requests
import os
import pytz
import json
from datetime import datetime, timedelta
from PIL import Image
from io import BytesIO
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import random

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="GDI: Mendoza Ops v15.0", layout="centered", page_icon="🧥")

# --- CONEXIÓN A GOOGLE SHEETS ---
def get_google_sheet_client():
    try:
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds_dict = dict(st.secrets["service_account"])
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        return client
    except Exception as e:
        return None

def load_data_gsheet():
    client = get_google_sheet_client()
    if not client: return pd.DataFrame(columns=['Code', 'Category', 'Season', 'Occasion', 'ImageURL', 'Status', 'LastWorn', 'Uses', 'LaundryStart'])
    try:
        sheet = client.open("GDI_Database").worksheet("inventory")
        data = sheet.get_all_records()
        if not data: return pd.DataFrame(columns=['Code', 'Category', 'Season', 'Occasion', 'ImageURL', 'Status', 'LastWorn', 'Uses', 'LaundryStart'])
        df = pd.DataFrame(data)
        df = df.astype(str)
        return df
    except: return pd.DataFrame(columns=['Code', 'Category', 'Season', 'Occasion', 'ImageURL', 'Status', 'LastWorn', 'Uses', 'LaundryStart'])

def save_data_gsheet(df):
    client = get_google_sheet_client()
    if not client: return
    try:
        sheet = client.open("GDI_Database").worksheet("inventory")
        sheet.clear()
        df_str = df.astype(str)
        datos = [df_str.columns.values.tolist()] + df_str.values.tolist()
        sheet.update(datos)
    except: pass

def load_feedback_gsheet():
    client = get_google_sheet_client()
    if not client: return pd.DataFrame()
    try:
        sheet = client.open("GDI_Database").worksheet("feedback")
        data = sheet.get_all_records()
        return pd.DataFrame(data)
    except: return pd.DataFrame()

def save_feedback_entry_gsheet(entry):
    client = get_google_sheet_client()
    if not client: return
    try:
        sheet = client.open("GDI_Database").worksheet("feedback")
        row = [str(v) for v in entry.values()]
        sheet.append_row(row)
    except: pass

# --- CONSTANTES ---
LIMITES_USO = {
    "R": 2,   # Remeras
    "Sh": 2,  # Shorts
    "DC": 2,  # Deportivo Corto
    "Je": 4,  # Jeans
    "B": 4,   # Buzos
    "CS": 1,  # Camisas
    "Ve": 2,  # Vestir
    "DL": 2,  # Deportivo Largo
    "C": 5    # Campera
}

def get_mendoza_time():
    try: return datetime.now(pytz.timezone('America/Argentina/Mendoza'))
    except: return datetime.now()

def get_current_season():
    m = get_mendoza_time().month
    if m in [12, 1, 2]: return 'V'
    if m in [6, 7, 8]: return 'W'
    return 'M'

@st.cache_data(show_spinner=False)
def cargar_imagen_desde_url(url):
    if not url: return None
    try:
        response = requests.get(url, timeout=3)
        if response.status_code == 200: return Image.open(BytesIO(response.content))
    except: return None

def decodificar_sna(codigo):
    try:
        c = str(codigo).strip().upper()
        if len(c) < 4: return None
        season = c[0]
        if len(c) > 2 and c[1:3] == 'CS': tipo = 'CS'; idx = 3
        else: tipo = c[1]; idx = 2
        attr = c[idx:idx+2]
        return {"season": season, "tipo": tipo, "attr": attr}
    except: return None

def get_limit_for_item(category, sna):
    if not sna: return 3
    if category == 'Pantalón': return LIMITES_USO.get(sna['attr'], 2)
    elif category in ['Remera', 'Camisa']: return LIMITES_USO.get(sna['tipo'], 1)
    return LIMITES_USO.get(sna['tipo'], 3)

# --- CLIMA LOCAL (MODIFICADO: DATA HORARIA) ---
def get_weather_open_meteo():
    try:
        # Pedimos hourly=temperature_2m para calcular horas de frío
        url = "https://api.open-meteo.com/v1/forecast?latitude=-32.8908&longitude=-68.8272&current=temperature_2m,apparent_temperature,weather_code&daily=temperature_2m_max,temperature_2m_min&hourly=temperature_2m&timezone=auto"
        res = requests.get(url).json()
        
        if 'current' not in res:
            return {"temp": 15, "feels_like": 14, "min": 10, "max": 20, "desc": "Error API", "hourly_temp": [], "hourly_time": []}

        current = res['current']
        daily = res['daily']
        hourly = res.get('hourly', {})
        
        code = current['weather_code']
        desc = "Despejado"
        if code in [1, 2, 3]: desc = "Algo Nublado"
        elif code in [45, 48]: desc = "Niebla"
        elif code >= 51: desc = "Lluvia/Llovizna"
        elif code >= 95: desc = "Tormenta"
        
        return {
            "temp": current['temperature_2m'],
            "feels_like": current['apparent_temperature'],
            "min": daily['temperature_2m_min'][0],
            "max": daily['temperature_2m_max'][0],
            "desc": desc,
            "hourly_temp": hourly.get('temperature_2m', []),
            "hourly_time": hourly.get('time', [])
        }
    except:
        return {"temp": 15, "feels_like": 14, "min": 10, "max": 20, "desc": "Error Conexión", "hourly_temp": [], "hourly_time": []}

# --- FUNCIONES VIAJE ---
def get_city_coords(city_name):
    try:
        url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=1&language=es&format=json"
        res = requests.get(url, timeout=3).json()
        if 'results' in res and res['results']:
            return res['results'][0]['latitude'], res['results'][0]['longitude'], res['results'][0]['country']
        return None, None, None
    except: return None, None, None

def get_travel_forecast(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_max,temperature_2m_min,weather_code&timezone=auto"
        res = requests.get(url, timeout=3).json()
        return res.get('daily', None)
    except: return None

def get_weather_emoji(code):
    if code <= 3: return "☀️"
    if code in [45, 48]: return "🌫️"
    if code >= 51: return "☔"
    return "☁️"

# --- LÓGICA DE NEGOCIO E INTELIGENCIA ---

# >>> NUEVO: CEREBRO IA PARA CALCULAR PUNTAJE <<<
def calculate_smart_score(item_code, current_temp, feedback_df):
    """Calcula puntaje basado en historial y clima."""
    base_score = 50.0 
    if feedback_df.empty: return base_score

    # Filtrar historial de ESTA prenda
    cols_to_check = [c for c in ['Top', 'Bottom', 'Outer'] if c in feedback_df.columns]
    if not cols_to_check: return base_score
    
    # Busca donde aparece el código en cualquiera de las columnas
    mask = pd.Series(False, index=feedback_df.index)
    for col in cols_to_check:
        mask |= (feedback_df[col] == item_code)
    
    history = feedback_df[mask]

    if history.empty: return base_score

    # 1. FACTOR GUSTO
    try:
        # Convertimos a numérico por si acaso
        s_val = pd.to_numeric(history['Rating_Seguridad'], errors='coerce').mean()
        c_val = pd.to_numeric(history['Rating_Comodidad'], errors='coerce').mean()
        avg_rating = (s_val + c_val) / 2
        gusto_score = (avg_rating / 5) * 100
        if pd.isna(gusto_score): gusto_score = 50
    except: gusto_score = 50

    # 2. FACTOR CLIMA
    try:
        history['Rating_Comodidad'] = pd.to_numeric(history['Rating_Comodidad'], errors='coerce')
        history['Temp_Real'] = pd.to_numeric(history['Temp_Real'], errors='coerce')
        good_history = history[history['Rating_Comodidad'] >= 3]
        
        if not good_history.empty:
            avg_temp_usage = good_history['Temp_Real'].mean()
            diff = abs(current_temp - avg_temp_usage)
            weather_penalty = diff * 5 
            weather_score = max(0, 100 - weather_penalty)
        else:
            weather_score = 50
    except: weather_score = 50

    # 60% Clima, 40% Gusto
    return (gusto_score * 0.4) + (weather_score * 0.6)

def is_item_usable(row):
    if row['Status'] != 'Limpio': return False
    sna = decodificar_sna(row['Code'])
    if not sna: return True
    limit = get_limit_for_item(row['Category'], sna)
    try:
        uses = int(float(row['Uses'])) if row['Uses'] not in ['', 'nan'] else 0
        if uses >= limit: return False
    except: pass
    return True

def is_needs_wash(row):
    if row['Status'] in ['Sucio', 'Lavando']: return True
    sna = decodificar_sna(row['Code'])
    if not sna: return False
    limit = get_limit_for_item(row['Category'], sna)
    try:
        uses = int(float(row['Uses'])) if row['Uses'] not in ['', 'nan'] else 0
        return uses >= limit
    except: return False

def recommend_outfit(df, weather, occasion, seed):
    usable_df = df[df.apply(is_item_usable, axis=1)].copy()
    if usable_df.empty: return pd.DataFrame(), 0, ""
    
    # 1. Lista negra (rechazos de hoy)
    blacklist = set()
    try:
        fb = load_feedback_gsheet()
        if not fb.empty:
            today = get_mendoza_time().strftime("%Y-%m-%d")
            fb['Date'] = fb['Date'].astype(str)
            rej = fb[(fb['Date'].str.contains(today, na=False)) & (fb['Action'] == 'Rejected')]
            blacklist = set(rej['Top'].dropna().tolist() + rej['Bottom'].dropna().tolist() + rej['Outer'].dropna().tolist())
    except: 
        fb = pd.DataFrame() # Fallback vacío si falla carga
    
    t_curr = weather['temp']
    t_max = weather['max']
    t_min = weather['min']
    t_feel = weather.get('feels_like', t_curr) + 3 # Ajuste personal
    
    final = []
    
    # 2. Análisis Horario para Abrigo
    coat_msg = ""
    needs_coat = False
    
    hourly_temps = weather.get('hourly_temp', [])
    hourly_times = weather.get('hourly_time', [])
    
    UMBRAL_FRIO = 18 # Menos de esto sugiere abrigo
    
    if hourly_temps and hourly_times:
        hours_cold = []
        now_date = get_mendoza_time().date()
        
        for t, time_str in zip(hourly_temps, hourly_times):
            try:
                dt_hour = datetime.fromisoformat(time_str)
                if dt_hour.date() == now_date:
                    if t < UMBRAL_FRIO:
                        hours_cold.append(dt_hour.hour)
            except: pass
            
        if hours_cold:
            needs_coat = True
            start_h = min(hours_cold)
            end_h = max(hours_cold)
            if len(hours_cold) >= 12:
                coat_msg = "❄️ Usar abrigo todo el día."
            else:
                coat_msg = f"🕒 Abrigo necesario de {start_h}:00 a {end_h}:00 hs"
        else:
            coat_msg = "☀️ No hace falta abrigo hoy."
            needs_coat = False

    target_occasions = [occasion]
    if occasion == 'F': target_occasions = ['F', 'U']

    def get_best(cats, category_type):
        curr_s = get_current_season()
        pool = usable_df[(usable_df['Category'].isin(cats)) & (usable_df['Occasion'].isin(target_occasions)) & ((usable_df['Season'] == curr_s) | (usable_df['Season'] == 'T'))]
        
        if pool.empty: pool = usable_df[(usable_df['Category'].isin(cats)) & (usable_df['Occasion'].isin(target_occasions))]
        if pool.empty: pool = usable_df[usable_df['Category'].isin(cats)]
        if pool.empty: return None
        
        cands = []
        for _, r in pool.iterrows():
            sna = decodificar_sna(r['Code'])
            if not sna: continue
            match = False
            
            # --- LÓGICA PANTALONES ---
            if category_type == 'bot':
                attr = sna['attr']
                if t_max > 27: 
                    if attr in ['Sh', 'DC', 'Ve']: match = True
                    elif attr in ['Je', 'DL']: match = False 
                elif t_max < 15: 
                    if attr in ['Je', 'DL', 'Ve']: match = True
                    elif attr in ['Sh', 'DC']: match = False
                else: match = True 
            
            # --- LÓGICA SUPERIOR ---
            elif category_type == 'top':
                attr = sna['attr']
                if t_max > 30: 
                    if attr in ['00', '01']: match = True
                elif t_max < 18: 
                    if attr == '02': match = True
                else: match = True

            # --- LÓGICA ABRIGO ---
            elif category_type == 'out':
                if not needs_coat: match = False 
                else:
                    try:
                        lvl = int(sna['attr'])
                        if t_min < 10 and lvl >= 3: match = True
                        elif t_min < 16 and lvl in [2, 3]: match = True
                        elif t_min < 22 and lvl == 1: match = True
                        else: match = False
                    except: pass
            
            if match: cands.append(r)
        
        # Filtramos blacklist
        f_pool = pd.DataFrame(cands) if cands else pool
        nb = f_pool[~f_pool['Code'].isin(blacklist)]
        
        candidates_df = nb if not nb.empty else f_pool
        if candidates_df.empty: return None

        # >>> AQUÍ SE APLICA LA INTELIGENCIA (LRU + SCORING) <<<
        try:
            # 1. Ordenamos por LRU (Last Recently Used) primero
            candidates_df['LastWornDate'] = pd.to_datetime(candidates_df['LastWorn'], errors='coerce').fillna(pd.Timestamp('2000-01-01'))
            # Tomamos el TOP 50% más antiguo para asegurar rotación
            candidates_df = candidates_df.sort_values('LastWornDate', ascending=True)
            top_lru_count = max(1, int(len(candidates_df) * 0.5))
            final_candidates = candidates_df.head(top_lru_count).copy()

            # 2. Calculamos Smart Score para esos candidatos
            final_candidates['AI_Score'] = final_candidates['Code'].apply(
                lambda x: calculate_smart_score(x, t_curr, fb)
            )
            
            # 3. Agregamos Ruido aleatorio (+/- 10 pts) para variedad
            final_candidates['Final_Score'] = final_candidates['AI_Score'] + final_candidates.apply(lambda x: random.uniform(-10, 10), axis=1)
            
            # 4. Ganador
            return final_candidates.sort_values('Final_Score', ascending=False).iloc[0]

        except Exception as e:
            # Fallback a aleatorio si falla la IA
            return candidates_df.sample(1, random_state=seed).iloc[0]

    top = get_best(['Remera', 'Camisa'], 'top'); 
    if top is not None: final.append(top)
    bot = get_best(['Pantalón'], 'bot'); 
    if bot is not None: final.append(bot)
    
    if needs_coat:
        out = get_best(['Campera', 'Buzo'], 'out')
        if out is not None: final.append(out)
        
    return pd.DataFrame(final), t_feel, coat_msg

# --- INTERFAZ PRINCIPAL ---
st.sidebar.title("GDI: Mendoza Ops")
st.sidebar.caption("v15.2 - Smart AI 🧠")

user_city = st.sidebar.text_input("📍 Ciudad", value="Mendoza, AR")
user_occ = st.sidebar.selectbox("🎯 Ocasión", ["U (Universidad)", "D (Deporte)", "C (Casa)", "F (Formal)"])
code_occ = user_occ[0]

# >>>>>> LÓGICA DE LIMPIEZA DE MANUALES AL CAMBIAR DE OCASIÓN <<<<<<
if 'last_occ_viewed' not in st.session_state: st.session_state['last_occ_viewed'] = code_occ

# Si la ocasión seleccionada AHORA es distinta a la ÚLTIMA VISTA:
if st.session_state['last_occ_viewed'] != code_occ:
    st.session_state['custom_overrides'] = {} # Borramos los manuales (000000000, etc)
    st.session_state['last_occ_viewed'] = code_occ # Actualizamos la referencia
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

if 'inventory' not in st.session_state: 
    with st.spinner("Cargando sistema..."):
        st.session_state['inventory'] = load_data_gsheet()
if 'seed' not in st.session_state: st.session_state['seed'] = random.randint(1, 1000) 
if 'custom_overrides' not in st.session_state: st.session_state['custom_overrides'] = {}
if 'change_mode' not in st.session_state: st.session_state['change_mode'] = False
if 'confirm_stage' not in st.session_state: st.session_state['confirm_stage'] = 0 
if 'alerts_buffer' not in st.session_state: st.session_state['alerts_buffer'] = []

df = st.session_state['inventory']
weather = get_weather_open_meteo()

# --- SIDEBAR STATUS (MODIFICADO) ---
with st.sidebar:
    st.divider()
    with st.expander("🕴️ Estado", expanded=True):
        try:
            fb = load_feedback_gsheet()
            
            # Variables de control
            last = None
            found_outfit = False
            today_str = get_mendoza_time().strftime("%Y-%m-%d")

            if not fb.empty and 'Action' in fb.columns:
                accepted = fb[fb['Action'] == 'Accepted'].copy()
                accepted['Date'] = accepted['Date'].astype(str)
                
                # 1. Intento de Match Exacto con Ocasión Seleccionada
                match_today_occ = accepted[
                    (accepted['Date'].str.contains(today_str, na=False)) & 
                    (accepted['Occasion'] == code_occ)
                ]

                if not match_today_occ.empty:
                    last = match_today_occ.iloc[-1]
                    st.success(f"✅ Registrado ({code_occ})")
                    found_outfit = True
                else:
                    # 2. Intento de Match con CUALQUIER cosa de hoy (Fallback)
                    match_any_today = accepted[accepted['Date'].str.contains(today_str, na=False)]
                    if not match_any_today.empty:
                        last = match_any_today.iloc[-1]
                        st.info(f"🕴️ Tienes puesto: ({last['Occasion']})")
                        found_outfit = True

                # Renderizado de la ropa si se encontró algo
                if found_outfit and last is not None:
                    def show_mini(code, label):
                        if code and code != 'N/A' and code != 'nan':
                            row = df[df['Code'] == code]
                            if not row.empty:
                                img = row.iloc[0]['ImageURL']
                                if img and len(str(img)) > 5:
                                    st.image(cargar_imagen_desde_url(img), width=80)
                                else:
                                    st.write(f"🏷️ {code}")
                            else:
                                st.write(f"{code}")
                    
                    c1, c2 = st.columns(2)
                    with c1: show_mini(last['Top'], "Top")
                    with c2: show_mini(last['Bottom'], "Bot")
                    if last['Outer'] and last['Outer'] != 'N/A': show_mini(last['Outer'], "Out")
                else:
                    st.warning("⚠️ Nada registrado hoy")
            else:
                st.warning("Sin datos.")
        except Exception as e:
            st.warning("Sin datos.")

# --- TABS ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["✨ Sugerencia", "🧺 Lavadero", "📦 Inventario", "➕ Nuevo Item", "📊 Estadísticas", "✈️ Viaje"])

with tab1:
    # --- LOGICA DE OUTFIT FIJO DEL DIA ---
    today_str = get_mendoza_time().strftime("%Y-%m-%d")
    outfit_of_the_day = None
    
    # Si NO estamos forzando un cambio, buscamos si ya elegimos algo hoy
    if not st.session_state['change_mode']:
        try:
            fb = load_feedback_gsheet()
            if not fb.empty and 'Action' in fb.columns:
                accepted = fb[fb['Action'] == 'Accepted']
                accepted['Date'] = accepted['Date'].astype(str)
                # Buscamos coincidencias con la fecha de hoy Y la ocasion actual
                match = accepted[(accepted['Date'].str.contains(today_str, na=False)) & (accepted['Occasion'] == code_occ)]
                if not match.empty:
                    outfit_of_the_day = match.iloc[-1] # El ultimo aceptado
        except: pass

    coat_advice = ""
    
    if outfit_of_the_day is not None:
        # CASO A: YA ELEGISTE ROPA HOY -> MOSTRAR ESA FIJA
        st.success(f"✅ Esta es tu prenda de hoy para '{code_occ}'")
        st.info("Ya registraste este outfit. Para generar uno nuevo, toca 'Cambiar'.")
        
        codes_to_show = []
        if outfit_of_the_day['Top'] not in ['N/A', 'nan']: codes_to_show.append(outfit_of_the_day['Top'])
        if outfit_of_the_day['Bottom'] not in ['N/A', 'nan']: codes_to_show.append(outfit_of_the_day['Bottom'])
        if outfit_of_the_day['Outer'] not in ['N/A', 'nan']: codes_to_show.append(outfit_of_the_day['Outer'])
        
        recs_df = df[df['Code'].isin(codes_to_show)]
        temp_calculada = float(outfit_of_the_day['User_Adj_Temp'])
        # Calculamos solo el mensaje del abrigo actual
        _, _, coat_advice = recommend_outfit(df, weather, code_occ, 0)
        
    else:
        # CASO B: GENERAR NUEVO
        recs_df, temp_calculada, coat_advice = recommend_outfit(df, weather, code_occ, st.session_state['seed'])

    # --- OVERRIDES MANUALES ---
    for cat_key, code_val in st.session_state['custom_overrides'].items():
        if code_val:
            if code_val == "000000000":
                if cat_key == 'top': recs_df = recs_df[~recs_df['Category'].isin(['Remera', 'Camisa'])]
                elif cat_key == 'bot': recs_df = recs_df[recs_df['Category'] != 'Pantalón']
                elif cat_key == 'out': recs_df = recs_df[~recs_df['Category'].isin(['Campera', 'Buzo'])]
            elif code_val in df['Code'].values:
                manual_item = df[df['Code'] == code_val].iloc[0]
                if manual_item['Category'] in ['Remera', 'Camisa']: recs_df = recs_df[~recs_df['Category'].isin(['Remera', 'Camisa'])]
                elif manual_item['Category'] == 'Pantalón': recs_df = recs_df[recs_df['Category'] != 'Pantalón']
                elif manual_item['Category'] in ['Campera', 'Buzo']: recs_df = recs_df[~recs_df['Category'].isin(['Campera', 'Buzo'])]
                recs_df = pd.concat([recs_df, manual_item.to_frame().T], ignore_index=True)

    with st.container(border=True):
        col_w1, col_w2, col_w3 = st.columns(3)
        col_w1.metric("Clima", f"{weather['temp']}°C", weather['desc'])
        col_w2.metric("Sensación", f"{weather['feels_like']}°C", f"Max: {weather['max']}°")
        col_w3.metric("Perfil", f"{temp_calculada:.1f}°C", "+3°C adj")
        if coat_advice:
            st.markdown(f"**{coat_advice}**")

    col_h1, col_h2 = st.columns([2, 2])
    with col_h1: st.subheader("Tu Outfit")
    with col_h2: 
        c_btn1, c_btn2 = st.columns(2)
        if c_btn1.button("🔄 Cambiar", use_container_width=True): 
            st.session_state['seed'] = random.randint(1, 1000)
            st.session_state['change_mode'] = True # Forzar modo cambio
            st.session_state['custom_overrides'] = {}
            st.rerun()
        if c_btn2.button("🛠️ Manual", use_container_width=True):
            st.session_state['show_custom_ui'] = not st.session_state.get('show_custom_ui', False)

    if st.session_state.get('show_custom_ui', False):
        with st.container(border=True):
            st.markdown("###### ✍️ Ingresá el código (o `000000000` para ir sin nada):")
            with st.form("custom_outfit_form"):
                cc1, cc2, cc3 = st.columns(3)
                new_top = cc1.text_input("Torso", placeholder="Code...")
                new_bot = cc2.text_input("Piernas", placeholder="Code...")
                new_out = cc3.text_input("Abrigo", placeholder="Code...")
                if st.form_submit_button("Aplicar"):
                    overrides = {}
                    if new_top.strip(): overrides['top'] = new_top.strip().upper()
                    if new_bot.strip(): overrides['bot'] = new_bot.strip().upper()
                    if new_out.strip(): overrides['out'] = new_out.strip().upper()
                    st.session_state['custom_overrides'] = overrides; st.session_state['show_custom_ui'] = False; st.rerun()

    rec_top, rec_bot, rec_out = None, None, None
    selected_items_codes = []

    def render_card(col, title, df_subset):
        with col:
            st.markdown(f"###### {title}")
            if not df_subset.empty:
                item = df_subset.iloc[0] 
                sna = decodificar_sna(item['Code'])
                limit = get_limit_for_item(item['Category'], sna)
                uses = int(float(item['Uses'])) if item['Uses'] not in ['', 'nan'] else 0
                health = max(0.0, min(1.0, (limit - uses) / limit))
                img_data = cargar_imagen_desde_url(item['ImageURL'])
                if img_data: st.image(img_data, use_column_width=True)
                else: st.empty()
                st.markdown(f"**{item['Category']}**")
                st.caption(f"Code: `{item['Code']}`")
                st.progress(health, text=f"Vida: {uses}/{limit}")
                if health == 0: st.error("⚠️ AL LIMITE: Lavar") 
                return item
            else: st.info("🤷‍♂️ N/A"); return None

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

        # Si estamos en modo CAMBIO (o no hay outfit fijo), mostramos feedback negativo para cambiar
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
                    save_feedback_entry_gsheet(entry)
                    st.session_state['seed'] = random.randint(1, 1000) 
                    st.session_state['change_mode'] = True # Seguir cambiando
                    st.rerun()
        
        # Si NO hay outfit del dia (o estamos forzando), mostramos calificacion POSITIVA
        # Si YA HAY outfit del dia, ocultamos esto (salvo que quieras volver a calificar, pero por simpleza lo oculto)
        if outfit_of_the_day is None or st.session_state['change_mode']:
            if st.session_state['confirm_stage'] == 0:
                st.markdown("### ⭐ Confirmar y Calificar")
                
                def show_gradient_bar():
                    st.markdown('<div style="background: linear-gradient(90deg, #3b82f6 0%, #ffffff 50%, #ef4444 100%); height: 8px; border-radius: 4px; margin-bottom: 5px; opacity: 0.8;"></div>', unsafe_allow_html=True)

                st.caption("Outfit Completo")
                c_fb1, c_fb2, c_fb3 = st.columns(3)
                with c_fb1: 
                    st.markdown("**🌡️ Abrigo (1-7)**")
                    show_gradient_bar()
                    r_abrigo = st.select_slider("Global Abrigo", options=[1, 2, 3, 4, 5, 6, 7], value=4, label_visibility="collapsed", key="fb_abrigo")
                with c_fb2: st.markdown("**☁️ Comodidad**"); r_comodidad = st.feedback("stars", key="fb_comodidad")
                with c_fb3: st.markdown("**⚡ Flow**"); r_seguridad = st.feedback("stars", key="fb_estilo")

                st.divider()
                st.markdown("### 🧥 Detalle por Prenda")
                
                rt_abr, rt_com, rt_flow = 4, None, None
                if rec_top and rec_top != "N/A":
                    st.markdown(f"**Top:** `{rec_top}`")
                    c_t1, c_t2, c_t3 = st.columns(3)
                    with c_t1: show_gradient_bar(); rt_abr = st.select_slider("Top Abr", options=[1, 2, 3, 4, 5, 6, 7], value=4, label_visibility="collapsed", key="s_top_a")
                    with c_t2: rt_com = st.feedback("stars", key="s_top_c")
                    with c_t3: rt_flow = st.feedback("stars", key="s_top_f")

                rb_abr, rb_com, rb_flow = 4, None, None
                if rec_bot and rec_bot != "N/A":
                    st.markdown(f"**Bottom:** `{rec_bot}`")
                    c_b1, c_b2, c_b3 = st.columns(3)
                    with c_b1: show_gradient_bar(); rb_abr = st.select_slider("Bot Abr", options=[1, 2, 3, 4, 5, 6, 7], value=4, label_visibility="collapsed", key="s_bot_a")
                    with c_b2: rb_com = st.feedback("stars", key="s_bot_c")
                    with c_b3: rb_flow = st.feedback("stars", key="s_bot_f")
                
                ro_abr, ro_com, ro_flow = 4, None, None
                if rec_out and rec_out != "N/A":
                    st.markdown(f"**Outer:** `{rec_out}`")
                    c_o1, c_o2, c_o3 = st.columns(3)
                    with c_o1: show_gradient_bar(); ro_abr = st.select_slider("Out Abr", options=[1, 2, 3, 4, 5, 6, 7], value=4, label_visibility="collapsed", key="s_out_a")
                    with c_o2: ro_com = st.feedback("stars", key="s_out_c")
                    with c_o3: ro_flow = st.feedback("stars", key="s_out_f")

                st.divider()
                is_sweat = st.checkbox("💦 Transpiración Alta (Mandar todo a lavar)")
                
                if st.button("✅ Registrar Uso", type="primary", use_container_width=True):
                    if is_sweat:
                        for item in selected_items_codes:
                            idx = df[df['Code'] == item['Code']].index[0]
                            df.at[idx, 'Status'] = 'Sucio'
                            df.at[idx, 'LastWorn'] = datetime.now().strftime("%Y-%m-%d")
                        st.session_state['inventory'] = df; save_data_gsheet(df); st.toast("💦 A lavar."); st.session_state['change_mode'] = False; st.rerun()
                    else:
                        alerts = []
                        for item in selected_items_codes:
                            idx = df[df['Code'] == item['Code']].index[0]
                            sna = decodificar_sna(item['Code'])
                            limit = get_limit_for_item(item['Category'], sna)
                            current_uses = int(float(df.at[idx, 'Uses'])) if df.at[idx, 'Uses'] not in ['', 'nan'] else 0
                            if (current_uses + 1) > limit: 
                                alerts.append({'code': item['Code'], 'cat': item['Category'], 'uses': current_uses, 'limit': limit})
                        
                        if alerts:
                            st.session_state['alerts_buffer'] = alerts
                            st.session_state['confirm_stage'] = 1
                            st.rerun()
                        else:
                            for item in selected_items_codes:
                                idx = df[df['Code'] == item['Code']].index[0]
                                curr = int(float(df.at[idx, 'Uses'])) if df.at[idx, 'Uses'] not in ['', 'nan'] else 0
                                df.at[idx, 'Uses'] = curr + 1
                                df.at[idx, 'LastWorn'] = datetime.now().strftime("%Y-%m-%d")

                            st.session_state['inventory'] = df
                            save_data_gsheet(df)
                            
                            ra = r_abrigo
                            rc = r_comodidad + 1 if r_comodidad is not None else 3
                            rs = r_seguridad + 1 if r_seguridad is not None else 3
                            v_rt_a = rt_abr; v_rt_c = rt_com + 1 if rt_com is not None else 3; v_rt_f = rt_flow + 1 if rt_flow is not None else 3
                            v_rb_a = rb_abr; v_rb_c = rb_com + 1 if rb_com is not None else 3; v_rb_f = rb_flow + 1 if rb_flow is not None else 3
                            v_ro_a = ro_abr; v_ro_c = ro_com + 1 if ro_com is not None else 3; v_ro_f = ro_flow + 1 if ro_flow is not None else 3
                            st.session_state['custom_overrides'] = {} 
                            st.session_state['change_mode'] = False # Apagamos el modo cambio
                            
                            entry = {
                                'Date': get_mendoza_time().strftime("%Y-%m-%d %H:%M"), 
                                'City': user_city, 'Temp_Real': weather['temp'], 'User_Adj_Temp': temp_calculada, 
                                'Occasion': code_occ, 
                                'Top': rec_top, 'Bottom': rec_bot, 'Outer': rec_out, 
                                'Rating_Abrigo': ra, 'Rating_Comodidad': rc, 'Rating_Seguridad': rs, 
                                'Action': 'Accepted',
                                'Top_Abrigo': v_rt_a, 'Top_Comodidad': v_rt_c, 'Top_Flow': v_rt_f,
                                'Bot_Abrigo': v_rb_a, 'Bot_Comodidad': v_rb_c, 'Bot_Flow': v_rb_f,
                                'Out_Abrigo': v_ro_a, 'Out_Comodidad': v_ro_c, 'Out_Flow': v_ro_f
                            }
                            save_feedback_entry_gsheet(entry)
                            st.toast("¡Outfit registrado!")
                            st.rerun()

            elif st.session_state['confirm_stage'] == 1:
                st.error("🚨 ¡Límite de uso alcanzado!")
                for alert in st.session_state['alerts_buffer']:
                    st.write(f"**{alert['cat']} ({alert['code']})** al límite ({alert['uses']}/{alert['limit']})")
                    c_w1, c_w2 = st.columns(2)
                    if c_w1.button("🧼 Lavar", key=f"w_{alert['code']}"):
                        idx = df[df['Code'] == alert['code']].index[0]
                        df.at[idx, 'Status'] = 'Lavando'; df.at[idx, 'Uses'] = 0; df.at[idx, 'LaundryStart'] = datetime.now().isoformat()
                        save_data_gsheet(df); st.rerun()
                    if c_w2.button("👟 Usar igual", key=f"k_{alert['code']}"):
                        idx = df[df['Code'] == alert['code']].index[0]
                        curr = int(float(df.at[idx, 'Uses'])) if df.at[idx, 'Uses'] not in ['', 'nan'] else 0
                        df.at[idx, 'Uses'] = curr + 1; df.at[idx, 'LastWorn'] = datetime.now().strftime("%Y-%m-%d"); save_data_gsheet(df); st.session_state['confirm_stage'] = 0; st.session_state['alerts_buffer'] = []; st.session_state['change_mode'] = False; st.rerun()
    else: st.error("No hay ropa limpia disponible (según filtros). ¡Lavá algo!")

with tab2: 
    st.header("Lavadero")
    dirty_list = df[df.apply(is_needs_wash, axis=1)]
    st.subheader(f"🧺 Canasto de Ropa Sucia ({len(dirty_list)})")
    if not dirty_list.empty: st.dataframe(dirty_list[['Code', 'Category', 'Uses', 'Status']], use_container_width=True)
    else: st.info("Todo impecable ✨")
    st.divider()
    with st.container(border=True):
        col_input, col_btn = st.columns([3, 1])
        with col_input:
            with st.form("quick_wash_form", clear_on_submit=True):
                code_input = st.text_input("Ingresar Código")
                col_b1, col_b2 = st.columns(2)
                with col_b1:
                    btn_lavar = st.form_submit_button("🧼 Lavar", use_container_width=True)
                with col_b2:
                    btn_sucio = st.form_submit_button("🗑️ Sucio", use_container_width=True)
                
                if code_input:
                    code_clean = code_input.strip().upper()
                    if code_clean in df['Code'].values:
                        idx = df[df['Code'] == code_clean].index[0]
                        if btn_lavar:
                            df.at[idx, 'Status'] = 'Lavando'
                            df.at[idx, 'Uses'] = 0
                            df.at[idx, 'LaundryStart'] = datetime.now().isoformat()
                            st.session_state['inventory'] = df
                            save_data_gsheet(df)
                            st.success(f"✅ {code_clean} lavando.")
                            st.rerun()
                        elif btn_sucio:
                            df.at[idx, 'Status'] = 'Sucio'
                            st.session_state['inventory'] = df
                            save_data_gsheet(df)
                            st.toast(f"🧺 {code_clean} marcada como sucia.")
                            st.rerun()
                    elif btn_lavar or btn_sucio:
                        st.error("❌ Código no existe.")
    
    # >>> SECCION NUEVA AGREGADA AQUI ABAJO <<<
    with st.expander("🛠️ Quitar/Agregar Uso Manual"):
        c_u_input, c_u_btns = st.columns([2, 2])
        with c_u_input:
            code_mod = st.text_input("Código para modif. usos", key="cmd_uses")
        with c_u_btns:
            b_add = st.button("➕ Sumar Uso", use_container_width=True)
            b_sub = st.button("➖ Restar Uso", use_container_width=True)

        if code_mod:
            clean_code = code_mod.strip().upper()
            if clean_code in df['Code'].values:
                idx = df[df['Code'] == clean_code].index[0]
                current_uses = int(float(df.at[idx, 'Uses'])) if df.at[idx, 'Uses'] not in ['', 'nan'] else 0

                if b_add:
                    df.at[idx, 'Uses'] = current_uses + 1
                    df.at[idx, 'LastWorn'] = datetime.now().strftime("%Y-%m-%d")
                    st.session_state['inventory'] = df
                    save_data_gsheet(df)
                    st.toast(f"📈 {clean_code}: Usos subidos a {current_uses + 1}")
                    st.rerun()

                if b_sub:
                    new_uses = max(0, current_uses - 1)
                    df.at[idx, 'Uses'] = new_uses
                    st.session_state['inventory'] = df
                    save_data_gsheet(df)
                    st.toast(f"📉 {clean_code}: Usos bajados a {new_uses}")
                    st.rerun()
            elif b_add or b_sub:
                st.error("Código no encontrado")
    # >>> FIN SECCION NUEVA <<<

    edited_laundry = st.data_editor(df[['Code', 'Category', 'Status', 'Uses']], key="ed_lav", column_config={"Status": st.column_config.SelectboxColumn("Estado", options=["Limpio", "Sucio", "Lavando"], required=True)}, hide_index=True, disabled=["Code", "Category", "Uses"], use_container_width=True)
    if st.button("🔄 Actualizar Planilla"):
        df.update(edited_laundry)
        for idx in df.index:
            if df.at[idx, 'Status'] == 'Lavando' and (pd.isna(df.at[idx, 'LaundryStart']) or df.at[idx, 'LaundryStart'] == ''):
                df.at[idx, 'LaundryStart'] = datetime.now().isoformat(); df.at[idx, 'Uses'] = 0
            elif df.at[idx, 'Status'] == 'Sucio': df.at[idx, 'Uses'] = 0; df.at[idx, 'LaundryStart'] = ''
            elif df.at[idx, 'Status'] == 'Limpio': df.at[idx, 'LaundryStart'] = ''
        st.session_state['inventory'] = df; save_data_gsheet(df); st.success("Actualizado")

with tab3: 
    st.header("Inventario Total")
    edited_inv = st.data_editor(df, num_rows="dynamic", use_container_width=True, column_config={"Uses": st.column_config.ProgressColumn("Desgaste", min_value=0, max_value=10, format="%d"), "ImageURL": st.column_config.LinkColumn("Foto")})
    if st.button("💾 Guardar Inventario Completo"): 
        st.session_state['inventory'] = edited_inv; save_data_gsheet(edited_inv); st.toast("Guardado")

with tab4: 
    st.header("Alta de Prenda")
    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            temp = st.selectbox("Temporada", ["V (Verano)", "W (Invierno)", "M (Media)", "T (Toda Estación)"]).split(" ")[0]
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
        existing_codes = [c for c in df['Code'] if str(c).startswith(prefix)]
        code = f"{prefix}{len(existing_codes) + 1:02d}"
        st.info(f"Código Generado: `{code}`")
        if st.button("Agregar a la Nube"):
            new = pd.DataFrame([{'Code': code, 'Category': tipo_f.split(" - ")[1], 'Season': temp, 'Occasion': occ, 'ImageURL': url, 'Status': 'Limpio', 'LastWorn': '', 'Uses': 0, 'LaundryStart': ''}])
            st.session_state['inventory'] = pd.concat([df, new], ignore_index=True)
            save_data_gsheet(st.session_state['inventory'])
            st.success(f"¡{code} subido a Google Sheets!")

with tab5:
    st.header("📊 Estadísticas Completas")
    if not df.empty:
        total_items = len(df)
        items_needs_wash = df[df.apply(is_needs_wash, axis=1)]
        count_dirty_real = len(items_needs_wash)
        count_clean_real = total_items - count_dirty_real
        rate_dirty = count_dirty_real / total_items if total_items > 0 else 0
        st.caption("🧺 Estado del Lavadero (Real)")
        st.progress(rate_dirty, text=f"Suciedad: {int(rate_dirty*100)}% ({count_clean_real} Limpias | {count_dirty_real} Sucias)")
    st.divider()
    c_s1, c_s2 = st.columns(2)
    with c_s1:
        st.subheader("🔥 Top 5 Más Usadas")
        if not df.empty:
            df['Uses'] = pd.to_numeric(df['Uses'], errors='coerce').fillna(0)
            top_5 = df.sort_values(by='Uses', ascending=False).head(5)
            st.dataframe(top_5[['Code', 'Category', 'Uses']], hide_index=True, use_container_width=True)
    with c_s2:
        st.subheader("👻 Prendas Muertas")
        st.caption(">90 días sin uso")
        def is_dead_stock(row):
            if row['Status'] != 'Limpio': return False
            if pd.isna(row['LastWorn']) or str(row['LastWorn']) in ['', 'nan', 'None']: return False
            try:
                last_date = datetime.fromisoformat(str(row['LastWorn']))
                if (datetime.now() - last_date).days > 90: return True
            except: return False
            return False
        dead_df = df[df.apply(is_dead_stock, axis=1)]
        if not dead_df.empty: st.dataframe(dead_df[['Category', 'Code']], hide_index=True, use_container_width=True)
        else: st.success("¡Rotación impecable!")

with tab6:
    st.header("✈️ Modo Viaje v3.0 (Smart)") 
    col_dest, col_days = st.columns([2, 1])
    with col_dest: dest_city = st.text_input("📍 Destino", value="Buenos Aires")
    with col_days: num_days = st.number_input("📅 Días", min_value=1, max_value=30, value=5)

    if 'travel_weather' not in st.session_state: st.session_state['travel_weather'] = None

    if st.button("🔍 Analizar Clima Destino", use_container_width=True):
        with st.spinner(f"Consultando satélite para {dest_city}..."):
            lat, lon, country = get_city_coords(dest_city)
            if lat:
                forecast = get_travel_forecast(lat, lon)
                if forecast:
                    st.info(f"✅ Pronóstico encontrado para: {dest_city}, {country}")
                    st.session_state['travel_weather'] = forecast
                    dias = forecast['time'][:num_days]
                    maxs = forecast['temperature_2m_max'][:num_days]
                    mins = forecast['temperature_2m_min'][:num_days]
                    codes = forecast['weather_code'][:num_days]
                    cols_weather = st.columns(len(dias) if len(dias) < 5 else 5)
                    st.session_state['travel_avg_max'] = sum(maxs) / len(maxs)
                    for i, (d, mx, mn, c) in enumerate(zip(dias, maxs, mins, codes)):
                        if i < 5:
                            with cols_weather[i]:
                                day_name = datetime.strptime(d, "%Y-%m-%d").strftime("%d/%m")
                                st.metric(label=f"{get_weather_emoji(c)} {day_name}", value=f"{int(mx)}°", delta=f"{int(mn)}° min")
                else: st.error("No se pudo obtener el clima.")
            else: st.error("Ciudad no encontrada.")
    
    st.divider()
    if st.button("🎒 Generar Propuesta de Valija", type="primary", use_container_width=True):
        packable = df[df['Status'] == 'Limpio']
        forecast = st.session_state.get('travel_weather')
        if packable.empty: st.error("¡No tenés ropa limpia para viajar!")
        else:
            USE_LAUNDRY = False
            if num_days > 7:
                USE_LAUNDRY = True; target_days = 6; st.toast("Viaje largo: Ajustando para lavar ropa allá 🧼")
            else:
                target_days = num_days; st.toast("Viaje corto: Llevando ropa para todos los días")

            n_tops = target_days + 1
            n_bots = (target_days // 2) + 1
            n_out = 2

            pool_tops = packable[packable['Category'].isin(['Remera', 'Camisa'])]
            pool_bots = packable[packable['Category'] == 'Pantalón']
            pool_outs = packable[packable['Category'].isin(['Campera', 'Buzo'])]

            avg_max = st.session_state.get('travel_avg_max', 20) 

            if forecast:
                if avg_max > 25: 
                    pool_bots = pool_bots[~pool_bots['Code'].apply(lambda x: 'DL' in x or 'JE' in x)]
                    pool_outs = pool_outs[pool_outs['Code'].apply(lambda x: '04' not in x and '05' not in x)] 
                elif avg_max < 15: 
                    pool_bots = pool_bots[~pool_bots['Code'].apply(lambda x: 'SH' in x or 'DC' in x)]
                    pool_tops = pool_tops[~pool_tops['Code'].apply(lambda x: '00' in x)]

            tops = pool_tops.sample(min(len(pool_tops), n_tops))
            bots = pool_bots.sample(min(len(pool_bots), n_bots))
            outs = pool_outs.sample(min(len(pool_outs), n_out))
            
            st.session_state['travel_pack'] = pd.concat([tops, bots, outs])
            st.session_state['travel_laundry_needed'] = USE_LAUNDRY
            st.session_state['travel_selections'] = {} 
            st.rerun() 

    if st.session_state.get('travel_pack') is not None:
        pack = st.session_state['travel_pack']
        if st.session_state.get('travel_laundry_needed'): st.info("ℹ️ **Modo Lavandería Activo:** Vas por muchos días, así que calculé una valija compacta.")

        st.divider()
        st.subheader(f"🧳 Tu Valija ({len(pack)} prendas)")
        cols = st.columns(3)
        for i, (index, row) in enumerate(pack.iterrows()):
            with cols[i % 3]:
                with st.container(border=True):
                    img = cargar_imagen_desde_url(row['ImageURL'])
                    if img: st.image(img, use_container_width=True)
                    else: st.write("📷 Sin foto")
                    st.caption(f"{row['Category']} ({row['Code']})")
                    c_ida, c_vuelta = st.columns(2)
                    is_ida = c_ida.checkbox("Ida", key=f"ida_{row['Code']}")
                    is_vuelta = c_vuelta.checkbox("Vuel", key=f"vuelta_{row['Code']}")
                    if 'travel_selections' not in st.session_state: st.session_state['travel_selections'] = {}
                    st.session_state['travel_selections'][row['Code']] = {'ida': is_ida, 'vuelta': is_vuelta}

        st.divider()
        sel = st.session_state.get('travel_selections', {})
        ida_items = [code for code, vals in sel.items() if vals.get('ida')]
        vuelta_items = [code for code, vals in sel.items() if vals.get('vuelta')]
        c1, c2 = st.columns(2)
        c1.info(f"🛫 **Ida:** {', '.join(ida_items) if ida_items else '---'}")
        c2.success(f"🛬 **Vuelta:** {', '.join(vuelta_items) if vuelta_items else '---'}")
        st.divider()
        if st.button("🗑️ Borrar Valija y Empezar de Nuevo", type="secondary", use_container_width=True):
            st.session_state['travel_pack'] = None; st.session_state['travel_selections'] = {}; st.rerun()

    st.divider()
    with st.expander("📋 Checklist de Supervivencia", expanded=False):
        essentials = ["DNI / Pasaporte", "Cargador", "Cepillo Dientes", "Desodorante", "Auriculares", "Medicamentos", "Lentes", "Billetera"]
        cols_ch = st.columns(2)
        for i, item in enumerate(essentials): cols_ch[i % 2].checkbox(item, key=f"check_{i}")
