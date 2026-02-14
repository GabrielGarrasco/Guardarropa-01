import streamlit as st
import pandas as pd
import requests
import os
import pytz
import json
from datetime import datetime, timedelta
from PIL import Image
from io import BytesIO
from streamlit_gsheets import GSheetsConnection # <--- LIBRERÍA NUEVA NECESARIA

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="GDI: Mendoza Ops v10.0", layout="centered", page_icon="🧥")

# --- LÍMITES DE USO ---
LIMITES_USO = {
    "Je": 6, "Ve": 4, "DL": 3, "DC": 2, "Sh": 1,
    "R": 2, "CS": 3,
    "B": 5, "C": 10
}

# --- CONEXIÓN A GOOGLE SHEETS ---
# Esto busca las credenciales en .streamlit/secrets.toml
# Si no las configuras, te dará error.
conn = st.connection("gsheets", type=GSheetsConnection)

# --- FUNCIONES AUXILIARES ---
def get_mendoza_time():
    try:
        tz = pytz.timezone('America/Argentina/Mendoza')
        return datetime.now(tz)
    except:
        return datetime.now()

def get_current_season():
    month = get_mendoza_time().month
    if month in [12, 1, 2]: return 'V'
    if month in [6, 7, 8]: return 'W'
    return 'M'

# <--- FUNCIONES API KEY (SOLO LOCAL O MEMORIA) --->
# Nota: En la nube, secrets.json también se borra. 
# Lo ideal es poner la API KEY en los "Secrets" de la plataforma de Streamlit.
FILE_SECRETS = 'secrets.json'

def load_api_key():
    # Intenta leer de los secretos de Streamlit (Nube)
    if "OPENWEATHER_KEY" in st.secrets:
        return st.secrets["OPENWEATHER_KEY"]
    # Si no, intenta archivo local (Local)
    if os.path.exists(FILE_SECRETS):
        try:
            with open(FILE_SECRETS, 'r') as f:
                data = json.load(f)
                return data.get('api_key', '')
        except: return ''
    return ''

def save_api_key_to_file(key):
    # Esto solo funciona persistente en LOCAL. En Nube se borrará.
    with open(FILE_SECRETS, 'w') as f:
        json.dump({'api_key': key}, f)

@st.cache_data(show_spinner=False)
def cargar_imagen_desde_url(url):
    if not url: return None
    try:
        response = requests.get(url, timeout=3)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
    except: return None
    return None

def decodificar_sna(codigo):
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
    if not sna_dict: return 5
    if category == 'Pantalón': return LIMITES_USO.get(sna_dict['attr'], 3)
    elif category in ['Remera', 'Camisa']: return LIMITES_USO.get(sna_dict['tipo'], 2)
    elif category in ['Campera', 'Buzo']: return LIMITES_USO.get(sna_dict['tipo'], 5)
    return 5

# --- NUEVAS FUNCIONES DE DATOS (CLOUD NATIVE) ---
def load_data():
    """Carga el inventario desde Google Sheets."""
    try:
        # ttl=0 asegura que no use caché viejo y traiga los datos frescos
        df = conn.read(worksheet="Inventario", ttl=0)
        df['Code'] = df['Code'].astype(str)
        # Asegurar columnas numéricas
        if 'Uses' in df.columns: df['Uses'] = df['Uses'].fillna(0).astype(int)
        return df
    except Exception as e:
        st.error(f"Error conectando a Google Sheets (Inventario): {e}")
        # Retorna estructura vacía por si falla la conexión para no romper la app
        return pd.DataFrame(columns=['Code', 'Category', 'Season', 'Occasion', 'ImageURL', 'Status', 'LastWorn', 'Uses', 'LaundryStart'])

def save_data(df):
    """Guarda el inventario en Google Sheets."""
    try:
        conn.update(worksheet="Inventario", data=df)
    except Exception as e:
        st.error(f"Error guardando en Google Sheets: {e}")

def load_feedback():
    try:
        df = conn.read(worksheet="Feedback", ttl=0)
        return df
    except:
        return pd.DataFrame(columns=['Date', 'City', 'Temp_Real', 'Feels_Like', 'User_Adj_Temp', 'Occasion', 'Top', 'Bottom', 'Outer', 'Rating_Abrigo', 'Rating_Comodidad', 'Rating_Seguridad', 'Action'])

def save_feedback_entry(entry):
    try:
        # Cargamos el actual
        df = load_feedback()
        # Agregamos la fila nueva
        new_row = pd.DataFrame([entry])
        df = pd.concat([df, new_row], ignore_index=True)
        # Guardamos todo
        conn.update(worksheet="Feedback", data=df)
    except Exception as e:
        st.error(f"Error guardando feedback: {e}")

# --- RESTO DE LÓGICA ---
def get_weather(api_key, city):
    if not api_key: return {"temp": 24, "feels_like": 22, "min": 18, "max": 30, "desc": "Modo Demo"}
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric&lang=es"
        res = requests.get(url).json()
        if res.get("cod") != 200: return {"temp": 0, "feels_like": 0, "min": 0, "max": 0, "desc": "Error API"}
        return {
            "temp": res['main']['temp'], "feels_like": res['main']['feels_like'],
            "min": res['main']['temp_min'], "max": res['main']['temp_max'], 
            "desc": res['weather'][0]['description'].capitalize()
        }
    except: return {"temp": 15, "feels_like": 14, "min": 10, "max": 20, "desc": "Error Conexión"}

def check_laundry_timers(df):
    updated = False
    now = datetime.now()
    for idx, row in df.iterrows():
        if row['Status'] == 'Lavando':
            if pd.notna(row['LaundryStart']) and row['LaundryStart'] != '':
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

def recommend_outfit(df, weather, occasion, seed):
    clean_df = df[df['Status'] == 'Limpio'].copy()
    if clean_df.empty: return pd.DataFrame(), 0

    blacklist = set()
    # Cargamos feedback desde Sheets
    fb = load_feedback()
    if not fb.empty:
        try:
            today_str = get_mendoza_time().strftime("%Y-%m-%d")
            rejected_today = fb[(fb['Date'].str.contains(today_str, na=False)) & (fb['Action'] == 'Rejected')]
            blacklist = set(rejected_today['Top'].dropna().tolist() + rejected_today['Bottom'].dropna().tolist() + rejected_today['Outer'].dropna().tolist())
        except: pass

    temp_actual = weather.get('feels_like', weather['temp']) + 3 
    temp_maxima = weather.get('max', weather['temp']) + 3
    temp_minima = weather.get('min', weather['temp']) + 3
    
    final_recs = []

    def get_best_for_category(categories, is_essential=True):
        curr_season = get_current_season()
        pool = clean_df[
            (clean_df['Category'].isin(categories)) & 
            (clean_df['Occasion'] == occasion) & 
            ((clean_df['Season'] == curr_season) | (clean_df['Season'] == 'T'))
        ]
        if pool.empty: pool = clean_df[(clean_df['Category'].isin(categories)) & (clean_df['Occasion'] == occasion)]
        if pool.empty and is_essential: pool = clean_df[clean_df['Category'].isin(categories)]
        if pool.empty: return None

        candidates = []
        for _, row in pool.iterrows():
            sna = decodificar_sna(row['Code'])
            match = False
            if row['Category'] == 'Pantalón':
                tipo_pan = sna['attr']
                if temp_maxima > 28:
                    if tipo_pan in ['Sh', 'DC']: match = True
                    elif temp_actual < 24 and tipo_pan in ['Je', 'DL']: match = True
                elif temp_actual > 20: match = True
                else: 
                    if tipo_pan in ['Je', 'Ve', 'DL']: match = True
            elif row['Category'] in ['Remera', 'Camisa']:
                manga = sna['attr']
                if temp_maxima > 30 and manga in ['00', '01']: match = True
                elif temp_actual < 18 and manga == '02': match = True
                else: match = True
            elif row['Category'] in ['Campera', 'Buzo']:
                try:
                    nivel = int(sna['attr'])
                    if temp_minima < 12 and nivel >= 4: match = True
                    elif temp_minima < 16 and nivel in [2, 3]: match = True
                    elif temp_minima < 22 and nivel == 1: match = True
                except: pass
            if match: candidates.append(row)

        final_pool = pd.DataFrame(candidates) if candidates else pool
        non_blacklisted = final_pool[~final_pool['Code'].isin(blacklist)]
        if not non_blacklisted.empty: return non_blacklisted.sample(1, random_state=seed).iloc[0]
        else: return final_pool.sample(1, random_state=seed).iloc[0]

    top = get_best_for_category(['Remera', 'Camisa'], is_essential=True)
    if top is not None: final_recs.append(top)
    bot = get_best_for_category(['Pantalón'], is_essential=True)
    if bot is not None: final_recs.append(bot)
    out = get_best_for_category(['Campera', 'Buzo'], is_essential=False)
    if out is not None: final_recs.append(out)
    return pd.DataFrame(final_recs), temp_actual

# --- INTERFAZ ---
st.sidebar.title("GDI: Mendoza Ops")
st.sidebar.caption("v10.0 - Cloud Database")
st.sidebar.markdown("---")

stored_api_key = load_api_key()
if stored_api_key:
    st.sidebar.success("🔑 API Key Detectada")
    if st.sidebar.button("Cambiar Local Key"):
        save_api_key_to_file("")
        st.rerun()
    api_key = stored_api_key
else:
    api_key_input = st.sidebar.text_input("🔑 Ingresar API Key", type="password")
    if api_key_input:
        save_api_key_to_file(api_key_input)
        st.rerun()
    api_key = api_key_input

user_city = st.sidebar.text_input("📍 Ciudad", value="Mendoza, AR")
user_occ = st.sidebar.selectbox("🎯 Ocasión", ["U (Universidad)", "D (Deporte)", "C (Casa)", "F (Formal)"])
code_occ = user_occ[0]

# --- VISOR OUTFIT ACTUAL (MODIFICADO PARA USAR SHEETS) ---
fb_data = load_feedback()
if not fb_data.empty:
    try:
        accepted_outfits = fb_data[fb_data['Action'] == 'Accepted']
        if not accepted_outfits.empty:
            last_outfit = accepted_outfits.iloc[-1]
            with st.sidebar.expander("🕴️ Outfit Actual (Puesto)", expanded=False):
                st.caption(f"📅 {last_outfit['Date']}")
                def mostrar_mini_item(code, label):
                    if pd.isna(code) or code == "N/A" or not code: return
                    # Como no tenemos session_state inventory cargado antes, lo leemos
                    # Pero OJO: llamar a load_data() aquí puede ser lento si se repite.
                    # Asumimos que lo leeremos abajo.
                    st.text(f"{label}: {code}")
                mostrar_mini_item(last_outfit['Top'], "👕 Torso")
                mostrar_mini_item(last_outfit['Bottom'], "👖 Piernas")
                mostrar_mini_item(last_outfit['Outer'], "🧥 Abrigo")
    except: pass

if 'inventory' not in st.session_state: st.session_state['inventory'] = load_data()
if 'seed' not in st.session_state: st.session_state['seed'] = 42
if 'change_mode' not in st.session_state: st.session_state['change_mode'] = False
if 'confirm_stage' not in st.session_state: st.session_state['confirm_stage'] = 0 
if 'alerts_buffer' not in st.session_state: st.session_state['alerts_buffer'] = []

df_checked, updated = check_laundry_timers(st.session_state['inventory'])
if updated:
    st.session_state['inventory'] = df_checked
    save_data(df_checked)
    st.toast("🧺 Ropa limpia recuperada y guardada en Nube")

df = st.session_state['inventory']
weather = get_weather(api_key, user_city)

# --- TABS ---
# Eliminé la tab 7 de Backup porque ya no es necesaria con Google Sheets
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["✨ Sugerencia", "🧺 Lavadero", "📦 Inventario", "➕ Nuevo Item", "📊 Estadísticas", "✈️ Modo Viaje"])

with tab1:
    recs_df, temp_calculada = recommend_outfit(df, weather, code_occ, st.session_state['seed'])
    
    if 'custom_overrides' not in st.session_state: st.session_state['custom_overrides'] = {}
    for cat_key, code_val in st.session_state['custom_overrides'].items():
        if code_val and code_val in df['Code'].values:
            manual_item = df[df['Code'] == code_val].iloc[0]
            manual_cat = manual_item['Category']
            if manual_cat in ['Remera', 'Camisa']: recs_df = recs_df[~recs_df['Category'].isin(['Remera', 'Camisa'])]
            elif manual_cat == 'Pantalón': recs_df = recs_df[recs_df['Category'] != 'Pantalón']
            elif manual_cat in ['Campera', 'Buzo']: recs_df = recs_df[~recs_df['Category'].isin(['Campera', 'Buzo'])]
            recs_df = pd.concat([recs_df, manual_item.to_frame().T], ignore_index=True)

    with st.container(border=True):
        col_w1, col_w2, col_w3 = st.columns(3)
        col_w1.metric("Clima", f"{weather['temp']}°C", weather['desc'])
        col_w2.metric("Sensación", f"{weather['feels_like']}°C")
        col_w3.metric("Tu Perfil", f"{temp_calculada:.1f}°C", "+3°C adj")

    col_h1, col_h2 = st.columns([2, 2])
    with col_h1: st.subheader("Outfit Recomendado")
    with col_h2: 
        c_btn1, c_btn2 = st.columns(2)
        if c_btn1.button("🔄 Cambiar", use_container_width=True): 
            st.session_state['change_mode'] = not st.session_state['change_mode']
            st.session_state['custom_overrides'] = {} 
            st.rerun()
        if c_btn2.button("🛠️ Personalizar", use_container_width=True):
            st.session_state['show_custom_ui'] = not st.session_state.get('show_custom_ui', False)

    if st.session_state.get('show_custom_ui', False):
        with st.container(border=True):
            with st.form("custom_outfit_form"):
                cc1, cc2, cc3 = st.columns(3)
                val_top = st.session_state['custom_overrides'].get('top', '')
                val_bot = st.session_state['custom_overrides'].get('bot', '')
                val_out = st.session_state['custom_overrides'].get('out', '')
                new_top = cc1.text_input("Torso", value=val_top, placeholder="Code...")
                new_bot = cc2.text_input("Piernas", value=val_bot, placeholder="Code...")
                new_out = cc3.text_input("Abrigo", value=val_out, placeholder="Code...")
                if st.form_submit_button("Aplicar"):
                    overrides = {}
                    if new_top.strip(): overrides['top'] = new_top.strip().upper()
                    if new_bot.strip(): overrides['bot'] = new_bot.strip().upper()
                    if new_out.strip(): overrides['out'] = new_out.strip().upper()
                    st.session_state['custom_overrides'] = overrides
                    st.session_state['show_custom_ui'] = False
                    st.rerun()

    rec_top, rec_bot, rec_out = None, None, None
    selected_items_codes = []

    def render_card(col, title, df_subset):
        with col:
            st.markdown(f"###### {title}")
            if not df_subset.empty:
                item = df_subset.iloc[0] 
                sna = decodificar_sna(item['Code'])
                limit = get_limit_for_item(item['Category'], sna)
                uses = int(item['Uses'])
                health = max(0.0, min(1.0, (limit - uses) / limit))
                img_data = cargar_imagen_desde_url(item['ImageURL'])
                if img_data: st.image(img_data, use_column_width=True)
                st.markdown(f"**{item['Category']}**")
                st.caption(f"`{item['Code']}`")
                st.progress(health, text=f"Vida útil: {uses}/{limit}")
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
                with c_fb1: r_abrigo = st.feedback("stars", key="fb_abrigo")
                with c_fb2: r_comodidad = st.feedback("stars", key="fb_comodidad")
                with c_fb3: r_seguridad = st.feedback("stars", key="fb_estilo")
                
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
                        st.session_state['custom_overrides'] = {} 
                        entry = {'Date': get_mendoza_time().strftime("%Y-%m-%d %H:%M"), 'City': user_city, 'Temp_Real': weather['temp'], 'User_Adj_Temp': temp_calculada, 'Occasion': code_occ, 'Top': rec_top, 'Bottom': rec_bot, 'Outer': rec_out, 'Rating_Abrigo': ra, 'Rating_Comodidad': rc, 'Rating_Seguridad': rs, 'Action': 'Accepted'}
                        save_feedback_entry(entry); st.toast("¡Outfit guardado en Nube!"); st.rerun()

            elif st.session_state['confirm_stage'] == 1:
                st.error("🚨 ¡Límite de uso alcanzado!")
                for alert in st.session_state['alerts_buffer']:
                    st.write(f"**{alert['cat']} ({alert['code']})** al límite ({alert['uses']}/{alert['limit']})")
                    c_w1, c_w2 = st.columns(2)
                    if c_w1.button("🧼 Lavar", key=f"w_{alert['code']}"):
                        idx = df[df['Code'] == alert['code']].index[0]
                        df.at[idx, 'Status'] = 'Lavando'; df.at[idx, 'Uses'] = 0
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
                        df.at[idx, 'Status'] = 'Lavando'; df.at[idx, 'Uses'] = 0
                        df.at[idx, 'LaundryStart'] = datetime.now().isoformat()
                        st.session_state['inventory'] = df; save_data(df)
                        st.success(f"✅ {code_clean} enviado a lavar."); st.rerun()
                    else: st.error(f"❌ El código {code_clean} no existe.")

    st.markdown("---")
    st.subheader("📋 Planilla de Control")
    edited_laundry = st.data_editor(df[['Code', 'Category', 'Status', 'Uses']], key="ed_lav", column_config={"Status": st.column_config.SelectboxColumn("Estado", options=["Limpio", "Sucio", "Lavando"], required=True)}, hide_index=True, disabled=["Code", "Category", "Uses"], use_container_width=True)
    if st.button("🔄 Actualizar Planilla Completa"):
        df.update(edited_laundry)
        for idx in df.index:
            if df.at[idx, 'Status'] == 'Lavando' and pd.isna(df.at[idx, 'LaundryStart']):
                df.at[idx, 'LaundryStart'] = datetime.now().isoformat(); df.at[idx, 'Uses'] = 0
            elif df.at[idx, 'Status'] == 'Sucio': df.at[idx, 'Uses'] = 0; df.at[idx, 'LaundryStart'] = None
            elif df.at[idx, 'Status'] == 'Limpio': df.at[idx, 'LaundryStart'] = None
        st.session_state['inventory'] = df; save_data(df); st.success("Inventario actualizado")

with tab3: 
    st.subheader("📦 Inventario Total")
    edited_inv = st.data_editor(df, num_rows="dynamic", use_container_width=True, column_config={"Uses": st.column_config.ProgressColumn("Desgaste", min_value=0, max_value=10, format="%d"), "ImageURL": st.column_config.LinkColumn("Foto")})
    if st.button("💾 Guardar Cambios en Nube"): st.session_state['inventory'] = edited_inv; save_data(edited_inv); st.toast("Guardado en Google Sheets")

with tab4: 
    st.subheader("🏷️ Alta de Prenda (SNA Encoder)")
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
        code = f"{prefix}{len([c for c in df['Code'] if str(c).startswith(prefix)]) + 1:02d}"
        st.info(f"Código Generado: `{code}`")
        if st.button("Agregar a la Nube"):
            new = pd.DataFrame([{'Code': code, 'Category': tipo_f.split(" - ")[1], 'Season': temp, 'Occasion': occ, 'ImageURL': url, 'Status': 'Limpio', 'LastWorn': get_mendoza_time().strftime("%Y-%m-%d"), 'Uses': 0}])
            st.session_state['inventory'] = pd.concat([df, new], ignore_index=True); save_data(st.session_state['inventory']); st.success(f"¡{code} agregado a Google Sheets!")

with tab5:
    st.subheader("📊 Inteligencia de Guardarropas")
    c_s1, c_s2 = st.columns(2)
    with c_s1:
        st.markdown("##### 🔥 Top 5 Más Usadas")
        if not df.empty:
            top_5 = df.sort_values(by='Uses', ascending=False).head(5)
            st.dataframe(top_5[['Code', 'Category', 'Uses']], hide_index=True, use_container_width=True, column_config={"Uses": st.column_config.ProgressColumn("Usos", min_value=0, max_value=10, format="%d")})
    with c_s2:
        st.markdown("##### 🕸️ Prendas 'Muertas' (+3 meses)")
        try:
            df['LastWorn_DT'] = pd.to_datetime(df['LastWorn'], errors='coerce')
            limit_date = datetime.now() - timedelta(days=90)
            dead_stock = df[(df['Status'] == 'Limpio') & (df['LastWorn_DT'] < limit_date)]
            if not dead_stock.empty: st.dataframe(dead_stock[['Code', 'Category', 'LastWorn']], hide_index=True, use_container_width=True)
            else: st.success("¡Tu armario está vivo! Todo se usa.")
        except: st.error("Error calculando fechas.")

    st.divider()
    st.markdown("##### 📈 Tendencia de Flow")
    fb = load_feedback()
    if not fb.empty:
        try:
            fb['Avg_Score'] = (fb['Rating_Abrigo'] + fb['Rating_Comodidad'] + fb['Rating_Seguridad']) / 3
            fb['Day'] = fb['Date'].str.slice(0, 10)
            daily_trend = fb.groupby('Day')['Avg_Score'].mean()
            st.line_chart(daily_trend)
        except: pass

with tab6:
    st.subheader("✈️ Despliegue Táctico (Armado de Valija)")
    if 'travel_end_date' in st.session_state and st.session_state['travel_end_date']:
        if datetime.now() > st.session_state['travel_end_date']:
            st.session_state['travel_pack'] = None; st.session_state['travel_end_date'] = None; st.toast("📅 Viaje terminado.")

    if 'travel_pack' not in st.session_state: st.session_state['travel_pack'] = None
    with st.container(border=True):
        c_dest, c_dias, c_motivo = st.columns(3)
        dest_city = c_dest.text_input("Destino", value="Buenos Aires")
        num_days = c_dias.number_input("Duración (Días)", min_value=1, value=3, step=1)
        trip_type = c_motivo.selectbox("Tipo de Misión", ["Ocio/Turismo", "Trabajo/Formal", "Aventura"])
    
    if st.button("🎒 Generar Loadout", type="primary", use_container_width=True):
        w_dest = get_weather(api_key, dest_city)
        st.info(f"🌤️ Clima: {w_dest['temp']}°C")
        qty_tops = num_days + 1; qty_bots = (num_days // 2) + 1; qty_outer = 1 if w_dest['min'] < 25 else 0 
        packable = df[df['Status'] == 'Limpio'].copy()
        if trip_type == "Trabajo/Formal": packable = packable[packable['Occasion'].isin(['F', 'U'])]
        else: packable = packable[packable['Occasion'].isin(['C', 'D', 'U'])]
            
        tops_pool = packable[packable['Category'].isin(['Remera', 'Camisa'])]
        sel_tops = tops_pool.sample(min(len(tops_pool), qty_tops), random_state=st.session_state['seed']) if not tops_pool.empty else pd.DataFrame()
        bots_pool = packable[packable['Category'] == 'Pantalón']
        sel_bots = bots_pool.sample(min(len(bots_pool), qty_bots), random_state=st.session_state['seed']) if not bots_pool.empty else pd.DataFrame()
        outer_pool = packable[packable['Category'].isin(['Campera', 'Buzo'])]
        sel_outer = pd.DataFrame()
        if qty_outer > 0 and not outer_pool.empty:
            ideal = outer_pool[outer_pool['Code'].str.contains('C03|C04|B03', na=False)] 
            sel_outer = ideal.sample(1) if not ideal.empty else outer_pool.sample(1)

        st.session_state['travel_pack'] = {'tops': sel_tops, 'bots': sel_bots, 'outer': sel_outer, 'weather_info': w_dest}
        st.session_state['travel_end_date'] = datetime.now() + timedelta(days=num_days + 1)
        for key in list(st.session_state.keys()):
            if key.startswith("go_") or key.startswith("ret_"): del st.session_state[key]
        st.rerun()

    if st.session_state['travel_pack'] is not None:
        pack = st.session_state['travel_pack']
        st.divider(); c_tit, c_reset = st.columns([3, 1]); c_tit.markdown("### 📋 Lista de Control")
        if c_reset.button("🗑️ Borrar"):
            st.session_state['travel_pack'] = None; st.session_state['travel_end_date'] = None
            for key in list(st.session_state.keys()):
                if key.startswith("go_") or key.startswith("ret_"): del st.session_state[key]
            st.rerun()

        def render_pack_row(items, label):
            if items.empty: return
            st.markdown(f"**{label} ({len(items)})**")
            cols = st.columns(len(items)) if len(items) > 0 else []
            for idx, (_, item) in enumerate(items.iterrows()):
                with cols[idx]:
                    img = cargar_imagen_desde_url(item['ImageURL'])
                    if img: st.image(img, use_column_width=True)
                    st.caption(f"{item['Category']}")
                    c_ida, c_vuelta = st.columns(2)
                    c_ida.checkbox("🛫", key=f"go_{item['Code']}"); c_vuelta.checkbox("🏠", key=f"ret_{item['Code']}")
        render_pack_row(pack['tops'], "👕 Tops"); render_pack_row(pack['bots'], "👖 Bottoms"); render_pack_row(pack['outer'], "🧥 Abrigo")
