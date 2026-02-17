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

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="GDI: Mendoza Ops v11.4", layout="centered", page_icon="🧥")

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

# --- LÍMITES Y FUNCIONES ---
LIMITES_USO = {"Je": 6, "Ve": 4, "DL": 3, "DC": 2, "Sh": 1, "R": 2, "CS": 3, "B": 5, "C": 10}

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
    if not sna: return 5
    if category == 'Pantalón': return LIMITES_USO.get(sna['attr'], 3)
    elif category in ['Remera', 'Camisa']: return LIMITES_USO.get(sna['tipo'], 2)
    return LIMITES_USO.get(sna['tipo'], 5)

def get_weather_open_meteo():
    try:
        url = "https://api.open-meteo.com/v1/forecast?latitude=-32.8908&longitude=-68.8272&current=temperature_2m,apparent_temperature,weather_code&daily=temperature_2m_max,temperature_2m_min&timezone=auto"
        res = requests.get(url).json()
        
        if 'current' not in res:
            return {"temp": 15, "feels_like": 14, "min": 10, "max": 20, "desc": "Error API"}

        current = res['current']
        daily = res['daily']
        
        code = current['weather_code']
        desc = "Despejado"
        if code in [1, 2, 3]: desc = "Algo Nublado"
        elif code in [45, 48]: desc = "Niebla"
        elif code in [51, 53, 55]: desc = "Llovizna"
        elif code in [61, 63, 65]: desc = "Lluvia"
        elif code in [71, 73, 75]: desc = "Nieve"
        elif code >= 95: desc = "Tormenta"
        
        return {
            "temp": current['temperature_2m'],
            "feels_like": current['apparent_temperature'],
            "min": daily['temperature_2m_min'][0],
            "max": daily['temperature_2m_max'][0],
            "desc": desc
        }
    except:
        return {"temp": 15, "feels_like": 14, "min": 10, "max": 20, "desc": "Error Conexión"}

def check_laundry_timers(df):
    updated = False
    now = datetime.now()
    for idx, row in df.iterrows():
        if row['Status'] == 'Lavando':
            if pd.notna(row['LaundryStart']) and str(row['LaundryStart']) not in ['', 'nan']:
                try:
                    start = datetime.fromisoformat(str(row['LaundryStart']))
                    if (now - start).total_seconds() > 86400:
                        df.at[idx, 'Status'] = 'Limpio'; df.at[idx, 'Uses'] = 0; df.at[idx, 'LaundryStart'] = ''; updated = True
                except: pass
            else:
                df.at[idx, 'LaundryStart'] = now.isoformat(); updated = True
    return df, updated

def recommend_outfit(df, weather, occasion, seed):
    clean = df[df['Status'] == 'Limpio'].copy()
    if clean.empty: return pd.DataFrame(), 0
    blacklist = set()
    try:
        fb = load_feedback_gsheet()
        if not fb.empty:
            today = get_mendoza_time().strftime("%Y-%m-%d")
            fb['Date'] = fb['Date'].astype(str)
            rej = fb[(fb['Date'].str.contains(today, na=False)) & (fb['Action'] == 'Rejected')]
            blacklist = set(rej['Top'].dropna().tolist() + rej['Bottom'].dropna().tolist() + rej['Outer'].dropna().tolist())
    except: pass
    
    t_feel = weather.get('feels_like', weather['temp']) + 3
    t_max = weather.get('max', weather['temp']) + 3
    t_min = weather.get('min', weather['temp']) + 3
    final = []

    target_occasions = [occasion]
    if occasion == 'F':
        target_occasions = ['F', 'U']

    def get_best(cats, ess=True):
        curr_s = get_current_season()
        pool = clean[(clean['Category'].isin(cats)) & (clean['Occasion'].isin(target_occasions)) & ((clean['Season'] == curr_s) | (clean['Season'] == 'T'))]
        
        if pool.empty: pool = clean[(clean['Category'].isin(cats)) & (clean['Occasion'].isin(target_occasions))]
        if pool.empty and ess: pool = clean[clean['Category'].isin(cats)]
        if pool.empty: return None
        
        cands = []
        for _, r in pool.iterrows():
            sna = decodificar_sna(r['Code'])
            if not sna: continue
            match = False
            if r['Category'] == 'Pantalón':
                attr = sna['attr']
                if t_max > 28:
                    if attr in ['Sh', 'DC']: match = True
                    elif t_feel < 24 and attr in ['Je', 'DL']: match = True
                elif t_feel > 20: match = True
                else: 
                    if attr in ['Je', 'Ve', 'DL']: match = True
            elif r['Category'] in ['Remera', 'Camisa']:
                attr = sna['attr']
                if t_max > 30 and attr in ['00', '01']: match = True
                elif t_feel < 18 and attr == '02': match = True
                else: match = True
            elif r['Category'] in ['Campera', 'Buzo']:
                try:
                    lvl = int(sna['attr'])
                    if t_min < 12 and lvl >= 4: match = True
                    elif t_min < 16 and lvl in [2, 3]: match = True
                    elif t_min < 22 and lvl == 1: match = True
                except: pass
            if match: cands.append(r)
        
        f_pool = pd.DataFrame(cands) if cands else pool
        nb = f_pool[~f_pool['Code'].isin(blacklist)]
        return nb.sample(1, random_state=seed).iloc[0] if not nb.empty else f_pool.sample(1, random_state=seed).iloc[0]

    top = get_best(['Remera', 'Camisa']); 
    if top is not None: final.append(top)
    bot = get_best(['Pantalón']); 
    if bot is not None: final.append(bot)
    out = get_best(['Campera', 'Buzo'], False)
    if out is not None: final.append(out)
    return pd.DataFrame(final), t_feel

# --- INTERFAZ PRINCIPAL ---
st.sidebar.title("GDI: Mendoza Ops")
st.sidebar.caption("v11.4 - Multi-Ocasión")

user_city = st.sidebar.text_input("📍 Ciudad", value="Mendoza, AR")
user_occ = st.sidebar.selectbox("🎯 Ocasión", ["U (Universidad)", "D (Deporte)", "C (Casa)", "F (Formal)"])
code_occ = user_occ[0]

if 'inventory' not in st.session_state: 
    with st.spinner("Cargando sistema..."):
        st.session_state['inventory'] = load_data_gsheet()
if 'seed' not in st.session_state: st.session_state['seed'] = 42
if 'custom_overrides' not in st.session_state: st.session_state['custom_overrides'] = {}
if 'change_mode' not in st.session_state: st.session_state['change_mode'] = False
if 'confirm_stage' not in st.session_state: st.session_state['confirm_stage'] = 0 
if 'alerts_buffer' not in st.session_state: st.session_state['alerts_buffer'] = []

df_checked, updated = check_laundry_timers(st.session_state['inventory'])
if updated:
    st.session_state['inventory'] = df_checked
    save_data_gsheet(df_checked) 

df = st.session_state['inventory']
weather = get_weather_open_meteo()

# --- VISOR SIDEBAR INTELIGENTE (FILTRADO POR OCASIÓN) ---
with st.sidebar:
    st.divider()
    with st.expander("🕴️ Estado: " + code_occ, expanded=True):
        try:
            fb = load_feedback_gsheet()
            found_outfit_for_occ = False
            today_str = get_mendoza_time().strftime("%Y-%m-%d")
            
            ya_registrado_hoy = False # Variable de control para el botón principal

            if not fb.empty and 'Action' in fb.columns:
                # 1. Filtramos SOLO los Aceptados
                accepted = fb[fb['Action'] == 'Accepted'].copy()
                accepted['Date'] = accepted['Date'].astype(str)
                
                # 2. Buscamos si hay registros para HOY y para la OCASIÓN SELECCIONADA
                # Esto es clave: filtramos por code_occ
                match_today_occ = accepted[
                    (accepted['Date'].str.contains(today_str, na=False)) & 
                    (accepted['Occasion'] == code_occ)
                ]

                # Si encontramos uno para hoy y esta ocasión
                if not match_today_occ.empty:
                    last = match_today_occ.iloc[-1] # El último de esta ocasión hoy
                    st.success(f"✅ Look de Hoy ({code_occ})")
                    found_outfit_for_occ = True
                    ya_registrado_hoy = True
                    
                    def show_mini(code, label):
                        if code and code != 'N/A' and code != 'nan':
                            row = df[df['Code'] == code]
                            if not row.empty:
                                img = row.iloc[0]['ImageURL']
                                st.image(cargar_imagen_desde_url(img), width=80) if img else st.write(f"{label}: {code}")
                            else: st.write(f"{label}: {code}")
                    
                    c1, c2 = st.columns(2)
                    with c1: show_mini(last['Top'], "Top")
                    with c2: show_mini(last['Bottom'], "Bot")
                    if last['Outer'] and last['Outer'] != 'N/A':
                        show_mini(last['Outer'], "Out")
                else:
                    # Si no hay de hoy para esta ocasión, mostramos aviso
                    st.warning(f"⚠️ Sin registrar {code_occ} hoy")
                    
                    # Opcional: Mostrar el último histórico de esta ocasión
                    match_hist_occ = accepted[accepted['Occasion'] == code_occ]
                    if not match_hist_occ.empty:
                        last_hist = match_hist_occ.iloc[-1]
                        st.caption(f"Último {code_occ}: {last_hist['Date']}")
            
            if not found_outfit_for_occ and not ya_registrado_hoy:
                st.info(f"Esperando selección para {code_occ}...")

        except Exception as e:
            st.warning("Sin datos.")

# --- TABS ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["✨ Sugerencia", "🧺 Lavadero", "📦 Inventario", "➕ Nuevo Item", "📊 Estadísticas", "✈️ Viaje"])

with tab1:
    recs_df, temp_calculada = recommend_outfit(df, weather, code_occ, st.session_state['seed'])

    for cat_key, code_val in st.session_state['custom_overrides'].items():
        if code_val and code_val in df['Code'].values:
            manual_item = df[df['Code'] == code_val].iloc[0]
            if manual_item['Category'] in ['Remera', 'Camisa']: recs_df = recs_df[~recs_df['Category'].isin(['Remera', 'Camisa'])]
            elif manual_item['Category'] == 'Pantalón': recs_df = recs_df[recs_df['Category'] != 'Pantalón']
            elif manual_item['Category'] in ['Campera', 'Buzo']: recs_df = recs_df[~recs_df['Category'].isin(['Campera', 'Buzo'])]
            recs_df = pd.concat([recs_df, manual_item.to_frame().T], ignore_index=True)

    with st.container(border=True):
        col_w1, col_w2, col_w3 = st.columns(3)
        col_w1.metric("Clima", f"{weather['temp']}°C", weather['desc'])
        col_w2.metric("Sensación", f"{weather['feels_like']}°C")
        col_w3.metric("Tu Perfil", f"{temp_calculada:.1f}°C", "+3°C adj")

    col_h1, col_h2 = st.columns([2, 2])
    with col_h1: st.subheader("Tu Outfit")
    with col_h2: 
        c_btn1, c_btn2 = st.columns(2)
        if c_btn1.button("🔄 Cambiar", use_container_width=True): 
            st.session_state['change_mode'] = not st.session_state['change_mode']; st.session_state['custom_overrides'] = {}; st.rerun()
        if c_btn2.button("🛠️ Manual", use_container_width=True):
            st.session_state['show_custom_ui'] = not st.session_state.get('show_custom_ui', False)

    if st.session_state.get('show_custom_ui', False):
        with st.container(border=True):
            st.markdown("###### ✍️ Ingresá el código:")
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
                if health < 0.25: st.warning("⚠️ Lavar pronto")
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
                    save_feedback_entry_gsheet(entry); st.session_state['seed'] += 1; st.session_state['change_mode'] = False; st.rerun()
        else:
            if st.session_state['confirm_stage'] == 0:
                st.markdown("### ⭐ Calificación del día")
                
                # --- HELPER VISUAL: GRADIENTE ---
                def show_gradient_bar():
                    st.markdown('<div style="background: linear-gradient(90deg, #3b82f6 0%, #ffffff 50%, #ef4444 100%); height: 8px; border-radius: 4px; margin-bottom: 5px; opacity: 0.8;"></div>', unsafe_allow_html=True)

                # --- GLOBAL ---
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
                
                # --- TOP ---
                rt_abr, rt_com, rt_flow = 4, None, None # Default 4 (Perfecto)
                if rec_top and rec_top != "N/A":
                    st.markdown(f"**Top:** `{rec_top}`")
                    c_t1, c_t2, c_t3 = st.columns(3)
                    with c_t1: 
                        st.caption("Abrigo (pH)")
                        show_gradient_bar()
                        rt_abr = st.select_slider("Top Abrigo", options=[1, 2, 3, 4, 5, 6, 7], value=4, label_visibility="collapsed", key="s_top_a")
                    with c_t2: st.caption("Comodidad"); rt_com = st.feedback("stars", key="s_top_c")
                    with c_t3: st.caption("Flow"); rt_flow = st.feedback("stars", key="s_top_f")

                # --- BOTTOM ---
                rb_abr, rb_com, rb_flow = 4, None, None
                if rec_bot and rec_bot != "N/A":
                    st.markdown(f"**Bottom:** `{rec_bot}`")
                    c_b1, c_b2, c_b3 = st.columns(3)
                    with c_b1: 
                        st.caption("Abrigo (pH)")
                        show_gradient_bar()
                        rb_abr = st.select_slider("Bot Abrigo", options=[1, 2, 3, 4, 5, 6, 7], value=4, label_visibility="collapsed", key="s_bot_a")
                    with c_b2: st.caption("Comodidad"); rb_com = st.feedback("stars", key="s_bot_c")
                    with c_b3: st.caption("Flow"); rb_flow = st.feedback("stars", key="s_bot_f")
                
                # --- OUTER ---
                ro_abr, ro_com, ro_flow = 4, None, None
                if rec_out and rec_out != "N/A":
                    st.markdown(f"**Outer:** `{rec_out}`")
                    c_o1, c_o2, c_o3 = st.columns(3)
                    with c_o1: 
                        st.caption("Abrigo (pH)")
                        show_gradient_bar()
                        ro_abr = st.select_slider("Out Abrigo", options=[1, 2, 3, 4, 5, 6, 7], value=4, label_visibility="collapsed", key="s_out_a")
                    with c_o2: st.caption("Comodidad"); ro_com = st.feedback("stars", key="s_out_c")
                    with c_o3: st.caption("Flow"); ro_flow = st.feedback("stars", key="s_out_f")

                st.divider()

                # --- LOGICA DE BLOQUEO POR USO DIARIO ---
                if ya_registrado_hoy:
                    st.success(f"✅ Ya registraste un outfit para '{code_occ}' hoy.")
                    force_register = st.checkbox("🔓 Permitir nuevo registro (ej. cambio de ropa)")
                    if not force_register:
                        st.button("✅ Registrar Uso", disabled=True, key="btn_disabled")
                    else:
                        if st.button("✅ Registrar Uso (Forzado)", type="primary", use_container_width=True):
                            alerts = []
                            for item in selected_items_codes:
                                idx = df[df['Code'] == item['Code']].index[0]
                                sna = decodificar_sna(item['Code'])
                                limit = get_limit_for_item(item['Category'], sna)
                                current_uses = int(float(df.at[idx, 'Uses'])) if df.at[idx, 'Uses'] not in ['', 'nan'] else 0
                                if (current_uses + 1) > limit: alerts.append({'code': item['Code'], 'cat': item['Category'], 'uses': current_uses, 'limit': limit})
                            
                            if alerts: st.session_state['alerts_buffer'] = alerts; st.session_state['confirm_stage'] = 1; st.rerun()
                            else:
                                for item in selected_items_codes:
                                    idx = df[df['Code'] == item['Code']].index[0]
                                    curr = int(float(df.at[idx, 'Uses'])) if df.at[idx, 'Uses'] not in ['', 'nan'] else 0
                                    df.at[idx, 'Uses'] = curr + 1
                                    df.at[idx, 'LastWorn'] = datetime.now().strftime("%Y-%m-%d")
                                st.session_state['inventory'] = df; save_data_gsheet(df)
                                ra = r_abrigo
                                rc = r_comodidad + 1 if r_comodidad is not None else 3
                                rs = r_seguridad + 1 if r_seguridad is not None else 3
                                v_rt_a = rt_abr; v_rt_c = rt_com + 1 if rt_com is not None else 3; v_rt_f = rt_flow + 1 if rt_flow is not None else 3
                                v_rb_a = rb_abr; v_rb_c = rb_com + 1 if rb_com is not None else 3; v_rb_f = rb_flow + 1 if rb_flow is not None else 3
                                v_ro_a = ro_abr; v_ro_c = ro_com + 1 if ro_com is not None else 3; v_ro_f = ro_flow + 1 if ro_flow is not None else 3
                                st.session_state['custom_overrides'] = {} 
                                entry = {'Date': get_mendoza_time().strftime("%Y-%m-%d %H:%M"), 'City': user_city, 'Temp_Real': weather['temp'], 'User_Adj_Temp': temp_calculada, 'Occasion': code_occ, 'Top': rec_top, 'Bottom': rec_bot, 'Outer': rec_out, 'Rating_Abrigo': ra, 'Rating_Comodidad': rc, 'Rating_Seguridad': rs, 'Action': 'Accepted', 'Top_Abrigo': v_rt_a, 'Top_Comodidad': v_rt_c, 'Top_Flow': v_rt_f, 'Bot_Abrigo': v_rb_a, 'Bot_Comodidad': v_rb_c, 'Bot_Flow': v_rb_f, 'Out_Abrigo': v_ro_a, 'Out_Comodidad': v_ro_c, 'Out_Flow': v_ro_f}
                                save_feedback_entry_gsheet(entry); st.toast("¡Outfit registrado!"); st.rerun()
                else:
                    if st.button("✅ Registrar Uso", type="primary", use_container_width=True):
                        alerts = []
                        for item in selected_items_codes:
                            idx = df[df['Code'] == item['Code']].index[0]
                            sna = decodificar_sna(item['Code'])
                            limit = get_limit_for_item(item['Category'], sna)
                            current_uses = int(float(df.at[idx, 'Uses'])) if df.at[idx, 'Uses'] not in ['', 'nan'] else 0
                            if (current_uses + 1) > limit: alerts.append({'code': item['Code'], 'cat': item['Category'], 'uses': current_uses, 'limit': limit})
                        
                        if alerts: st.session_state['alerts_buffer'] = alerts; st.session_state['confirm_stage'] = 1; st.rerun()
                        else:
                            for item in selected_items_codes:
                                idx = df[df['Code'] == item['Code']].index[0]
                                curr = int(float(df.at[idx, 'Uses'])) if df.at[idx, 'Uses'] not in ['', 'nan'] else 0
                                df.at[idx, 'Uses'] = curr + 1
                                df.at[idx, 'LastWorn'] = datetime.now().strftime("%Y-%m-%d")

                            st.session_state['inventory'] = df; save_data_gsheet(df)
                            
                            ra = r_abrigo
                            rc = r_comodidad + 1 if r_comodidad is not None else 3
                            rs = r_seguridad + 1 if r_seguridad is not None else 3
                            
                            v_rt_a = rt_abr; v_rt_c = rt_com + 1 if rt_com is not None else 3; v_rt_f = rt_flow + 1 if rt_flow is not None else 3
                            v_rb_a = rb_abr; v_rb_c = rb_com + 1 if rb_com is not None else 3; v_rb_f = rb_flow + 1 if rb_flow is not None else 3
                            v_ro_a = ro_abr; v_ro_c = ro_com + 1 if ro_com is not None else 3; v_ro_f = ro_flow + 1 if ro_flow is not None else 3

                            st.session_state['custom_overrides'] = {} 
                            
                            entry = {
                                'Date': get_mendoza_time().strftime("%Y-%m-%d %H:%M"), 
                                'City': user_city, 
                                'Temp_Real': weather['temp'], 
                                'User_Adj_Temp': temp_calculada, 
                                'Occasion': code_occ, 
                                'Top': rec_top, 
                                'Bottom': rec_bot, 
                                'Outer': rec_out, 
                                'Rating_Abrigo': ra, 
                                'Rating_Comodidad': rc, 
                                'Rating_Seguridad': rs, 
                                'Action': 'Accepted',
                                'Top_Abrigo': v_rt_a, 'Top_Comodidad': v_rt_c, 'Top_Flow': v_rt_f,
                                'Bot_Abrigo': v_rb_a, 'Bot_Comodidad': v_rb_c, 'Bot_Flow': v_rb_f,
                                'Out_Abrigo': v_ro_a, 'Out_Comodidad': v_ro_c, 'Out_Flow': v_ro_f
                            }
                            
                            save_feedback_entry_gsheet(entry); st.toast("¡Outfit registrado!"); st.rerun()

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
                        df.at[idx, 'Uses'] = curr + 1; df.at[idx, 'LastWorn'] = datetime.now().strftime("%Y-%m-%d"); save_data_gsheet(df); st.session_state['confirm_stage'] = 0; st.session_state['alerts_buffer'] = []; st.rerun()
    else: st.error("No hay ropa limpia disponible.")

with tab2: 
    st.header("Lavadero")
    with st.container(border=True):
        col_input, col_btn = st.columns([3, 1])
        with col_input:
            with st.form("quick_wash_form", clear_on_submit=True):
                code_input = st.text_input("Ingresar Código")
                if st.form_submit_button("🧼 Lavar", use_container_width=True) and code_input:
                    code_clean = code_input.strip().upper()
                    if code_clean in df['Code'].values:
                        idx = df[df['Code'] == code_clean].index[0]
                        df.at[idx, 'Status'] = 'Lavando'; df.at[idx, 'Uses'] = 0; df.at[idx, 'LaundryStart'] = datetime.now().isoformat()
                        st.session_state['inventory'] = df; save_data_gsheet(df); st.success(f"✅ {code_clean} lavando."); st.rerun()
                    else: st.error("❌ Código no existe.")

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
        dirty_items = df[df['Status'].isin(['Sucio', 'Lavando'])]
        count_dirty = len(dirty_items)
        count_clean = total_items - count_dirty
        
        rate_dirty = count_dirty / total_items if total_items > 0 else 0
        
        st.caption("🧺 Estado del Lavadero")
        st.progress(rate_dirty, text=f"Suciedad: {int(rate_dirty*100)}% ({count_clean} Limpias | {count_dirty} Sucias)")
    
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
            # Si no tiene fecha, asumimos que es NUEVA y no la mostramos en "Muertas"
            if pd.isna(row['LastWorn']) or str(row['LastWorn']) in ['', 'nan', 'None']: return False
            try:
                last_date = datetime.fromisoformat(str(row['LastWorn']))
                if (datetime.now() - last_date).days > 90: return True
            except: return False
            return False

        dead_df = df[df.apply(is_dead_stock, axis=1)]
        if not dead_df.empty:
            st.dataframe(dead_df[['Category', 'Code']], hide_index=True, use_container_width=True)
        else:
            st.success("¡Rotación impecable!")

    st.divider()

    c_f1, c_f2 = st.columns(2)

    with c_f1:
        st.subheader("⭐ Ranking Flow")
        try:
            fb = load_feedback_gsheet()
            if not fb.empty and 'Action' in fb.columns:
                accepted = fb[fb['Action'] == 'Accepted'].copy()
                cols_rate = ['Rating_Abrigo', 'Rating_Comodidad', 'Rating_Seguridad']
                for c in cols_rate: accepted[c] = pd.to_numeric(accepted[c], errors='coerce').fillna(3)
                accepted['Score'] = accepted[cols_rate].mean(axis=1)
                
                melted = accepted.melt(id_vars=['Score'], value_vars=['Top', 'Bottom', 'Outer'], value_name='Code').dropna()
                melted = melted[~melted['Code'].isin(['N/A', 'nan', ''])]
                ranking = melted.groupby('Code')['Score'].mean().reset_index().sort_values(by='Score', ascending=False).head(5)
                ranking = ranking.merge(df[['Code', 'Category', 'ImageURL']], on='Code', how='left')
                
                st.dataframe(ranking[['Category', 'Score']], hide_index=True, use_container_width=True)
            else: st.info("Falta feedback.")
        except: st.error("Error en Flow.")

    with c_f2:
        st.subheader("📈 Tendencia Histórica")
        try:
            fb = load_feedback_gsheet()
            if not fb.empty:
                fb['Avg_Score'] = (pd.to_numeric(fb['Rating_Abrigo'], errors='coerce') + 
                                   pd.to_numeric(fb['Rating_Comodidad'], errors='coerce') + 
                                   pd.to_numeric(fb['Rating_Seguridad'], errors='coerce')) / 3
                fb['Day'] = fb['Date'].astype(str).str.slice(0, 10)
                daily_trend = fb.groupby('Day')['Avg_Score'].mean()
                st.line_chart(daily_trend)
        except: st.info("Sin datos.")

with tab6:
    st.header("✈️ Modo Viaje v2.0") 
    
    col_dest, col_days = st.columns([2, 1])
    with col_dest: dest_city = st.text_input("📍 Destino", value="Buenos Aires")
    with col_days: num_days = st.number_input("📅 Días", min_value=1, max_value=30, value=3)

    if st.button("🎒 Generar Propuesta de Valija", type="primary", use_container_width=True):
        packable = df[df['Status'] == 'Limpio']
        if packable.empty: st.error("¡No tenés ropa limpia para viajar!")
        else:
            n_tops = num_days + 1; n_bots = (num_days // 2) + 1; n_out = 2
            tops = packable[packable['Category'].isin(['Remera', 'Camisa'])]
            if len(tops) > n_tops: tops = tops.sample(n_tops)
            bots = packable[packable['Category'] == 'Pantalón']
            if len(bots) > n_bots: bots = bots.sample(n_bots)
            outs = packable[packable['Category'].isin(['Campera', 'Buzo'])]
            if len(outs) > n_out: outs = outs.sample(n_out)
            st.session_state['travel_pack'] = pd.concat([tops, bots, outs])
            st.session_state['travel_selections'] = {} 
            st.rerun() 

    if st.session_state.get('travel_pack') is not None:
        pack = st.session_state['travel_pack']
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
    with st.expander("📋 Checklist de Supervivencia (No olvidar)", expanded=False):
        essentials = ["DNI / Pasaporte", "Cargador", "Cepillo Dientes", "Desodorante", "Auriculares", "Medicamentos", "Lentes", "Billetera"]
        cols_ch = st.columns(2)
        for i, item in enumerate(essentials): cols_ch[i % 2].checkbox(item, key=f"check_{i}")
            
Que opinas de este codigo y app?
