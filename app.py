import streamlit as st
import pandas as pd
import requests
import os
import plotly.express as px
from datetime import datetime, timedelta

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="GDI: Mendoza Ops v9.0", layout="centered", page_icon="🧥")

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
        df = pd.DataFrame(columns=['Date', 'City', 'Temp_Real', 'User_Adj_Temp', 'Occasion', 'Top', 'Bottom', 'Outer', 'Rating_Abrigo', 'Rating_Comodidad', 'Rating_Seguridad', 'Action'])
    else:
        df = pd.read_csv(FILE_FEEDBACK)
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
            "temp": res['main']['temp'], "feels_like": res['main']['feels_like'],
            "min": res['main']['temp_min'], "max": res['main']['temp_max'], 
            "desc": res['weather'][0]['description'].capitalize()
        }
    except: return {"temp": 15, "feels_like": 14, "min": 10, "max": 20, "desc": "Error Conexión"}

def recommend_outfit(df, weather, occasion, seed):
    clean_df = df[df['Status'] == 'Limpio'].copy()
    if clean_df.empty: return pd.DataFrame(), 0
    t_act = weather.get('feels_like', weather['temp']) + 3 
    t_max = weather.get('max', weather['temp']) + 3
    t_min = weather.get('min', weather['temp']) + 3
    recs = []
    for _, row in clean_df.iterrows():
        sna = decodificar_sna(row['Code'])
        if not sna or sna['occasion'] != occasion: continue
        m = False
        if row['Category'] == 'Pantalón':
            if t_max > 28: m = sna['attr'] in ['Sh', 'DC']
            else: m = True
        elif row['Category'] in ['Remera', 'Camisa']:
            if t_max > 30: m = sna['attr'] in ['00', '01']
            else: m = True
        elif row['Category'] in ['Campera', 'Buzo']:
            lvl = int(sna['attr'])
            if t_min < 12: m = lvl >= 4
            elif t_min < 18: m = lvl in [2, 3]
            elif t_min < 22: m = lvl == 1
        if m: recs.append(row)
    return pd.DataFrame(recs), t_act

# --- INTERFAZ ---
st.sidebar.title("GDI: Mendoza Ops v9.0")
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

tab1, tab2, tab3, tab4, tab5 = st.tabs(["✨ Sugerencia", "🧺 Lavadero", "📦 Inventario", "➕ Nuevo Item", "📊 Estadísticas"])

with tab1:
    recs_df, temp_calculada = recommend_outfit(df, weather, code_occ, st.session_state['seed'])
    with st.container(border=True):
        c_w1, c_w2, c_w3 = st.columns(3)
        c_w1.metric("Clima", f"{weather['temp']}°C", weather['desc'])
        c_w2.metric("Sensación", f"{weather['feels_like']}°C")
        c_w3.metric("Tu Perfil", f"{temp_calculada:.1f}°C", "+3°C adj")

    col_h1, col_h2 = st.columns([3, 1])
    with col_h1: st.subheader("Outfit Recomendado")
    with col_h2: 
        if st.button("🔄 Cambiar"): st.session_state['change_mode'] = not st.session_state['change_mode']

    rec_top, rec_bot, rec_out = None, None, None
    selected_items = []

    if not recs_df.empty:
        c1, c2, c3 = st.columns(3)
        base = recs_df[recs_df['Category'].isin(['Remera', 'Camisa'])]
        legs = recs_df[recs_df['Category'] == 'Pantalón']
        outr = recs_df[recs_df['Category'].isin(['Campera', 'Buzo'])]

        with c1:
            st.markdown("###### Torso")
            if not base.empty:
                item = base.sample(1, random_state=st.session_state['seed']).iloc[0]
                rec_top = item['Code']; selected_items.append(item)
                st.info(f"**{item['Category']}**\n\nCode: `{item['Code']}`")
                if pd.notna(item['ImageURL']) and item['ImageURL']: st.image(item['ImageURL'])
        with c2:
            st.markdown("###### Piernas")
            if not legs.empty:
                item = legs.sample(1, random_state=st.session_state['seed']).iloc[0]
                rec_bot = item['Code']; selected_items.append(item)
                st.success(f"**{item['Category']}**\n\nCode: `{item['Code']}`")
                if pd.notna(item['ImageURL']) and item['ImageURL']: st.image(item['ImageURL'])
        with c3:
            st.markdown("###### Abrigo")
            if not outr.empty:
                item = outr.sample(1, random_state=st.session_state['seed']).iloc[0]
                rec_out = item['Code']; selected_items.append(item)
                st.warning(f"**{item['Category']}**\n\nCode: `{item['Code']}`")
                if pd.notna(item['ImageURL']) and item['ImageURL']: st.image(item['ImageURL'])
            else: st.write("✅ No necesario"); rec_out = "N/A"

        st.divider()

        if st.session_state['change_mode']:
            st.info("¿Qué no te convenció?")
            with st.container(border=True):
                cf1, cf2, cf3 = st.columns(3)
                with cf1: n_abr = st.feedback("stars", key="neg_abr")
                with cf2: n_com = st.feedback("stars", key="neg_com")
                with cf3: n_seg = st.feedback("stars", key="neg_seg")
                if st.button("🎲 Nueva Sugerencia"):
                    save_feedback_entry({'Date': datetime.now().strftime("%Y-%m-%d %H:%M"), 'Top': rec_top, 'Bottom': rec_bot, 'Outer': rec_out, 'Rating_Abrigo': n_abr+1 if n_abr else 3, 'Action': 'Rejected'})
                    st.session_state['seed'] += 1; st.session_state['change_mode'] = False; st.rerun()
        else:
            if st.session_state['confirm_stage'] == 0:
                st.markdown("### ⭐ Calificación")
                cf1, cf2, cf3 = st.columns(3)
                with cf1: r_abr = st.feedback("stars", key="fb_abr")
                with cf2: r_com = st.feedback("stars", key="fb_com")
                with cf3: r_seg = st.feedback("stars", key="fb_seg")
                if st.button("✅ Registrar Uso", type="primary", use_container_width=True):
                    alerts = []
                    for it in selected_items:
                        idx = df[df['Code'] == it['Code']].index[0]
                        sna = decodificar_sna(it['Code'])
                        lim = LIMITES_USO.get(sna['attr'] if it['Category']=='Pantalón' else sna['tipo'], 3)
                        if (int(df.at[idx, 'Uses']) + 1) > lim: alerts.append(it)
                    if alerts: st.session_state['alerts_buffer'] = alerts; st.session_state['confirm_stage'] = 1; st.rerun()
                    else:
                        for it in selected_items: df.loc[df['Code']==it['Code'], 'Uses'] += 1
                        save_data(df); save_feedback_entry({'Date': datetime.now().strftime("%Y-%m-%d %H:%M"), 'Rating_Abrigo': r_abr+1 if r_abr else 3, 'Rating_Comodidad': r_com+1 if r_com else 3, 'Rating_Seguridad': r_seg+1 if r_seg else 3, 'Action': 'Accepted'})
                        st.toast("¡Outfit guardado!"); st.rerun()
            elif st.session_state['confirm_stage'] == 1:
                st.error("🚨 ¡Límite de uso!")
                for al in st.session_state['alerts_buffer']:
                    with st.container(border=True):
                        st.write(f"**{al['Category']} ({al['Code']})**")
                        cw1, cw2 = st.columns(2)
                        if cw1.button("🧼 Lavar", key=f"w_{al['Code']}"):
                            df.loc[df['Code']==al['Code'], ['Status', 'Uses']] = ['Sucio', 0]; save_data(df); st.rerun()
                        if cw2.button("👟 Usar igual", key=f"k_{al['Code']}"):
                            df.loc[df['Code']==al['Code'], 'Uses'] += 1; save_data(df); st.session_state['confirm_stage'] = 0; st.rerun()
    else: st.error("No hay ropa limpia.")

with tab2: # Lavadero
    st.subheader("🧺 Gestión de Lavado")
    edit_lav = st.data_editor(df[['Code', 'Category', 'Status', 'Uses']], key="ed_lav", column_config={"Status": st.column_config.SelectboxColumn("Estado", options=["Limpio", "Sucio", "Lavando"], required=True)}, hide_index=True, disabled=["Code", "Category", "Uses"])
    if st.button("🔄 Actualizar"):
        df.update(edit_lav)
        for idx in df.index:
            if df.at[idx, 'Status'] in ['Lavando', 'Sucio']: df.at[idx, 'Uses'] = 0
        save_data(df); st.success("¡Listo!")

with tab3: # Inventario
    st.subheader("📦 Inventario Total")
    edit_inv = st.data_editor(df, num_rows="dynamic", use_container_width=True, column_config={"Uses": st.column_config.ProgressColumn("Desgaste", min_value=0, max_value=10, format="%d")})
    if st.button("💾 Guardar Inventario"): save_data(edit_inv); st.toast("Guardado")

with tab4: # Nuevo Item
    st.subheader("🏷️ Alta de Prenda")
    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            tmp = st.selectbox("Temporada", ["V", "W", "M"])
            tpf = st.selectbox("Tipo", ["R - Remera", "CS - Camisa", "P - Pantalón", "C - Campera", "B - Buzo"])
            t_c = tpf.split(" - ")[0]
            if t_c == "P": atr = st.selectbox("Corte", ["Je", "Sh", "DL", "DC", "Ve"])
            elif t_c in ["C", "B"]: atr = f"0{st.selectbox('Abrigo', ['1', '2', '3', '4', '5'])}"
            else: atr = st.selectbox("Manga", ["00", "01", "02"])
        with c2:
            occ = st.selectbox("Ocasión", ["U", "D", "C", "F"])
            col = st.selectbox("Color", ["01-Blanco", "02-Negro", "03-Gris", "04-Azul", "05-Verde", "06-Rojo", "07-Amarillo", "08-Beige", "09-Marron", "10-Denim"])[:2]
            url = st.text_input("URL Foto")
        if st.button("Agregar"):
            code = f"{tmp}{t_c}{atr}{occ}{col}{len(df)+1:02d}"
            new = pd.DataFrame([{'Code': code, 'Category': tpf.split(" - ")[1], 'Season': tmp, 'Occasion': occ, 'ImageURL': url, 'Status': 'Limpio', 'LastWorn': datetime.now().strftime("%Y-%m-%d"), 'Uses': 0}])
            save_data(pd.concat([df, new], ignore_index=True)); st.success(f"Agregado: {code}")

with tab5:
    st.header("📊 GDI Analytics")
    if not df.empty:
        c_s1, c_s2 = st.columns(2)
        with c_s1:
            ratio = (len(df[df['Status'] != 'Limpio']) / len(df) * 100)
            st.metric("🧺 Tasa de Lavado", f"{ratio:.1f}%")
            st.progress(ratio/100)
        with c_s2:
            st.metric("👕 Total Prendas", len(df))
        
        st.subheader("🏆 Top 5 más usadas")
        st.plotly_chart(px.bar(df.nlargest(5, 'Uses'), x='Code', y='Uses', color='Category'), use_container_width=True)
        
        st.subheader("💀 Prendas Muertas")
        dead = df[(df['Uses'] == 0) | (pd.to_datetime(df['LastWorn']) < (datetime.now() - timedelta(days=60)))]
        st.dataframe(dead[['Code', 'Category', 'LastWorn', 'Uses']], hide_index=True)
        
        if os.path.exists(FILE_FEEDBACK):
            st.subheader("⭐ Satisfacción")
            fb = pd.read_csv(FILE_FEEDBACK)
            fb['Avg'] = fb[['Rating_Abrigo', 'Rating_Comodidad', 'Rating_Seguridad']].mean(axis=1)
            st.plotly_chart(px.line(fb, x=fb.index, y='Avg'), use_container_width=True)
