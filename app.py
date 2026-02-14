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

# --- LÍMITES DE USO ---
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
        return {
            "temp": res['main']['temp'],
            "feels_like": res['main']['feels_like'],
            "min": res['main']['temp_min'], 
            "max": res['main']['temp_max'], 
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
st.sidebar.title("GDI: Mendoza Ops")
api_key = st.sidebar.text_input("🔑 API Key", type="password")
user_city = st.sidebar.text_input("📍 Ciudad", value="Mendoza, AR")
user_occ = st.sidebar.selectbox("🎯 Ocasión", ["U (Universidad)", "D (Deporte)", "C (Casa)", "F (Formal)"])
code_occ = user_occ[0]

if 'inventory' not in st.session_state: st.session_state['inventory'] = load_data()
df = st.session_state['inventory']
weather = get_weather(api_key, user_city)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["✨ Sugerencia", "🧺 Lavadero", "📦 Inventario", "➕ Nuevo Item", "📊 Estadísticas"])

# --- TAB 1 A 4: (Lógica preservada de v8.4) ---
with tab1:
    # (Aquí va toda tu lógica de sugerencia, cambio de semilla y confirmación de la v8.4)
    st.info("Pestaña de sugerencia activa (Lógica v8.4)")
    # [Insertar aquí el bloque de Tab 1 enviado anteriormente]

with tab5:
    st.header("📊 GDI Analytics")
    
    # 1. Tasa de Lavado
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        total = len(df)
        dirty = len(df[df['Status'] != 'Limpio'])
        ratio = (dirty / total * 100) if total > 0 else 0
        st.metric("🧺 Tasa de Lavado", f"{ratio:.1f}%", f"{dirty} prendas sucias")
        st.progress(ratio/100)

    # 2. Top 5 Prendas (Caballitos de Batalla)
    st.subheader("🏆 Top 5 Prendas más usadas")
    top_5 = df.nlargest(5, 'Uses')[['Code', 'Category', 'Uses']]
    fig_top = px.bar(top_5, x='Code', y='Uses', color='Category', title="Uso acumulado por prenda")
    st.plotly_chart(fig_top, use_container_width=True)

    # 3. Prendas "Muertas" (Sin uso en 30 días o nunca usadas)
    st.subheader("💀 Prendas Muertas")
    # Asumimos que si LastWorn es muy viejo o Uses es 0, es candidata a salir
    dead_clothes = df[(df['Uses'] == 0) | (pd.to_datetime(df['LastWorn']) < (datetime.now() - timedelta(days=60)))]
    if not dead_clothes.empty:
        st.warning(f"Tenés {len(dead_clothes)} prendas que casi no usás. ¿Candidatas a donar?")
        st.dataframe(dead_clothes[['Code', 'Category', 'LastWorn', 'Uses']], hide_index=True)
    else:
        st.success("¡Usás todo tu armario! Gran eficiencia.")

    # 4. Gráfico de Satisfacción (Feedback)
    if os.path.exists(FILE_FEEDBACK):
        st.subheader("⭐ Evolución de Satisfacción")
        fb_df = pd.read_csv(FILE_FEEDBACK)
        fb_df['Date'] = pd.to_datetime(fb_df['Date'])
        # Promedio de las 3 métricas
        fb_df['Avg_Rating'] = fb_df[['Rating_Abrigo', 'Rating_Comodidad', 'Rating_Seguridad']].mean(axis=1)
        
        fig_line = px.line(fb_df, x='Date', y='Avg_Rating', title="Satisfacción promedio por día")
        st.plotly_chart(fig_line, use_container_width=True)
