import streamlit as st
import pandas as pd
import requests
from datetime import datetime
import os

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="GDI: Mendoza Ops", layout="wide")

FILE_INV = 'inventory.csv'

def load_data():
    if not os.path.exists(FILE_INV):
        return pd.DataFrame(columns=['Code', 'Category', 'Season', 'Occasion', 'ImageURL', 'Status', 'LastWorn'])
    return pd.read_csv(FILE_INV)

def save_data(df):
    df.to_csv(FILE_INV, index=False)

def get_weather(api_key):
    lat, lon = -32.8908, -68.8272
    if not api_key:
        return {"temp": 12, "min": 5, "max": 25, "desc": "Modo Demo"}
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric&lang=es"
        res = requests.get(url).json()
        return {"temp": res['main']['temp'], "min": res['main']['temp_min'], "max": res['main']['temp_max'], "desc": res['weather'][0]['description'].capitalize()}
    except:
        return {"temp": 12, "min": 5, "max": 21, "desc": "Error API"}

# --- LÓGICA DE ABRIGO E INGENIERÍA ---
def recommend_outfit(df, weather, occasion):
    clean_df = df[df['Status'] == 'Limpio'].copy()
    if clean_df.empty: return pd.DataFrame()

    temp = weather['temp']
    
    # El usuario es caluroso, ajustamos el umbral
    # Si hace 15°C, para él se siente como 18°C.
    adj_temp = temp + 3 

    # 1. Filtrar Tops por Nivel de Abrigo (Dígitos 3-4 del código)
    # C=Campera, B=Buzo. El "largo" ahora es el nivel de abrigo.
    def filter_tops(row):
        if row['Category'] not in ['Campera', 'Buzo']: return True
        level = int(row['Code'][2:4]) # Extrae el nivel 01, 02...
        if adj_temp < 10: return level >= 4
        if 10 <= adj_temp < 18: return level in [2, 3]
        if adj_temp >= 18: return level == 1
        return False

    rec_df = clean_df[clean_df['Occasion'] == occasion]
    # Aplicar lógica de abrigo solo a abrigos
    tops_abrigos = rec_df[rec_df['Category'].isin(['Campera', 'Buzo'])]
    tops_abrigos = tops_abrigos[tops_abrigos.apply(filter_tops, axis=1)]
    
    remeras_camisas = rec_df[rec_df['Category'].isin(['Remera', 'Camisa'])]
    pantalones = rec_df[rec_df['Category'] == 'Pantalón']

    return pd.concat([remeras_camisas, tops_abrigos, pantalones])

# --- INTERFAZ ---
st.sidebar.title("🛠️ GDI Control")
api_key = st.sidebar.text_input("API Key", type="password")
user_occ = st.sidebar.selectbox("Ocasión", ["U", "D", "C", "F"])

df = load_data()
weather = get_weather(api_key)

tab1, tab2, tab3 = st.tabs(["📊 Outfit", "🧺 Lavadero", "➕ Carga"])

with tab1:
    st.title(f"Mendoza: {weather['temp']}°C - {weather['desc']}")
    if (weather['max'] - weather['min']) > 15:
        st.warning("⚠️ Amplitud térmica alta detectada.")

    recs = recommend_outfit(df, weather, user_occ)
    if not recs.empty:
        cols = st.columns(3)
        # Mostrar una remera/camisa, un abrigo (si hace falta) y un pantalón
        for i, cat in enumerate(['Remera', 'Campera', 'Pantalón']):
            subset = recs[recs['Category'] == cat]
            if not subset.empty:
                item = subset.sample(1).iloc[0]
                cols[i].image(item['ImageURL'], caption=item['Code'])

with tab2:
    st.subheader("Estado de Prendas")
    edited_df = st.data_editor(df, hide_index=True)
    if st.button("Guardar Cambios"):
        save_data(edited_df)
        st.success("Guardado en sesión.")
    
    # BOTÓN PARA QUE NO PIERDAS TUS PRENDAS
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Descargar Inventario para GitHub", csv, "inventory.csv", "text/csv")

with tab3:
    st.subheader("Carga de Items")
    c_temp = st.selectbox("Temporada", ["V", "W", "M"])
    c_type_label = st.selectbox("Tipo", ["R (Remera)", "CS (Camisa)", "P (Pantalón)", "C (Campera)", "B (Buzo)"])
    type_code = c_type_label.split(" ")[0]

    # Lógica de Largo/Abrigo según pedido
    if type_code == "P":
        c_len = st.selectbox("Largo", ["Sh", "DL", "DC", "Je", "Ve"], format_func=lambda x: {"Sh":"Short", "DL":"Dep. Largo", "DC":"Dep. Corto", "Je":"Jean", "Ve":"Vestir"}.get(x))
    elif type_code in ["C", "B"]:
        c_len = st.selectbox("Abrigo", ["01", "02", "03", "04", "05"], format_func=lambda x: ["1-Rompevientos", "2-Fina", "3-Común", "4-Gruesa", "5-Muy Gruesa"][int(x)-1])
    else:
        c_len = st.selectbox("Manga", ["00", "01", "02"])

    c_col = st.selectbox("Color", ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"])
    c_url = st.text_input("URL Foto")
    
    generated_code = f"{c_temp}{type_code}{c_len}{user_occ}{c_col}01"
    st.code(f"Código: {generated_code}")
    
    if st.button("Añadir"):
        new_row = pd.DataFrame([{'Code': generated_code, 'Category': c_type_label.split(" ")[1].strip("()"), 'Season': c_temp, 'Occasion': user_occ, 'ImageURL': c_url, 'Status': 'Limpio', 'LastWorn': '2026-02-13'}])
        df = pd.concat([df, new_row], ignore_index=True)
        save_data(df)
        st.success("Añadido.")
