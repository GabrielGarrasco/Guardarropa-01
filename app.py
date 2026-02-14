import streamlit as st
import pandas as pd
import requests
from datetime import datetime
import os

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="GDI: Guardarropa Digital Inteligente",
    page_icon="🧥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS para modo oscuro industrial
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #c9d1d9;
    }
    .stMetric {
        background-color: #161b22;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #30363d;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #30363d;
    }
    </style>
    """, unsafe_allow_html=True)

# --- GESTIÓN DE DATOS ---
FILE_INV = 'inventory.csv'
FILE_FEED = 'feedback.csv'

def load_data():
    if not os.path.exists(FILE_INV):
        # Crear estructura vacía si no existe
        df = pd.DataFrame(columns=['Code', 'Category', 'Season', 'Occasion', 'ImageURL', 'Status', 'LastWorn'])
        df.to_csv(FILE_INV, index=False)
        return df
    return pd.read_csv(FILE_INV)

def save_data(df):
    df.to_csv(FILE_INV, index=False)

def save_feedback(rating, sensation, temp):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_data = pd.DataFrame([[timestamp, rating, sensation, temp]], 
                            columns=['Timestamp', 'Rating', 'Sensation', 'Temperature'])
    
    if not os.path.exists(FILE_FEED):
        new_data.to_csv(FILE_FEED, index=False)
    else:
        new_data.to_csv(FILE_FEED, mode='a', header=False, index=False)

# --- MÓDULO CLIMA (OpenWeatherMap) ---
def get_weather(api_key):
    # Coordenadas Mendoza, AR
    lat, lon = -32.8908, -68.8272
    if not api_key:
        # MOCK DATA para demostración si no hay API Key
        return {"temp": 24, "min": 12, "max": 28, "desc": "Cielo despejado (Demo)"}
    
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric&lang=es"
        res = requests.get(url).json()
        return {
            "temp": res['main']['temp'],
            "min": res['main']['temp_min'],
            "max": res['main']['temp_max'],
            "desc": res['weather'][0]['description'].capitalize()
        }
    except:
        st.error("Error conectando a OpenWeatherMap. Usando datos demo.")
        return {"temp": 24, "min": 12, "max": 28, "desc": "Error API"}

# --- LÓGICA DE RECOMENDACIÓN (ENGINEERING LOGIC) ---
def recommend_outfit(df, weather, occasion):
    # 1. Filtro base: Solo ropa limpia
    clean_df = df[df['Status'] == 'Limpio'].copy()
    
    # 2. Lógica Térmica (Usuario Caluroso)
    # Ajuste: El usuario siente 3 grados más que la realidad
    feels_like = weather['temp'] + 3 
    
    target_season = []
    if feels_like > 25:
        target_season = ['V'] # Verano puro
    elif feels_like > 18:
        target_season = ['V', 'M'] # Verano o Media estación
    else:
        target_season = ['M', 'W'] # Media o Invierno
        
    # Filtrar por temporada y ocasión
    rec_df = clean_df[
        (clean_df['Season'].isin(target_season)) & 
        (clean_df['Occasion'] == occasion)
    ]
    
    return rec_df

# --- INTERFAZ PRINCIPAL ---

st.sidebar.title("🛠️ Panel de Control")
api_key = st.sidebar.text_input("OpenWeatherMap API Key", type="password")
user_occasion = st.sidebar.selectbox("Ocasión", ["U", "D", "C", "F"], format_func=lambda x: {"U": "Universidad", "D": "Deporte", "C": "Casa", "F": "Formal"}.get(x))

df = load_data()
weather = get_weather(api_key)
amplitude = weather['max'] - weather['min']

# TABS
tab1, tab2, tab3 = st.tabs(["📊 Dashboard & Outfit", "🧺 Gestión de Lavado", "➕ Carga de Items"])

with tab1:
    st.title("GDI: Mendoza Operations Center")
    
    # Métricas Clima
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Temperatura Actual", f"{weather['temp']}°C")
    c2.metric("Amplitud Térmica", f"{amplitude:.1f}°C")
    c3.metric("Condición", weather['desc'])
    
    # Alerta de Capas (Mendoza Logic)
    if amplitude > 15:
        c4.warning("⚠️ ALERTA ZONDA/AMPLITUD: Usar sistema de capas (Cebolla).")
    else:
        c4.success("✅ Clima estable.")

    st.markdown("---")
    st.subheader("💡 Sugerencia del Día")

    recommendations = recommend_outfit(df, weather, user_occasion)
    
    if recommendations.empty:
        st.error("No hay prendas limpias disponibles para este clima/ocasión. ¡A lavar!")
    else:
        col_top, col_bot = st.columns(2)
        
        # Separar partes de arriba y abajo basado en Código SNA (Tipo)
        # R=Remera, C=Campera, B=Buzo vs P=Pantalón, S=Short
        tops = recommendations[recommendations['Category'].isin(['Remera', 'Campera', 'Buzo'])]
        bottoms = recommendations[recommendations['Category'].isin(['Pantalón', 'Short'])]
        
        selected_top = tops.sample(1).iloc[0] if not tops.empty else None
        selected_bottom = bottoms.sample(1).iloc[0] if not bottoms.empty else None
        
        with col_top:
            if selected_top is not None:
                st.image(selected_top['ImageURL'], caption=f"Top: {selected_top['Category']} [{selected_top['Code']}]", use_column_width=True)
                st.code(selected_top['Code'])
        
        with col_bot:
            if selected_bottom is not None:
                st.image(selected_bottom['ImageURL'], caption=f"Bottom: {selected_bottom['Category']} [{selected_bottom['Code']}]", use_column_width=True)
                st.code(selected_bottom['Code'])

    # Feedback Loop
    st.markdown("---")
    with st.expander("📝 Feedback del Sistema"):
        with st.form("feedback_form"):
            stars = st.slider("Calificación del Outfit", 1, 5, 3)
            sensation = st.select_slider("Sensación Térmica Real", options=["Muy Frío", "Frío", "Perfecto", "Calor", "Muy Caluroso"], value="Perfecto")
            submitted = st.form_submit_button("Registrar Datos")
            if submitted:
                save_feedback(stars, sensation, weather['temp'])
                st.success("Datos guardados para reentrenamiento futuro.")

with tab2:
    st.subheader("Gestión de Ciclo de Vida (Lavado)")
    st.info("Edita el estado directamente en la tabla.")
    
    # Data Editor permite editar el CSV visualmente
    edited_df = st.data_editor(
        df,
        column_config={
            "ImageURL": st.column_config.ImageColumn("Preview"),
            "Status": st.column_config.SelectboxColumn(
                "Estado",
                options=["Limpio", "Sucio", "Lavando", "Para Doblar"],
                required=True
            )
        },
        hide_index=True,
        num_rows="dynamic"
    )
    
    if st.button("💾 Guardar Cambios de Estado"):
        save_data(edited_df)
        st.success("Inventario actualizado.")

with tab3:
    st.subheader("Ingreso de Nuevo Item (SNA)")
    
    col1, col2 = st.columns(2)
    with col1:
        c_temp = st.selectbox("Temporada", ["V", "W", "M"])
        c_type = st.selectbox("Tipo", ["R (Remera)", "P (Pantalón)", "C (Campera)", "B (Buzo)", "S (Short)"])
        c_len = st.selectbox("Largo", ["00", "01", "02"])
    
    with col2:
        c_occ = st.selectbox("Ocasión", ["U", "D", "C", "F"])
        # Aquí agregamos el selector de color según nuestra tabla
        c_col = st.selectbox("Color", ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"], 
                             format_func=lambda x: {
                                 "01": "Blanco", "02": "Negro", "03": "Rojo", "04": "Azul", 
                                 "05": "Gris", "06": "Verde", "07": "Am/Nar", 
                                 "08": "Mar/Beige", "09": "Estampado", "10": "Denim"
                             }.get(x))
        c_id = st.text_input("ID Individual (2 dígitos)", "01")
    
    c_url = st.text_input("URL Foto", "https://via.placeholder.com/150")
    
    type_map = {"R (Remera)": "R", "P (Pantalón)": "P", "C (Campera)": "C", "B (Buzo)": "B", "S (Short)": "S"}
    cat_map = {"R (Remera)": "Remera", "P (Pantalón)": "Pantalón", "C (Campera)": "Campera", "B (Buzo)": "Buzo", "S (Short)": "Short"}
    
    # Lógica de 9 dígitos: [T][Type][Len][Occ][Col][ID]
    generated_code = f"{c_temp}{type_map[c_type]}{c_len}{c_occ}{c_col}{c_id}"
    st.markdown(f"**Código Generado:** `{generated_code}`")
    
    if st.button("Agregar al Inventario"):
        new_row = pd.DataFrame([{
            'Code': generated_code,
            'Category': cat_map[c_type],
            'Season': c_temp,
            'Occasion': c_occ,
            'ImageURL': c_url,
            'Status': 'Limpio',
            'LastWorn': datetime.now().strftime("%Y-%m-%d")
        }])
        df = pd.concat([df, new_row], ignore_index=True)
        save_data(df)
        st.success(f"Item {generated_code} agregado correctamente.")
