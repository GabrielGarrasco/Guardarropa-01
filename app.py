import streamlit as st
import pandas as pd
import requests
import os
from datetime import datetime

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="GDI: Mendoza Ops v8.0", layout="centered", page_icon="🧥")

FILE_INV = 'inventory.csv'
FILE_FEEDBACK = 'feedback.csv'

# --- LÍMITES DE USO (INGENIERÍA DE CICLO DE VIDA) ---
# Claves basadas en el atributo SNA o Categoría general
LIMITES_USO = {
    # Pantalones (por Atributo)
    "Je": 6, "Ve": 4, "DL": 3, "DC": 2, "Sh": 1,
    # Tops (por Tipo)
    "R": 2, "CS": 3, 
    # Abrigos (por Tipo)
    "B": 5, "C": 10
}

# --- FUNCIONES AUXILIARES ---
def decodificar_sna(codigo):
    """Parsea el código SNA."""
    try:
        codigo = str(codigo).strip()
        if len(codigo) < 4: return None
        
        if codigo[1:3] == 'CS':
            tipo = 'CS'
            idx_start_attr = 3
        else:
            tipo = codigo[1]
            idx_start_attr = 2
            
        attr = codigo[idx_start_attr : idx_start_attr + 2]
        idx_occ = idx_start_attr + 2
        
        if idx_occ < len(codigo):
            occasion = codigo[idx_occ]
        else:
            occasion = "C"
            
        return {"tipo": tipo, "attr": attr, "occasion": occasion}
    except:
        return None

def load_data():
    if not os.path.exists(FILE_INV):
        return pd.DataFrame(columns=['Code', 'Category', 'Season', 'Occasion', 'ImageURL', 'Status', 'LastWorn', 'Uses'])
    
    df = pd.read_csv(FILE_INV)
    df['Code'] = df['Code'].astype(str)
    
    # Migración V8: Asegurar que existe columna Uses
    if 'Uses' not in df.columns:
        df['Uses'] = 0
    
    return df

def save_data(df):
    df.to_csv(FILE_INV, index=False)

def save_feedback_entry(entry):
    if not os.path.exists(FILE_FEEDBACK):
        df = pd.DataFrame(columns=['Date', 'City', 'Temp_Real', 'Feels_Like', 'User_Adj_Temp', 'Occasion', 'Top', 'Bottom', 'Outer', 'Rating_Abrigo', 'Rating_Comodidad', 'Rating_Seguridad'])
    else:
        df = pd.read_csv(FILE_FEEDBACK)
    new_row = pd.DataFrame([entry])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(FILE_FEEDBACK, index=False)

def get_weather(api_key, city="Mendoza, AR"):
    if not city: city = "Mendoza, AR"
    if not api_key:
        return {"temp": 24, "feels_like": 22, "min": 18, "max": 30, "desc": "Modo Demo (Sin API)"}
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric&lang=es"
        res = requests.get(url).json()
        if res.get("cod") != 200:
             return {"temp": 0, "feels_like": 0, "min": 0, "max": 0, "desc": f"Error: {res.get('message')}"}
        return {
            "temp": res['main']['temp'],
            "feels_like": res['main']['feels_like'],
            "min": res['main']['temp_min'], 
            "max": res['main']['temp_max'], 
            "desc": res['weather'][0]['description'].capitalize()
        }
    except:
        return {"temp": 15, "feels_like": 14, "min": 10, "max": 20, "desc": "Error Conexión"}

# --- LÓGICA DE RECOMENDACIÓN ---
def recommend_outfit(df, weather, occasion):
    clean_df = df[df['Status'] == 'Limpio'].copy()
    if clean_df.empty: return pd.DataFrame(), 0

    st_meteo = weather.get('feels_like', weather['temp'])
    temp_decision = st_meteo + 3 # Tu perfil caluroso
    
    recs = []
    
    for index, row in clean_df.iterrows():
        sna = decodificar_sna(row['Code'])
        if not sna: continue
        if sna['occasion'] != occasion: continue

        # Reglas de Negocio
        if row['Category'] == 'Pantalón':
            tipo_pan = sna['attr']
            if temp_decision > 26 and tipo_pan in ['Sh', 'DC']: recs.append(row)
            elif temp_decision > 20 and tipo_pan in ['Je', 'Ve', 'DL', 'Sh']: recs.append(row)
            elif temp_decision <= 20 and tipo_pan in ['Je', 'Ve', 'DL']: recs.append(row)

        elif row['Category'] in ['Remera', 'Camisa']:
            manga = sna['attr']
            if temp_decision > 25 and manga in ['00', '01']: recs.append(row)
            elif temp_decision < 15 and manga in ['02', '01']: recs.append(row)
            else: recs.append(row)

        elif row['Category'] in ['Campera', 'Buzo']:
            try:
                nivel = int(sna['attr'])
                if temp_decision < 12 and nivel >= 4: recs.append(row)
                elif temp_decision < 18 and nivel in [2, 3]: recs.append(row)
                elif temp_decision < 24 and nivel == 1: recs.append(row)
            except:
                continue

    return pd.DataFrame(recs), temp_decision

# --- INTERFAZ ---
st.sidebar.title("GDI: Mendoza Ops")
st.sidebar.markdown("---")
api_key = st.sidebar.text_input("🔑 API Key OpenWeather", type="password")
user_city = st.sidebar.text_input("📍 Ciudad", value="Mendoza, AR")
user_occ = st.sidebar.selectbox("🎯 Ocasión", ["U (Universidad)", "D (Deporte)", "C (Casa)", "F (Formal)"])
code_occ = user_occ[0]

if 'inventory' not in st.session_state:
    st.session_state['inventory'] = load_data()
df = st.session_state['inventory']

# Inicialización de estado para flujo de confirmación
if 'confirm_stage' not in st.session_state: st.session_state['confirm_stage'] = 0 # 0: Nada, 1: Validando Limites, 2: Feedback
if 'alerts_buffer' not in st.session_state: st.session_state['alerts_buffer'] = [] # Para guardar prendas en conflicto

weather = get_weather(api_key, user_city)

tab1, tab2, tab3, tab4 = st.tabs(["✨ Sugerencia", "🧺 Lavadero", "📦 Inventario", "➕ Nuevo Item"])

# --- TAB 1: SUGERENCIA INTELIGENTE ---
with tab1:
    recs_df, temp_calculada = recommend_outfit(df, weather, code_occ)

    # 1. Header Clima (Estética v7.0)
    with st.container(border=True):
        col_w1, col_w2, col_w3 = st.columns(3)
        col_w1.metric("Clima", f"{weather['temp']}°C", weather['desc'])
        col_w2.metric("Sensación", f"{weather['feels_like']}°C")
        col_w3.metric("Tu Perfil", f"{temp_calculada:.1f}°C", "+3°C adj")
        if (weather['max'] - weather['min']) > 15:
            st.warning("⚠️ ¡Alerta de amplitud térmica! (Zonda/Mendoza). Llevá capas.", icon="🌬️")

    # Selección de prendas
    rec_top, rec_bot, rec_out = None, None, None
    selected_items_codes = []

    if not recs_df.empty:
        st.subheader("Outfit Recomendado")
        c1, c2, c3 = st.columns(3)
        
        base = recs_df[recs_df['Category'].isin(['Remera', 'Camisa'])]
        legs = recs_df[recs_df['Category'] == 'Pantalón']
        outer = recs_df[recs_df['Category'].isin(['Campera', 'Buzo'])]

        # Cards Visuales
        with c1:
            st.markdown("###### Torso")
            if not base.empty:
                item = base.sample(1).iloc[0]
                rec_top = item['Code']
                selected_items_codes.append(item)
                st.info(f"**{item['Category']}**\n\nCode: `{item['Code']}`")
                if pd.notna(item['ImageURL']) and item['ImageURL']: st.image(item['ImageURL'], use_column_width=True)
            else: st.write("🚫 Sin opciones")

        with c2:
            st.markdown("###### Piernas")
            if not legs.empty:
                item = legs.sample(1).iloc[0]
                rec_bot = item['Code']
                selected_items_codes.append(item)
                st.success(f"**{item['Category']}**\n\nCode: `{item['Code']}`")
                if pd.notna(item['ImageURL']) and item['ImageURL']: st.image(item['ImageURL'], use_column_width=True)
            else: st.write("🚫 Sin opciones")

        with c3:
            st.markdown("###### Abrigo")
            if not outer.empty:
                item = outer.sample(1).iloc[0]
                rec_out = item['Code']
                selected_items_codes.append(item)
                st.warning(f"**{item['Category']}**\n\nCode: `{item['Code']}`")
                if pd.notna(item['ImageURL']) and item['ImageURL']: st.image(item['ImageURL'], use_column_width=True)
            else:
                st.write("✅ No necesario")
                rec_out = "N/A"

        st.divider()

        # --- LÓGICA DE CICLO DE VIDA (NUEVO V8.0) ---
        
        # Etapa 0: Botón de Confirmación
        if st.session_state['confirm_stage'] == 0:
            if st.button("✅ Confirmar y Usar Outfit", type="primary", use_container_width=True):
                # 1. Incrementar Usos
                alerts = []
                for item in selected_items_codes:
                    idx = df[df['Code'] == item['Code']].index[0]
                    df.at[idx, 'Uses'] = int(df.at[idx, 'Uses']) + 1
                    df.at[idx, 'LastWorn'] = datetime.now().strftime("%Y-%m-%d")
                    
                    # 2. Chequear Límites
                    sna = decodificar_sna(item['Code'])
                    limit = 99 # Default infinito
                    if sna:
                        if item['Category'] == 'Pantalón': limit = LIMITES_USO.get(sna['attr'], 3)
                        elif item['Category'] in ['Remera', 'Camisa']: limit = LIMITES_USO.get(sna['tipo'], 2)
                        elif item['Category'] in ['Campera', 'Buzo']: limit = LIMITES_USO.get(sna['tipo'], 5)
                    
                    if df.at[idx, 'Uses'] >= limit:
                        alerts.append({'code': item['Code'], 'cat': item['Category'], 'uses': df.at[idx, 'Uses'], 'limit': limit})
                
                st.session_state['inventory'] = df
                save_data(df)
                
                if alerts:
                    st.session_state['alerts_buffer'] = alerts
                    st.session_state['confirm_stage'] = 1 # Hay conflictos
                    st.rerun()
                else:
                    st.session_state['confirm_stage'] = 2 # Todo OK, pasar a estrellas
                    st.rerun()

        # Etapa 1: Resolución de Conflictos (Límite Excedido)
        if st.session_state['confirm_stage'] == 1:
            st.error("🚨 ¡Atención! Algunas prendas alcanzaron su límite de uso.")
            
            alerts = st.session_state['alerts_buffer']
            remaining_alerts = []
            
            for alert in alerts:
                with st.container(border=True):
                    c_alert1, c_alert2 = st.columns([3, 2])
                    with c_alert1:
                        st.markdown(f"**{alert['cat']} ({alert['code']})** tiene **{alert['uses']}** usos. (Límite: {alert['limit']})")
                    with c_alert2:
                        col_btn1, col_btn2 = st.columns(2)
                        if col_btn1.button("🧼 A Lavar", key=f"wash_{alert['code']}"):
                            # Lógica Lavar
                            idx = df[df['Code'] == alert['code']].index[0]
                            df.at[idx, 'Status'] = 'Sucio'
                            df.at[idx, 'Uses'] = 0
                            save_data(df)
                            st.toast(f"{alert['code']} enviado al canasto.", icon="🧺")
                            # Remover de la lista de alertas visuales en prox rerun
                            # (Se hace implicitamente al no volver a guardarlo, pero necesitamos limpiar el buffer actual para salir del loop)
                        
                        elif col_btn2.button("👌 Usable", key=f"keep_{alert['code']}"):
                            st.toast("Prenda mantenida. ¡Ojo la próxima!", icon="👀")
                        
                        else:
                            # Si no se tocó botón, mantener en la lista para el próximo frame
                            remaining_alerts.append(alert)
            
            # Si el usuario interactuó (los botones reinician el script), verificamos si quedan alertas
            # Nota: Streamlit es stateless. Al hacer click en un botón arriba, se ejecuta, guarda data y hace rerun.
            # Necesitamos un botón "Continuar" final si quedan dudas, o detectar si se resolvieron.
            
            if st.button("Continuar a Calificación"):
                st.session_state['confirm_stage'] = 2
                st.session_state['alerts_buffer'] = []
                st.rerun()

        # Etapa 2: Feedback (Estrellas)
        if st.session_state['confirm_stage'] == 2:
            st.success("✅ Outfit registrado. ¡Disfrutalo!")
            
            st.markdown("### ⭐ Calificalo")
            c_fb1, c_fb2, c_fb3 = st.columns(3)
            with c_fb1:
                st.write("🌡️ Abrigo")
                r_abrigo = st.feedback("stars", key="fb_abrigo")
            with c_fb2:
                st.write("😌 Comodidad")
                r_comodidad = st.feedback("stars", key="fb_comodidad")
            with c_fb3:
                st.write("😎 Estilo")
                r_seguridad = st.feedback("stars", key="fb_estilo")

            if st.button("Guardar Feedback", type="primary"):
                ra = r_abrigo + 1 if r_abrigo is not None else 3
                rc = r_comodidad + 1 if r_comodidad is not None else 3
                rs = r_seguridad + 1 if r_seguridad is not None else 3
                
                entry = {
                    'Date': datetime.now().strftime("%Y-%m-%d %H:%M"),
                    'City': user_city,
                    'Temp_Real': weather['temp'],
                    'Feels_Like': weather['feels_like'],
                    'User_Adj_Temp': temp_calculada,
                    'Occasion': code_occ,
                    'Top': rec_top, 'Bottom': rec_bot, 'Outer': rec_out,
                    'Rating_Abrigo': ra, 'Rating_Comodidad': rc, 'Rating_Seguridad': rs
                }
                save_feedback_entry(entry)
                st.session_state['confirm_stage'] = 0 # Reiniciar ciclo
                st.toast("Feedback guardado exitosamente!", icon="🎉")
                # Opcional: st.rerun() para limpiar pantalla

    else:
        st.error("No encontré ropa limpia adecuada.")

# --- TAB 2: LAVADERO ---
with tab2:
    st.subheader("🧺 Gestión de Lavado")
    
    col_l1, col_l2 = st.columns([2, 1])
    with col_l1:
        st.info("Marcá lo que pusiste a lavar.")
        columnas_lavado = ['Code', 'Category', 'Status', 'Uses']
        edited_laundry = st.data_editor(
            df[columnas_lavado], 
            key="editor_lavadero",
            column_config={
                "Status": st.column_config.SelectboxColumn("Estado", options=["Limpio", "Sucio", "Lavando"], required=True),
                "Uses": st.column_config.NumberColumn("Usos Act.", disabled=True)
            },
            hide_index=True,
            disabled=["Code", "Category", "Uses"],
            use_container_width=True
        )
        if st.button("🔄 Actualizar Estados"):
            df.update(edited_laundry)
            # Si alguien pone 'Limpio' o 'Lavando', resetear usos?
            # Por seguridad, si pasa a 'Lavando', Uses -> 0
            for idx in df.index:
                if df.at[idx, 'Status'] in ['Lavando', 'Sucio'] and df.at[idx, 'Status'] != st.session_state['inventory'].at[idx, 'Status']:
                     df.at[idx, 'Uses'] = 0
            
            st.session_state['inventory'] = df
            save_data(df)
            st.success("¡Estados actualizados!")
            
    with col_l2:
        st.write("---")
        st.markdown("**Carga Rápida a Lavar**")
        code_to_wash = st.text_input("Ingresá Código")
        if st.button("Mandar a Lavar"):
            mask = df['Code'] == code_to_wash
            if mask.any():
                df.loc[mask, 'Status'] = 'Lavando'
                df.loc[mask, 'Uses'] = 0
                st.session_state['inventory'] = df
                save_data(df)
                st.success(f"{code_to_wash} -> Lavando")
            else:
                st.error("Código no encontrado")

# --- TAB 3: INVENTARIO GENERAL ---
with tab3:
    st.subheader("📋 Inventario Completo")
    st.markdown("Monitor de Ciclo de Vida activo.")
    
    edited_inventory = st.data_editor(
        df, 
        key="editor_inventario", 
        num_rows="dynamic",
        column_config={
             "Uses": st.column_config.ProgressColumn("Desgaste/Uso", min_value=0, max_value=10, format="%d")
        },
        use_container_width=True
    )
    
    if st.button("💾 Guardar Cambios Inventario"):
        st.session_state['inventory'] = edited_inventory
        save_data(edited_inventory)
        st.toast("Base de datos actualizada", icon="💾")
        
    st.download_button("📥 Backup CSV", df.to_csv(index=False).encode('utf-8'), "gdi_backup.csv")

# --- TAB 4: CARGA MANUAL ---
with tab4:
    st.subheader("🏷️ Alta de Prenda (SNA Gen)")
    with st.container(border=True):
        col_a, col_b = st.columns(2)
        with col_a:
            c_temp = st.selectbox("Temporada", ["V (Verano)", "W (Invierno)", "M (Media)"]).split(" ")[0]
            c_type_full = st.selectbox("Tipo", ["R - Remera", "CS - Camisa", "P - Pantalón", "C - Campera", "B - Buzo"])
            type_map = {"R - Remera": "R", "CS - Camisa": "CS", "P - Pantalón": "P", "C - Campera": "C", "B - Buzo": "B"}
            type_code = type_map[c_type_full]
            category_name = c_type_full.split(" - ")[1]
            
            if type_code == "P": 
                c_attr = st.selectbox("Corte", ["Je (Jean)", "Sh (Short)", "DL (Deportivo Largo)", "DC (Dep. Corto)", "Ve (Vestir)"]).split(" ")[0]
            elif type_code in ["C", "B"]: 
                c_attr = f"0{st.selectbox('Nivel Abrigo (1-5)', ['1 (Liviano)', '2', '3', '4', '5 (Pesado)']).split(' ')[0]}"
            else: 
                c_attr = st.selectbox("Manga", ["00 (Musculosa)", "01 (Corta)", "02 (Larga)"]).split(" ")[0]

        with col_b:
            c_occ = st.selectbox("Ocasión Uso", ["U (Universidad)", "D (Deporte)", "C (Casa)", "F (Formal)"]).split(" ")[0]
            # Lista de 10+ colores preservada
            colors_list = [
                "01-Blanco", "02-Negro", "03-Gris", "04-Azul", "05-Verde", 
                "06-Rojo", "07-Amarillo", "08-Beige", "09-Marron", "10-Denim", 
                "11-Naranja", "12-Violeta", "99-Estampado"
            ]
            c_col = st.selectbox("Color Principal", colors_list)[:2]
            c_url = st.text_input("URL Foto (Opcional)")

        prefix = f"{c_temp}{type_code}{c_attr}{c_occ}{c_col}"
        count = len([c for c in df['Code'] if str(c).startswith(prefix)])
        final_code = f"{prefix}{count + 1:02d}"
        
        st.markdown(f"### Código Generado: `{final_code}`")
        
        if st.button("Agregar al Guardarropa", type="primary"):
            new_row = pd.DataFrame([{
                'Code': final_code, 'Category': category_name, 'Season': c_temp, 
                'Occasion': c_occ, 'ImageURL': c_url, 'Status': 'Limpio', 
                'LastWorn': datetime.now().strftime("%Y-%m-%d"), 'Uses': 0
            }])
            updated_df = pd.concat([df, new_row], ignore_index=True)
            st.session_state['inventory'] = updated_df
            save_data(updated_df)
            st.success(f"¡Agregado! {final_code}")
