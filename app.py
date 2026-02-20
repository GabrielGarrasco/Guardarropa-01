import streamlit as st
import pandas as pd
import requests
import os
import pytz
import json
from datetime import datetime, timedelta, date
from PIL import Image, ImageOps
from io import BytesIO
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import random
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import calendar
import altair as alt
import colorsys
import telebot
import schedule
import time
import threading

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="GDI: Mendoza Ops v21.0", layout="centered", page_icon="🧥")

# ==========================================
# --- CONFIGURACIÓN DE SECRETOS Y BOT ---
# ==========================================
try:
    TELEGRAM_TOKEN = st.secrets["TELEGRAM_TOKEN"]
    TELEGRAM_CHAT_ID = st.secrets["TELEGRAM_CHAT_ID"]
    bot = telebot.TeleBot(TELEGRAM_TOKEN) 
except Exception as e:
    st.error(f"Error cargando secretos o iniciando Bot: {e}")
    bot = None 

# ==========================================
# --- MOTOR DE INTELIGENCIA ARTIFICIAL V3.0 ---
# ==========================================
class OutfitAI:
    def __init__(self):
        self.model = None
        self.encoders = {}
        self.is_trained = False
        self.co_occurrence_matrix = {}

    def _extract_features(self, temp, occasion, code, wind=0, humidity=50, date_obj=None):
        sna = decodificar_sna(code)
        
        weekday = 0
        if date_obj:
            try: weekday = date_obj.weekday()
            except: pass
        else:
            weekday = datetime.now().weekday()

        if not sna:
            return {'Temp': temp, 'Occasion': occasion, 'Season': 'X', 'Attr': '00', 'Color': '99', 'Wind': wind, 'Hum': humidity, 'Day': weekday}
        
        return {
            'Temp': float(temp),
            'Occasion': str(occasion),
            'Season': sna['season'],
            'Attr': sna['attr'],
            'Color': sna['color'],
            'Wind': float(wind),
            'Hum': float(humidity),
            'Day': int(weekday)
        }

    def train(self, feedback_df, inventory_df):
        if feedback_df.empty or len(feedback_df) < 5: return False 
        try:
            data = feedback_df.copy()
            data['Temp_Real'] = pd.to_numeric(data['Temp_Real'], errors='coerce').fillna(20)
            
            def calculate_weighted_score(row):
                if row['Action'] == 'Rejected': return 0
                r_abr = float(row.get('Rating_Abrigo', 4))
                r_com = float(row.get('Rating_Comodidad', 3))
                r_seg = float(row.get('Rating_Seguridad', 3))
                temp = float(row.get('Temp_Real', 20))
                
                w_abr = 2.0 if (temp < 15 or temp > 30) else 1.0
                w_seg = 1.5 if row['Occasion'] in ['F', 'U'] else 1.0
                w_com = 1.0
                
                total_w = w_abr + w_seg + w_com
                weighted_avg = ((r_abr * w_abr) + (r_seg * w_seg) + (r_com * w_com)) / total_w
                return (weighted_avg / 7) * 100

            data['Target_Score'] = data.apply(calculate_weighted_score, axis=1)

            self.co_occurrence_matrix = {}
            high_rated = data[data['Target_Score'] > 70]
            for _, row in high_rated.iterrows():
                if row['Top'] and row['Bottom']:
                    pair_key = f"{row['Top']}_{row['Bottom']}"
                    self.co_occurrence_matrix[pair_key] = self.co_occurrence_matrix.get(pair_key, 0) + 1

            training_rows = []
            for _, row in data.iterrows():
                for part in ['Top', 'Bottom', 'Outer']:
                    code = row[part]
                    if code and code not in ['N/A', 'nan', 'None', '']:
                        w = row.get('Wind', 0)
                        h = row.get('Humidity', 50)
                        try: d_obj = datetime.strptime(str(row['Date']), "%Y-%m-%d %H:%M")
                        except: d_obj = None
                        
                        features = self._extract_features(row['Temp_Real'], row['Occasion'], code, w, h, d_obj)
                        features['Score'] = row['Target_Score']
                        training_rows.append(features)
            
            df_train = pd.DataFrame(training_rows)
            if df_train.empty: return False

            cat_cols = ['Occasion', 'Season', 'Attr', 'Color']
            X = df_train[['Temp', 'Wind', 'Hum', 'Day'] + cat_cols].copy()
            y = df_train['Score']

            for col in cat_cols:
                le = LabelEncoder()
                le.fit(list(X[col].unique()) + ['Unknown'])
                self.encoders[col] = le
                X[col] = le.transform(X[col])

            self.model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
            self.model.fit(X, y)
            self.is_trained = True
            return True
        except Exception as e:
            return False

    def predict_score(self, item_row, current_temp, occasion_code, partner_code=None, wind=0, humidity=50):
        if not self.is_trained: return 50.0 
        try:
            features = self._extract_features(current_temp, occasion_code, item_row['Code'], wind, humidity)
            
            input_vector = [features['Temp'], features['Wind'], features['Hum'], features['Day']]
            for col in ['Occasion', 'Season', 'Attr', 'Color']:
                val = features[col]
                if col in self.encoders:
                    if val not in self.encoders[col].classes_: val = 'Unknown'
                    input_vector.append(self.encoders[col].transform([val])[0])
                else:
                    input_vector.append(0)
            
            input_np = np.array([input_vector])
            predicted_score = self.model.predict(input_np)[0]

            uses = int(float(item_row['Uses'])) if item_row['Uses'] not in ['', 'nan'] else 0
            if uses > 2: predicted_score -= 15 

            if partner_code:
                pair_1 = f"{item_row['Code']}_{partner_code}"
                pair_2 = f"{partner_code}_{item_row['Code']}"
                if pair_1 in self.co_occurrence_matrix or pair_2 in self.co_occurrence_matrix:
                    predicted_score += 10

            return predicted_score
        except: return 50.0

if 'outfit_ai' not in st.session_state: st.session_state['outfit_ai'] = OutfitAI()

# ==========================================
# --- CONEXIÓN A GOOGLE SHEETS ---
# ==========================================
def get_google_sheet_client():
    try:
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds_dict = dict(st.secrets["service_account"])
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        return client
    except: return None

def load_data_gsheet():
    client = get_google_sheet_client()
    if not client: return pd.DataFrame(columns=['Code', 'Category', 'Season', 'Occasion', 'ImageURL', 'Status', 'LastWorn', 'Uses', 'LaundryStart'])
    try:
        sheet = client.open("GDI_Database").worksheet("inventory")
        data = sheet.get_all_records()
        if not data: return pd.DataFrame(columns=['Code', 'Category', 'Season', 'Occasion', 'ImageURL', 'Status', 'LastWorn', 'Uses', 'LaundryStart'])
        
        df = pd.DataFrame(data).astype(str)
        
        # --- AUTO-LIMPIEZA DE 5 HORAS ---
        changed = False
        now = datetime.now()
        for idx, row in df.iterrows():
            if row.get('Status') == 'Lavando' and row.get('LaundryStart') not in ['', 'nan', 'None']:
                try:
                    start_time = datetime.fromisoformat(row['LaundryStart'])
                    if now - start_time >= timedelta(hours=5):
                        df.at[idx, 'Status'] = 'Limpio'
                        df.at[idx, 'LaundryStart'] = ''
                        df.at[idx, 'Uses'] = '0'
                        changed = True
                except: pass
        
        if changed:
            save_data_gsheet(df)
            
        return df
    except: return pd.DataFrame(columns=['Code', 'Category', 'Season', 'Occasion', 'ImageURL', 'Status', 'LastWorn', 'Uses', 'LaundryStart'])

def save_data_gsheet(df):
    client = get_google_sheet_client()
    if not client: 
        st.error("No se pudo conectar a Google Sheets.")
        return
    try:
        sheet = client.open("GDI_Database").worksheet("inventory")
        sheet.clear()
        df_str = df.astype(str)
        datos = [df_str.columns.values.tolist()] + df_str.values.tolist()
        sheet.update(values=datos, range_name="A1")
    except Exception as e:
        st.error(f"Error escribiendo en Inventory: {e}")

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
    if not client: 
        st.error("No se pudo conectar a Google Sheets.")
        st.stop()
        return
    try:
        sheet = client.open("GDI_Database").worksheet("feedback")
        records = sheet.get_all_records()
        
        if records:
            df = pd.DataFrame(records)
            df.replace('', pd.NA, inplace=True)
            df.dropna(how='all', inplace=True)
        else:
            df = pd.DataFrame(columns=entry.keys())
            
        new_row = pd.DataFrame([entry])
        df = pd.concat([df, new_row], ignore_index=True)
        df.fillna('', inplace=True)
        
        sheet.clear()
        df_str = df.astype(str)
        datos = [df_str.columns.values.tolist()] + df_str.values.tolist()
        sheet.update(values=datos, range_name="A1")
        
    except Exception as e:
        st.error(f"Error escribiendo en Feedback: {e}")
        st.stop()

# ==========================================
# --- NUEVO: CARGA Y GUARDADO DE MODO VIAJE ---
# ==========================================
def load_travel_gsheet():
    client = get_google_sheet_client()
    if not client: return {}
    try:
        sheet = client.open("GDI_Database").worksheet("travel")
        data = sheet.get_all_records()
        if data and 'state_json' in data[0]:
            state = json.loads(data[0]['state_json'])
            if 'travel_pack' in state and state['travel_pack']:
                state['travel_pack'] = pd.DataFrame(state['travel_pack'])
            return state
        return {}
    except: return {}

def save_travel_gsheet(state_dict):
    client = get_google_sheet_client()
    if not client: return
    try:
        sheet = client.open("GDI_Database").worksheet("travel")
        state_copy = {
            'active_trip': state_dict.get('active_trip', False),
            'trip_current_day': state_dict.get('trip_current_day', 0),
            'travel_selections': state_dict.get('travel_selections', {})
        }
        if 'travel_pack' in state_dict and isinstance(state_dict.get('travel_pack'), pd.DataFrame):
            state_copy['travel_pack'] = state_dict['travel_pack'].to_dict(orient='records')
        
        sheet.clear()
        sheet.update(values=[["state_json"], [json.dumps(state_copy)]], range_name="A1")
    except Exception as e:
        st.error(f"Error guardando viaje en la nube: {e}")

# --- CONSTANTES ---
LIMITES_USO = {"R": 2, "Sh": 2, "DC": 2, "Je": 4, "B": 4, "CS": 1, "Ve": 2, "DL": 2, "C": 5}

COLOR_MAP = {
    "01": "#F5F5F5", "02": "#1A1A1A", "03": "#808080", "04": "#0000CD",
    "05": "#228B22", "06": "#B22222", "07": "#FFD700", "08": "#F5F5DC",
    "09": "#8B4513", "10": "#4682B4", "11": "#FF8C00", "12": "#8A2BE2",
    "99": "#FF69B4" 
}
COLOR_NAMES = {
    "01": "Blanco", "02": "Negro", "03": "Gris", "04": "Azul", "05": "Verde",
    "06": "Rojo", "07": "Amarillo", "08": "Beige", "09": "Marron", "10": "Denim",
    "11": "Naranja", "12": "Violeta", "99": "Estampado"
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

def estimar_color_dominante(image):
    try:
        img_small = image.resize((1, 1))
        color = img_small.getpixel((0, 0))
        return '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])
    except: return "#000000"

def decodificar_sna(codigo):
    try:
        c = str(codigo).strip()
        if len(c) < 4: return None
        if c.startswith('Ri'): return None 
        season = c[0]
        if len(c) > 2 and c[1:3] == 'CS': 
            tipo = 'CS' 
            offset = 3
        else: 
            tipo = c[1] 
            offset = 2
        
        attr = c[offset:offset+2]
        color_code = "99" 
        try:
            occ_idx = offset + 2
            col_idx = occ_idx + 1
            if len(c) >= col_idx + 2:
                color_code = c[col_idx:col_idx+2]
        except: pass
        
        return {"season": season, "tipo": tipo, "attr": attr, "color": color_code}
    except: return None

def get_limit_for_item(category, sna):
    if category == 'Ropa interior': return 9999
    if not sna: return 3
    if category == 'Pantalón': return LIMITES_USO.get(sna['attr'], 2)
    elif category in ['Remera', 'Camisa']: return LIMITES_USO.get(sna['tipo'], 1)
    return LIMITES_USO.get(sna['tipo'], 3)

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def calcular_armonia_color(hex1, hex2):
    r1, g1, b1 = hex_to_rgb(hex1)
    r2, g2, b2 = hex_to_rgb(hex2)
    h1, s1, v1 = colorsys.rgb_to_hsv(r1/255, g1/255, b1/255)
    h2, s2, v2 = colorsys.rgb_to_hsv(r2/255, g2/255, b2/255)
    
    diff_h = abs(h1 - h2) * 360
    if diff_h > 180: diff_h = 360 - diff_h
    
    if diff_h < 15: return 1.2
    if 15 <= diff_h <= 45: return 1.1 
    if 160 <= diff_h <= 200: return 1.3 
    if 110 <= diff_h <= 130: return 1.1
    if v1 < 0.2 and v2 < 0.2: return 0.8
    return 1.0

def check_harmony(code_top, code_bot):
    s_top = decodificar_sna(code_top)
    s_bot = decodificar_sna(code_bot)
    if not s_top or not s_bot: return True
    
    hex_t = COLOR_MAP.get(s_top.get('color', '99'), "#000000")
    hex_b = COLOR_MAP.get(s_bot.get('color', '99'), "#000000")
    
    neutros = ['01', '02', '03', '99']
    if s_top.get('color') in neutros or s_bot.get('color') in neutros: return True
        
    factor = calcular_armonia_color(hex_t, hex_b)
    return factor >= 0.9

def get_weather_open_meteo():
    try:
        url = "https://api.open-meteo.com/v1/forecast?latitude=-32.8908&longitude=-68.8272&current=temperature_2m,apparent_temperature,weather_code,wind_speed_10m,relative_humidity_2m&daily=temperature_2m_max,temperature_2m_min&hourly=temperature_2m&timezone=auto"
        res = requests.get(url).json()
        if 'current' not in res: return {"temp": 15, "feels_like": 14, "min": 10, "max": 20, "desc": "Error API", "hourly_temp": [], "hourly_time": [], "wind": 0, "humidity": 50}
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
            "hourly_time": hourly.get('time', []),
            "wind": current.get('wind_speed_10m', 0),
            "humidity": current.get('relative_humidity_2m', 50),
            "weather_code": code
        }
    except: return {"temp": 15, "feels_like": 14, "min": 10, "max": 20, "desc": "Error Conexión", "hourly_temp": [], "hourly_time": [], "wind": 0, "humidity": 50, "weather_code": 0}

def get_city_coords(city_name):
    try:
        url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=1&language=es&format=json"
        res = requests.get(url, timeout=3).json()
        if 'results' in res and res['results']: return res['results'][0]['latitude'], res['results'][0]['longitude'], res['results'][0]['country']
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

def create_outfit_canvas(top_code, bot_code, out_code, df_inv):
    try:
        imgs = []
        codes = [c for c in [top_code, bot_code, out_code] if c and c not in ['N/A', 'nan']]
        
        for c in codes:
            row = df_inv[df_inv['Code'] == c]
            if not row.empty:
                url = row.iloc[0]['ImageURL']
                pil_img = cargar_imagen_desde_url(url)
                if pil_img: imgs.append(pil_img)
        
        if not imgs: return None

        min_width = 300
        resized_imgs = []
        for i in imgs:
            w_percent = (min_width / float(i.size[0]))
            h_size = int((float(i.size[1]) * float(w_percent)))
            resized_imgs.append(i.resize((min_width, h_size), Image.Resampling.LANCZOS))
        
        total_height = sum([i.size[1] for i in resized_imgs])
        canvas = Image.new('RGB', (min_width, total_height), (255, 255, 255))
        
        y_offset = 0
        for i in resized_imgs:
            canvas.paste(i, (0, y_offset))
            y_offset += i.size[1]
            
        return canvas
    except: return None

def enviar_foto_telegram(caption, image):
    try:
        bio = BytesIO()
        image.save(bio, format='PNG')
        bio.seek(0)
        bot.send_photo(TELEGRAM_CHAT_ID, photo=bio, caption=caption, parse_mode='Markdown')
        return True, "Mensaje enviado con éxito"
    except Exception as e:
        return False, f"Error: {str(e)}"

def enviar_briefing_diario():
    try:
        if 'inventory' in st.session_state and not st.session_state['inventory'].empty:
            df_inv = st.session_state['inventory']
        else:
            df_inv = load_data_gsheet()
            
        if df_inv.empty: return False, "Error: No se pudo cargar el inventario."

        w_data = get_weather_open_meteo()
        
        rain_warning = ""
        try:
            url_rain = "https://api.open-meteo.com/v1/forecast?latitude=-32.8908&longitude=-68.8272&hourly=precipitation_probability&timezone=auto&forecast_days=1"
            res_rain = requests.get(url_rain).json()
            probs = res_rain['hourly']['precipitation_probability']
            times = res_rain['hourly']['time']
            now_hour = datetime.now().hour
            rain_hits = []
            for i in range(now_hour, min(now_hour + 12, len(probs))):
                if probs[i] > 30:
                    rain_hits.append(f"{datetime.fromisoformat(times[i]).strftime('%H:%M')} ({probs[i]}%)")
            if rain_hits: rain_warning = f"\n☔ *ALERTA LLUVIA:* Probabilidad alta: {', '.join(rain_hits)}"
            else: rain_warning = "\n✅ Sin lluvias próximas."
        except: rain_warning = ""

        coat_schedule = ""
        hourly_temps = w_data.get('hourly_temp', [])
        hourly_times = w_data.get('hourly_time', [])
        cold_hours = []
        now = datetime.now()
        for t, time_str in zip(hourly_temps, hourly_times):
            dt = datetime.fromisoformat(time_str)
            if dt.day == now.day and dt.hour >= now.hour:
                if t < 18: cold_hours.append(dt.hour)
        if cold_hours:
            coat_schedule = f"\n🧥 *ABRIGO:* Necesario entre {min(cold_hours)}:00 y {max(cold_hours)}:00 hs."
        else:
            coat_schedule = "\n👕 Clima agradable."

        laundry_msg = ""
        dirty_count = len(df_inv[df_inv['Status'] == 'Sucio'])
        clean_shirts = len(df_inv[(df_inv['Category'].isin(['Remera', 'Camisa'])) & (df_inv['Status'] == 'Limpio')])
        if dirty_count > 5: laundry_msg += f"\n🧺 *LAVANDERÍA:* {dirty_count} prendas sucias acumuladas."
        if clean_shirts < 3: laundry_msg += f"\n⚠️ *URGENTE:* Solo quedan {clean_shirts} remeras limpias."

        occ_brief = 'U' 
        recs, _, _ = recommend_outfit(df_inv, w_data, occ_brief, random.randint(1, 10000))
        
        if recs.empty: return False, "Falta ropa para generar outfit."

        r_top = recs[recs['Category'].isin(['Remera', 'Camisa'])].iloc[0]['Code'] if not recs[recs['Category'].isin(['Remera', 'Camisa'])].empty else None
        r_bot = recs[recs['Category'] == 'Pantalón'].iloc[0]['Code'] if not recs[recs['Category'] == 'Pantalón'].empty else None
        r_out = recs[recs['Category'].isin(['Campera', 'Buzo'])].iloc[0]['Code'] if not recs[recs['Category'].isin(['Campera', 'Buzo'])].empty else None
        
        canvas = create_outfit_canvas(r_top, r_bot, r_out, df_inv)
        if not canvas: return False, "Error generando imagen."

        mensaje = (
            f"🚀 *GDI AUTO-BRIEFING*\n"
            f"📅 {datetime.now().strftime('%d/%m %H:%M')}\n\n"
            f"🌡️ {w_data['temp']}°C (ST {w_data['feels_like']}°C) - {w_data['desc']}\n"
            f"{rain_warning}{coat_schedule}\n\n"
            f"👔 *Propuesta ({occ_brief}):*\n• {r_top}\n• {r_bot}\n• {r_out}"
            f"{laundry_msg}"
        )
        return enviar_foto_telegram(mensaje, canvas)

    except Exception as e:
        return False, f"Error loop: {str(e)}"

def calculate_smart_score(item_row, current_temp, occasion, feedback_df, weather_data=None, partner_code=None):
    ai = st.session_state.get('outfit_ai')
    score = 50.0
    wind = weather_data.get('wind', 0) if weather_data else 0
    hum = weather_data.get('humidity', 50) if weather_data else 50
    
    if ai and ai.is_trained:
        score = ai.predict_score(item_row, current_temp, occasion, partner_code, wind, hum)
    else:
        item_code = item_row['Code']
        if not feedback_df.empty:
            hist = feedback_df[feedback_df['Top'] == item_code]
            if not hist.empty:
                try: score = (pd.to_numeric(hist['Rating_Seguridad'], errors='coerce').mean() / 5) * 100
                except: pass

    try:
        if item_row['LastWorn'] not in ['', 'nan', 'None']:
            last_date = datetime.strptime(str(item_row['LastWorn']), "%Y-%m-%d").date()
            today = get_mendoza_time().date()
            days_diff = (today - last_date).days
            
            if days_diff <= 2: score -= 40
            elif days_diff <= 5: score -= 15
            else: score += (days_diff * 0.5)
    except: pass

    if weather_data:
        is_sunny = weather_data.get('weather_code', 0) <= 3 
        sna = decodificar_sna(item_row['Code'])
        
        if sna:
            color_code = sna.get('color', '99')
            attr = sna.get('attr', '00')
            
            is_dark = color_code in ['02', '04', '09', '12']
            if is_sunny:
                if current_temp < 18 and is_dark: score += 10
                elif current_temp > 28 and is_dark: score -= 15
                elif current_temp > 28 and not is_dark: score += 10

            if wind > 20: 
                if item_row['Category'] in ['Campera', 'Buzo']:
                    if attr == '01': score += 20
                    elif attr in ['03', '04']: score -= 10
                
                if item_row['Category'] == 'Pantalón' and attr in ['Sh', 'DC']:
                    score -= 20

    return score

def is_item_usable(row):
    if 'Archived' in str(row['Status']): return False
    if row['Status'] != 'Limpio': return False
    if row['Category'] == 'Ropa interior': return False 
    sna = decodificar_sna(row['Code'])
    if not sna: return True
    limit = get_limit_for_item(row['Category'], sna)
    try:
        uses = int(float(row['Uses'])) if row['Uses'] not in ['', 'nan'] else 0
        if uses >= limit: return False
    except: pass
    return True

def is_needs_wash(row):
    if 'Archived' in str(row['Status']): return False
    if row['Status'] in ['Sucio', 'Lavando']: return True
    sna = decodificar_sna(row['Code'])
    if not sna: return False
    limit = get_limit_for_item(row['Category'], sna)
    try:
        uses = int(float(row['Uses'])) if row['Uses'] not in ['', 'nan'] else 0
        return uses >= limit
    except: return False

def get_thermal_offset(feedback_df):
    if feedback_df.empty: return 3 
    try:
        avg_abrigo = pd.to_numeric(feedback_df['Rating_Abrigo'], errors='coerce').mean()
        if pd.isna(avg_abrigo): return 3
        custom_offset = 3 + (4 - avg_abrigo)
        return custom_offset
    except: return 3

def recommend_outfit(df, weather, occasion, seed):
    usable_df = df[df.apply(is_item_usable, axis=1)].copy()
    if usable_df.empty: return pd.DataFrame(), 0, ""
    
    blacklist = set()
    fb = pd.DataFrame()
    try:
        fb = load_feedback_gsheet()
        if not fb.empty:
            fb['DateObj'] = pd.to_datetime(fb['Date'], errors='coerce').dt.date
            today_obj = get_mendoza_time().date()
            rej = fb[(fb['DateObj'] == today_obj) & (fb['Action'].astype(str).str.strip() == 'Rejected')]
            blacklist = set(rej['Top'].dropna().tolist() + rej['Bottom'].dropna().tolist() + rej['Outer'].dropna().tolist())
    except: pass
    
    if 'temp_blacklist' in st.session_state:
        blacklist.update(st.session_state['temp_blacklist'])

    if 'agenda_reserves' in st.session_state:
        today_date = get_mendoza_time().date()
        for res_date, codes in st.session_state['agenda_reserves'].items():
            if res_date > today_date:
                blacklist.update(codes)

    personal_offset = get_thermal_offset(fb)
    t_curr = weather['temp']
    t_max = weather['max']
    t_min = weather['min']
    t_feel = weather.get('feels_like', t_curr) + personal_offset 
    
    w_speed = weather.get('wind', 0)
    w_hum = weather.get('humidity', 50)

    final = []
    
    coat_msg = ""
    needs_coat = False
    hourly_temps = weather.get('hourly_temp', [])
    hourly_times = weather.get('hourly_time', [])
    UMBRAL_FRIO = 18 
    
    if hourly_temps and hourly_times:
        hours_cold = []
        now_date = get_mendoza_time().date()
        for t, time_str in zip(hourly_temps, hourly_times):
            try:
                dt_hour = datetime.fromisoformat(time_str)
                if dt_hour.date() == now_date and t < UMBRAL_FRIO: hours_cold.append(dt_hour.hour)
            except: pass
        if hours_cold:
            needs_coat = True
            if len(hours_cold) >= 12: coat_msg = "❄️ Usar abrigo todo el día."
            else: coat_msg = f"🕒 Abrigo necesario de {min(hours_cold)}:00 a {max(hours_cold)}:00 hs"
        else:
            coat_msg = "☀️ No hace falta abrigo hoy."
            needs_coat = False

    if w_speed > 30 and w_hum < 30:
        coat_msg += " ⚠️ ALERTA ZONDA: Viento y tierra."

    target_occs = [occasion]
    if occasion in ['F', 'U']:
        target_occs = ['F', 'U'] 

    def get_best(cats, category_type, selected_partner_code=None):
        curr_s = get_current_season()
        pool = usable_df[(usable_df['Category'].isin(cats)) & (usable_df['Occasion'].isin(target_occs)) & ((usable_df['Season'] == curr_s) | (usable_df['Season'] == 'T'))]
        if pool.empty: pool = usable_df[(usable_df['Category'].isin(cats)) & (usable_df['Occasion'].isin(target_occs))]
        
        if pool.empty: return None
        
        cands = []
        for _, r in pool.iterrows():
            sna = decodificar_sna(r['Code'])
            if not sna: continue
            
            if selected_partner_code:
                if not check_harmony(r['Code'], selected_partner_code):
                    pass 

            match = False
            if category_type == 'bot':
                attr = sna['attr']
                if t_max > 27: match = attr in ['Sh', 'DC', 'Ve']
                elif t_max < 15: match = attr in ['Je', 'DL', 'Ve']
                else: match = True 
            elif category_type == 'top':
                attr = sna['attr']
                if t_max > 30: match = attr in ['00', '01']
                elif t_max < 18: match = attr == '02'
                else: match = True
            elif category_type == 'out':
                if not needs_coat: match = False 
                else:
                    try:
                        lvl = int(sna['attr'])
                        match = (t_min < 10 and lvl >= 3) or (t_min < 16 and lvl in [2, 3]) or (t_min < 22 and lvl == 1)
                    except: match = False
            if match: cands.append(r)
        
        f_pool = pd.DataFrame(cands) if cands else pool
        nb = f_pool[~f_pool['Code'].isin(blacklist)]
        candidates_df = nb if not nb.empty else f_pool
        if candidates_df.empty: return None

        try:
            candidates_df = candidates_df.copy()
            
            candidates_df['AI_Score'] = candidates_df.apply(
                lambda x: calculate_smart_score(x, t_curr, occasion, fb, weather, selected_partner_code), 
                axis=1
            )

            exploration_mode = random.random() < 0.1 

            if exploration_mode:
                candidates_df['Exploration_Boost'] = (10 - pd.to_numeric(candidates_df['Uses'], errors='coerce').fillna(0).clip(upper=10)) * 5
                candidates_df['Final_Score'] = candidates_df['AI_Score'] + candidates_df['Exploration_Boost']
            else:
                candidates_df['Color_Harmony'] = 0
                if selected_partner_code:
                      candidates_df['Color_Harmony'] = candidates_df.apply(
                        lambda x: 15 if check_harmony(x['Code'], selected_partner_code) else -15, 
                        axis=1
                    )
                
                candidates_df['Final_Score'] = candidates_df['AI_Score'] + candidates_df['Color_Harmony'] + candidates_df.apply(lambda x: random.uniform(-1, 1), axis=1)

            return candidates_df.sort_values('Final_Score', ascending=False).iloc[0]
        except: return candidates_df.sample(1, random_state=seed).iloc[0]

    bot_item = get_best(['Pantalón'], 'bot') 
    top = None
    if bot_item is not None:
        final.append(bot_item)
        top = get_best(['Remera', 'Camisa'], 'top', selected_partner_code=bot_item['Code'])
    else:
        top = get_best(['Remera', 'Camisa'], 'top')
    
    if top is not None: final.append(top)
    
    if needs_coat:
        out = get_best(['Campera', 'Buzo'], 'out') 
        if out is not None: final.append(out)
        
    return pd.DataFrame(final), t_feel, coat_msg

def calcular_peso_prenda(cat, code):
    code_str = str(code)
    if cat == 'Ropa interior':
        if code_str.startswith('RiBr'): return 71
        if code_str.startswith('RiB'): return 71
        if code_str.startswith('RiM'): return 59 
        return 50
    if cat == 'Pantalón':
        if 'Je' in code_str: return 760
        if 'Sh' in code_str or 'DC' in code_str: return 350
        return 450
    if cat in ['Remera', 'Camisa']:
        if '02' in code_str: return 230
        return 170
    if cat in ['Buzo', 'Campera']:
        if '04' in code_str or '05' in code_str: return 1000
        return 500
    return 300

# ==========================================
# --- INICIALIZACIÓN Y ESTADOS ---
# ==========================================
if 'travel_loaded' not in st.session_state:
    travel_data = load_travel_gsheet()
    if travel_data:
        st.session_state.update(travel_data)
    st.session_state['travel_loaded'] = True

# ==========================================
# --- INTERFAZ PRINCIPAL ---
# ==========================================
st.sidebar.title("GDI: Mendoza Ops")
st.sidebar.caption("v21.1 - Telegram Ops")

user_city = st.sidebar.text_input("📍 Ciudad", value="Mendoza, AR")
user_occ = st.sidebar.selectbox("🎯 Ocasión", ["U (Universidad)", "D (Deporte)", "C (Casa)", "F (Formal)"])
code_occ = user_occ[0]

st.sidebar.markdown("---")
st.sidebar.markdown("###### 🧢 Hoy llevas puesto:")
if 'inventory' not in st.session_state: 
    with st.spinner("Cargando sistema..."): st.session_state['inventory'] = load_data_gsheet()
df = st.session_state['inventory']

try:
    fb_side = load_feedback_gsheet()
    found_today = False
    
    if not fb_side.empty and 'Action' in fb_side.columns:
        fb_side['DateObj'] = pd.to_datetime(fb_side['Date'], errors='coerce').dt.date
        today_obj = get_mendoza_time().date()
        match_today = fb_side[(fb_side['DateObj'] == today_obj) & (fb_side['Action'].astype(str).str.strip() == 'Accepted')]
        
        if not match_today.empty:
            found_today = True
            match_today = match_today.sort_values('Date')
            last_fit = match_today.iloc[-1]
            
            def mostrar_mini_sidebar(code, label):
                if code and code not in ['N/A', 'nan', 'None', '']:
                    item_data = df[df['Code'] == code]
                    if not item_data.empty:
                        img_url = item_data.iloc[0]['ImageURL']
                        if img_url and len(str(img_url)) > 5:
                            st.sidebar.image(cargar_imagen_desde_url(img_url), use_container_width=True)
                        else:
                            st.sidebar.caption(f"{code}")
                    else:
                        st.sidebar.caption(f"{code}")
                else:
                    st.sidebar.caption("-")

            s1, s2, s3 = st.sidebar.columns(3)
            with s1: 
                st.caption("Top")
                mostrar_mini_sidebar(last_fit['Top'], "Top")
            with s2: 
                st.caption("Bot")
                mostrar_mini_sidebar(last_fit['Bottom'], "Bot")
            with s3: 
                st.caption("Out")
                mostrar_mini_sidebar(last_fit['Outer'], "Out")
        
    if not found_today:
        st.sidebar.info("🤷‍♂️ Nada registrado.")
        
except:
    st.sidebar.error("Error al cargar.")

if 'last_occ_viewed' not in st.session_state: st.session_state['last_occ_viewed'] = code_occ
if st.session_state['last_occ_viewed'] != code_occ:
    st.session_state['custom_overrides'] = {}; st.session_state['last_occ_viewed'] = code_occ
if 'seed' not in st.session_state: st.session_state['seed'] = random.randint(1, 1000) 
if 'custom_overrides' not in st.session_state: st.session_state['custom_overrides'] = {}
if 'change_mode' not in st.session_state: st.session_state['change_mode'] = False
if 'confirm_stage' not in st.session_state: st.session_state['confirm_stage'] = 0 
if 'alerts_buffer' not in st.session_state: st.session_state['alerts_buffer'] = []
if 'agenda_reserves' not in st.session_state: st.session_state['agenda_reserves'] = {}

weather = get_weather_open_meteo()

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["✨ Sugerencia", "🧺 Lavadero", "📦 Inventario", "➕ Nuevo Item", "📊 Estadísticas", "✈️ Viaje", "📅 Agenda"])

with tab1:
    today_str = get_mendoza_time().strftime("%Y-%m-%d")
    outfit_of_the_day = None
    
    reserved_today_codes = st.session_state['agenda_reserves'].get(get_mendoza_time().date(), [])
    
    if not st.session_state['change_mode']:
        try:
            fb = load_feedback_gsheet()
            if not fb.empty and 'Action' in fb.columns:
                accepted = fb[fb['Action'].astype(str).str.strip() == 'Accepted'].copy()
                accepted['DateObj'] = pd.to_datetime(accepted['Date'], errors='coerce').dt.date
                match = accepted[(accepted['DateObj'] == get_mendoza_time().date()) & (accepted['Occasion'] == code_occ)]
                if not match.empty: 
                    match = match.sort_values('Date')
                    outfit_of_the_day = match.iloc[-1]
        except: pass
    
    coat_advice = ""
    recs_df = pd.DataFrame()
    temp_calculada = weather['temp']

    if outfit_of_the_day is not None:
        st.success(f"✅ Esta es tu prenda de hoy para '{code_occ}'")
        st.info("Ya registraste este outfit. Para generar uno nuevo, toca 'Cambiar'.")
        codes_to_show = []
        if outfit_of_the_day['Top'] not in ['N/A', 'nan']: codes_to_show.append(outfit_of_the_day['Top'])
        if outfit_of_the_day['Bottom'] not in ['N/A', 'nan']: codes_to_show.append(outfit_of_the_day['Bottom'])
        if outfit_of_the_day['Outer'] not in ['N/A', 'nan']: codes_to_show.append(outfit_of_the_day['Outer'])
        recs_df = df[df['Code'].isin(codes_to_show)]
        temp_calculada = float(outfit_of_the_day['User_Adj_Temp'])
        _, _, coat_advice = recommend_outfit(df, weather, code_occ, 0)
    elif reserved_today_codes:
        st.info("📅 Tienes un outfit reservado para hoy en la Agenda.")
        recs_df = df[df['Code'].isin(reserved_today_codes)]
        _, temp_calculada, coat_advice = recommend_outfit(df, weather, code_occ, 0)
    else:
        recs_df, temp_calculada, coat_advice = recommend_outfit(df, weather, code_occ, st.session_state['seed'])

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
        col_w2.metric("Sensación", f"{weather['feels_like']}°C", f"Wind: {weather['wind']}km/h")
        offset_val = temp_calculada - weather.get('feels_like', weather['temp'])
        col_w3.metric("Perfil", f"{temp_calculada:.1f}°C", f"{offset_val:+.1f}°C (Smart)")
        if coat_advice: st.markdown(f"**{coat_advice}**")
        
        st.divider()
        if st.button("🚀 Enviar a Telegram", use_container_width=True):
            with st.spinner("Conectando con GDI HQ..."):
                ok, msg = enviar_briefing_diario()
                if ok:
                    st.success(f"Enviado: {msg}")
                else:
                    st.error(f"Error: {msg}")

    col_h1, col_h2 = st.columns([2, 2])
    with col_h1: st.subheader("Tu Outfit (AI)")
    with col_h2: 
        c_btn1, c_btn2 = st.columns(2)
        if c_btn1.button("🔄 Cambiar", use_container_width=True): 
            st.session_state['seed'] = random.randint(1, 1000)
            st.session_state['change_mode'] = True 
            st.session_state['custom_overrides'] = {}
            st.rerun()
        if c_btn2.button("🛠️ Manual", use_container_width=True): st.session_state['show_custom_ui'] = not st.session_state.get('show_custom_ui', False)

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
                    if new_top.strip(): overrides['top'] = new_top.strip()
                    if new_bot.strip(): overrides['bot'] = new_bot.strip()
                    if new_out.strip(): overrides['out'] = new_out.strip()
                    st.session_state['custom_overrides'] = overrides; st.session_state['show_custom_ui'] = False; st.rerun()

    rec_top, rec_bot, rec_out = None, None, None
    selected_items_codes = []

    if not recs_df.empty:
        t_row = recs_df[recs_df['Category'].isin(['Remera', 'Camisa'])]
        b_row = recs_df[recs_df['Category'] == 'Pantalón']
        o_row = recs_df[recs_df['Category'].isin(['Campera', 'Buzo'])]
        
        rec_top = t_row.iloc[0]['Code'] if not t_row.empty else "N/A"
        rec_bot = b_row.iloc[0]['Code'] if not b_row.empty else "N/A"
        rec_out = o_row.iloc[0]['Code'] if not o_row.empty else "N/A"

        if not t_row.empty: selected_items_codes.append(t_row.iloc[0])
        if not b_row.empty: selected_items_codes.append(b_row.iloc[0])
        if not o_row.empty: selected_items_codes.append(o_row.iloc[0])

        st.markdown("###### 📸 Visual Canvas")
        canvas = create_outfit_canvas(rec_top, rec_bot, rec_out, df)
        if canvas:
            st.image(canvas, use_column_width=True)
        else:
            st.info("No se pudo generar el canvas visual (faltan fotos).")
        st.divider()

        c1, c2, c3 = st.columns(3)
        with c1: 
            st.caption("Torso")
            st.write(f"**{rec_top}**")
        with c2: 
            st.caption("Piernas")
            st.write(f"**{rec_bot}**")
        with c3: 
            st.caption("Abrigo")
            st.write(f"**{rec_out}**")

        st.divider()

        if st.session_state['change_mode']:
            st.info("¿Qué falló en esta propuesta?")
            with st.container(border=True):
                reason = st.radio("Motivo del rechazo:", 
                        ["Hace frío/calor para esto", "No combinan los colores", "No tengo ganas de usar esa prenda", "Evento formal/informal"],
                        horizontal=True)

                if st.button("🎲 Dame otra opción"):
                    ra = 3
                    if reason == "Hace frío/calor para esto": ra = 1 
                    
                    entry = {
                        'Date': get_mendoza_time().strftime("%Y-%m-%d %H:%M"), 
                        'City': user_city, 
                        'Temp_Real': weather['temp'], 
                        'User_Adj_Temp': temp_calculada, 
                        'Occasion': code_occ, 
                        'Top': rec_top, 'Bottom': rec_bot, 'Outer': rec_out, 
                        'Rating_Abrigo': ra, 
                        'Rating_Comodidad': 1, 
                        'Rating_Seguridad': 1, 
                        'Action': 'Rejected',
                        'Reason': reason 
                    }
                    save_feedback_entry_gsheet(entry)

                    if 'temp_blacklist' not in st.session_state: st.session_state['temp_blacklist'] = []
                    if reason == "No tengo ganas de usar esa prenda":
                        st.session_state['temp_blacklist'].extend([rec_top, rec_bot, rec_out])

                    st.session_state['seed'] = random.randint(1, 1000)
                    st.session_state['change_mode'] = True
                    st.rerun()
        
        if outfit_of_the_day is None or st.session_state['change_mode']:
            if st.session_state['confirm_stage'] == 0:
                st.markdown("### ⭐ Confirmar y Calificar")
                def show_gradient_bar(): st.markdown('<div style="background: linear-gradient(90deg, #3b82f6 0%, #ffffff 50%, #ef4444 100%); height: 8px; border-radius: 4px; margin-bottom: 5px; opacity: 0.8;"></div>', unsafe_allow_html=True)
                st.caption("Outfit Completo")
                c_fb1, c_fb2, c_fb3 = st.columns(3)
                with c_fb1: st.markdown("**🌡️ Abrigo (1-7)**"); show_gradient_bar(); r_abrigo = st.select_slider("Global Abrigo", options=[1, 2, 3, 4, 5, 6, 7], value=4, label_visibility="collapsed", key="fb_abrigo")
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
                    # 1. PREPARAMOS EL ENTRY PRIMERO PARA QUE NO SE PIERDA
                    ra = r_abrigo; rc = r_comodidad + 1 if r_comodidad is not None else 3; rs = r_seguridad + 1 if r_seguridad is not None else 3
                    v_rt_a = rt_abr; v_rt_c = rt_com + 1 if rt_com is not None else 3; v_rt_f = rt_flow + 1 if rt_flow is not None else 3
                    v_rb_a = rb_abr; v_rb_c = rb_com + 1 if rb_com is not None else 3; v_rb_f = rb_flow + 1 if rb_flow is not None else 3
                    v_ro_a = ro_abr; v_ro_c = ro_com + 1 if ro_com is not None else 3; v_ro_f = ro_flow + 1 if ro_flow is not None else 3
                    
                    entry = {'Date': get_mendoza_time().strftime("%Y-%m-%d %H:%M"), 'City': user_city, 'Temp_Real': weather['temp'], 'User_Adj_Temp': temp_calculada, 'Occasion': code_occ, 'Top': rec_top, 'Bottom': rec_bot, 'Outer': rec_out, 'Rating_Abrigo': ra, 'Rating_Comodidad': rc, 'Rating_Seguridad': rs, 'Action': 'Accepted', 'Top_Abrigo': v_rt_a, 'Top_Comodidad': v_rt_c, 'Top_Flow': v_rt_f, 'Bot_Abrigo': v_rb_a, 'Bot_Comodidad': v_rb_c, 'Bot_Flow': v_rb_f, 'Out_Abrigo': v_ro_a, 'Out_Comodidad': v_ro_c, 'Out_Flow': v_ro_f}
                    
                    # Lo guardamos en memoria por si salta la alerta de límite
                    st.session_state['pending_feedback_entry'] = entry

                    if is_sweat:
                        for item in selected_items_codes:
                            idx = df[df['Code'] == item['Code']].index[0]
                            df.at[idx, 'Status'] = 'Sucio'; df.at[idx, 'LastWorn'] = datetime.now().strftime("%Y-%m-%d")
                        st.session_state['inventory'] = df; save_data_gsheet(df)
                        
                        save_feedback_entry_gsheet(entry) # <- Guardado Normal
                        
                        st.toast("💦 A lavar."); st.session_state['change_mode'] = False; st.rerun()
                    else:
                        alerts = []
                        for item in selected_items_codes:
                            idx = df[df['Code'] == item['Code']].index[0]
                            sna = decodificar_sna(item['Code'])
                            limit = get_limit_for_item(item['Category'], sna)
                            current_uses = int(float(df.at[idx, 'Uses'])) if df.at[idx, 'Uses'] not in ['', 'nan'] else 0
                            if (current_uses + 1) > limit: alerts.append({'code': item['Code'], 'cat': item['Category'], 'uses': current_uses, 'limit': limit})
                        
                        if alerts:
                            st.session_state['alerts_buffer'] = alerts; st.session_state['confirm_stage'] = 1; st.rerun()
                        else:
                            for item in selected_items_codes:
                                idx = df[df['Code'] == item['Code']].index[0]
                                curr = int(float(df.at[idx, 'Uses'])) if df.at[idx, 'Uses'] not in ['', 'nan'] else 0
                                df.at[idx, 'Uses'] = curr + 1; df.at[idx, 'LastWorn'] = datetime.now().strftime("%Y-%m-%d")
                            st.session_state['inventory'] = df; save_data_gsheet(df)
                            st.session_state['custom_overrides'] = {}; st.session_state['change_mode'] = False
                            
                            save_feedback_entry_gsheet(entry) # <- Guardado Normal
                            st.toast("¡Outfit registrado!"); st.rerun()

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
                        df.at[idx, 'Uses'] = curr + 1; df.at[idx, 'LastWorn'] = datetime.now().strftime("%Y-%m-%d")
                        save_data_gsheet(df)
                        
                        # Guardamos el feedback que había quedado en pausa
                        if 'pending_feedback_entry' in st.session_state:
                            save_feedback_entry_gsheet(st.session_state['pending_feedback_entry'])
                            
                        st.session_state['confirm_stage'] = 0; st.session_state['alerts_buffer'] = []; st.session_state['change_mode'] = False; st.rerun()
    else: st.error("No hay ropa limpia disponible (según filtros). ¡Lavá algo!")

with tab2: 
    st.header("Lavadero")
    
    with st.expander("🧠 Smart Laundry (Carga Inteligente)", expanded=True):
        if not df.empty:
            dirty_pool = df[df['Status'].isin(['Sucio', 'Lavando'])]
            clean_pool = df[df['Status'] == 'Limpio']
            
            # 1. APRENDIZAJE DE CAPACIDAD (Peso exacto basado en la nueva función)
            df['Weight_Est'] = df.apply(lambda r: calcular_peso_prenda(r['Category'], r['Code']), axis=1)
            
            learned_capacity = 4000 # Default 4kg
            try:
                history_wash = df[df['LaundryStart'] != '']
                if not history_wash.empty:
                    # IA APRENDE DEL LOTE: Agrupa por la marca de tiempo exacta
                    daily_loads = history_wash.groupby('LaundryStart')['Weight_Est'].sum()
                    real_loads = daily_loads[daily_loads > 1000]
                    if not real_loads.empty:
                        learned_capacity = int(real_loads.mean())
            except: pass
            
            st.caption(f"⚖️ Capacidad de lavado aprendida: **{learned_capacity/1000:.1f} kg**")

            if dirty_pool.empty:
                st.success("¡Nada que lavar!")
            else:
                total_counts = df['Category'].value_counts()
                clean_counts = clean_pool['Category'].value_counts()
                
                recs_smart = []
                for _, row in dirty_pool.iterrows():
                    c_cat = row['Category']
                    c_code = row['Code']
                    c_weight = calcular_peso_prenda(c_cat, c_code)
                    
                    scarcity = 1 - (clean_counts.get(c_cat, 0) / total_counts.get(c_cat, 1))
                    score_prio = (scarcity * 100) + random.uniform(0, 5)
                    
                    recs_smart.append({
                        'Code': c_code, 
                        'Category': c_cat, 
                        'Score': score_prio,
                        'Weight': c_weight
                    })
                
                recs_sorted = sorted(recs_smart, key=lambda x: x['Score'], reverse=True)
                
                curr_w = 0
                final_basket = []
                
                for item in recs_sorted:
                    if curr_w + item['Weight'] <= (learned_capacity * 1.1):
                        final_basket.append(item)
                        curr_w += item['Weight']
                
                st.info(f"💡 Sugerencia para optimizar carga ({curr_w/1000:.2f}kg / {learned_capacity/1000:.1f}kg):")
                
                cols = st.columns(3)
                for i, item in enumerate(final_basket):
                    cols[i%3].write(f"• `{item['Code']}`")
                
                if len(final_basket) < len(dirty_pool):
                    leftover = len(dirty_pool) - len(final_basket)
                    st.caption(f"Quedan {leftover} prendas sucias de menor prioridad para el próximo lavado.")
    
    st.divider()
    dirty_list = df[df.apply(is_needs_wash, axis=1)]
    st.subheader(f"🧺 Canasto de Ropa Sucia ({len(dirty_list)})")
    if not dirty_list.empty: st.dataframe(dirty_list[['Code', 'Category', 'Uses', 'Status']], use_container_width=True)
    else: st.info("Todo impecable ✨")
    
    # --- NUEVA LÓGICA DE LAVADO POR LOTE ---
    if 'wash_batch' not in st.session_state: st.session_state['wash_batch'] = []

    with st.container(border=True):
        st.subheader("🧺 Cargar Lavarropas (Lote)")
        col_m1, col_m2 = st.columns([3, 1])
        with col_m1: code_input = st.text_input("Ingresar Código de Prenda", key="batch_input")
        with col_m2:
            if st.button("➕ Agregar"):
                if code_input:
                    clean_code = code_input.strip()
                    if clean_code in df['Code'].values:
                        if clean_code not in st.session_state['wash_batch']:
                            st.session_state['wash_batch'].append(clean_code)
                            st.rerun()
                        else: st.warning("Ya está en el lavarropas.")
                    else: st.error("Código inválido.")

        if st.session_state['wash_batch']:
            st.write("**Prendas en el tambor ahora:**")
            st.info(", ".join(st.session_state['wash_batch']))
            
            col_b1, col_b2 = st.columns(2)
            if col_b1.button("🚀 Finalizar Lista (Lavar Lote)", type="primary", use_container_width=True):
                exact_timestamp = datetime.now().isoformat()
                for c_batch in st.session_state['wash_batch']:
                    idx = df[df['Code'] == c_batch].index[0]
                    df.at[idx, 'Status'] = 'Lavando'
                    df.at[idx, 'Uses'] = 0
                    df.at[idx, 'LaundryStart'] = exact_timestamp
                st.session_state['inventory'] = df
                save_data_gsheet(df)
                st.session_state['wash_batch'] = []
                st.success("✅ Lote enviado a lavar. La IA aprendió el peso de esta carga.")
                st.rerun()
                
            if col_b2.button("🗑️ Vaciar Lista", use_container_width=True):
                st.session_state['wash_batch'] = []
                st.rerun()
    
    edited_laundry = st.data_editor(df[['Code', 'Category', 'Status', 'Uses']], key="ed_lav", column_config={"Status": st.column_config.SelectboxColumn("Estado", options=["Limpio", "Sucio", "Lavando"], required=True)}, hide_index=True, disabled=["Code", "Category", "Uses"], use_container_width=True)
    if st.button("🔄 Actualizar Planilla"):
        df.update(edited_laundry)
        st.session_state['inventory'] = df; save_data_gsheet(df); st.success("Actualizado")

with tab3: 
    st.header("Inventario Total")
    
    with st.expander("🎲 Juego: ¿Si la perdiera?"):
        st.caption("La IA elige una prenda al azar. Si la perdieras hoy, ¿la comprarías de nuevo?")
        if st.button("🔄 Jugar"):
            clean_items = df[df['Status'] == 'Limpio']
            if not clean_items.empty:
                random_item = clean_items.sample(1).iloc[0]
                st.session_state['lost_game_item'] = random_item
            else: st.warning("No hay items limpios.")
            
        if 'lost_game_item' in st.session_state:
            item = st.session_state['lost_game_item']
            col_img, col_info = st.columns(2)
            with col_img:
                img = cargar_imagen_desde_url(item['ImageURL'])
                if img: st.image(img, width=150)
                else: st.write("📷 Sin foto")
            with col_info:
                st.markdown(f"**{item['Category']}**")
                st.caption(f"`{item['Code']}`")
                c_y, c_n = st.columns(2)
                if c_y.button("✅ Sí, la compro"):
                    st.toast("¡Bien! Es una prenda valiosa.")
                    del st.session_state['lost_game_item']; st.rerun()
                if c_n.button("❌ No, chau"):
                    st.error(f"Considera donar: {item['Code']}")
                    del st.session_state['lost_game_item']; st.rerun()
    
    active_inv = df[~df['Status'].str.contains('Archived', na=False)]
    edited_inv = st.data_editor(active_inv, num_rows="dynamic", use_container_width=True, column_config={"Uses": st.column_config.ProgressColumn("Desgaste", min_value=0, max_value=10, format="%d"), "ImageURL": st.column_config.LinkColumn("Foto")})
    if st.button("💾 Guardar Inventario Completo"): 
        df.update(edited_inv)
        st.session_state['inventory'] = df; save_data_gsheet(df); st.toast("Guardado")

    st.divider()
    st.subheader("🧹 Gestión Manual y Limpieza")
    st.info("Ingresa un código para modificar sus usos o Archivarlo (sacarlo de circulación).")
    
    with st.container(border=True):
        col_m_in, col_m_act = st.columns([1, 2])
        with col_m_in:
            manual_code = st.text_input("Código de Prenda")
        with col_m_act:
            c_act1, c_act2, c_act3 = st.columns(3)
            if manual_code:
                clean_m_code = manual_code.strip()
                if clean_m_code in df['Code'].values:
                    idx = df[df['Code'] == clean_m_code].index[0]
                    curr_uses = int(float(df.at[idx, 'Uses'])) if df.at[idx, 'Uses'] not in ['', 'nan'] else 0
                    
                    with c_act1:
                        if st.button("➕ Sumar Uso"):
                            df.at[idx, 'Uses'] = curr_uses + 1; df.at[idx, 'LastWorn'] = datetime.now().strftime("%Y-%m-%d")
                            st.session_state['inventory'] = df; save_data_gsheet(df); st.toast(f"📈 Usos: {curr_uses + 1}"); st.rerun()
                    with c_act2:
                        if st.button("➖ Restar Uso"):
                            df.at[idx, 'Uses'] = max(0, curr_uses - 1)
                            st.session_state['inventory'] = df; save_data_gsheet(df); st.toast(f"📉 Usos: {max(0, curr_uses - 1)}"); st.rerun()
                    
                    st.divider()
                    st.caption("Zona de Archivo (Desaparece del armario)")
                    reason = st.selectbox("Motivo de baja", ["Rota ✂️", "Vieja 👴", "No me gusta 👎", "Donada 🎁"])
                    if st.button(f"🗑️ Archivar como {reason}"):
                        tag = reason.split(" ")[0]
                        df.at[idx, 'Status'] = f"Archived_{tag}"
                        st.session_state['inventory'] = df; save_data_gsheet(df); st.success(f"Adiós {clean_m_code}! Archivada como {tag}"); st.rerun()
                else:
                    st.error("Código no encontrado.")

with tab4: 
    st.header("Alta de Prenda")
    with st.container(border=True):
        tipo_f = st.selectbox("Tipo", ["R - Remera", "CS - Camisa", "P - Pantalón", "C - Campera", "B - Buzo", "Ri - Ropa interior"])
        t_code = tipo_f.split(" - ")[0]

        if t_code == "Ri":
            ri_tipo = st.selectbox("Corte / Tipo", ["Boxer", "Brief", "Media"])
            prefix = {"Boxer": "RiB", "Brief": "RiBr", "Media": "RiM"}[ri_tipo]
            
            existing_codes = [c for c in df['Code'] if str(c).startswith(prefix)]
            code = f"{prefix}{len(existing_codes) + 1:02d}"
            
            st.info(f"Código Generado: `{code}` (Sin color, temporada, ni ocasión)")
            if st.button("Agregar a la Nube"):
                new = pd.DataFrame([{'Code': code, 'Category': 'Ropa interior', 'Season': 'T', 'Occasion': 'C', 'ImageURL': '', 'Status': 'Limpio', 'LastWorn': '', 'Uses': 0, 'LaundryStart': ''}])
                st.session_state['inventory'] = pd.concat([df, new], ignore_index=True)
                save_data_gsheet(st.session_state['inventory'])
                st.success(f"¡{code} subido a Google Sheets!")
                
        else:
            c1, c2 = st.columns(2)
            with c1:
                temp = st.selectbox("Temporada", ["V (Verano)", "W (Invierno)", "M (Media)", "T (Toda Estación)"]).split(" ")[0]
                if t_code == "P": attr = st.selectbox("Corte", ["Je (Jean)", "Sh (Short)", "DL (Deportivo)", "DC (Corto)", "Ve (Vestir)"]).split(" ")[0]
                elif t_code in ["C", "B"]: attr = f"0{st.selectbox('Abrigo', ['1 (Rompevientos)', '2 (Liviana)', '3 (Normal)', '4 (Gruesa)', '5 (Muy Gruesa)']).split(' ')[0]}"
                else: attr = st.selectbox("Manga", ["00 (Musculosa)", "01 (Corta)", "02 (Larga)"]).split(" ")[0]
            with c2:
                occ = st.selectbox("Ocasión", ["U", "D", "C", "F"])
                url = st.text_input("URL Foto")
                if url and st.button("👁️ Detectar Color"):
                    try:
                        img = cargar_imagen_desde_url(url)
                        if img:
                            hex_col = estimar_color_dominante(img)
                            st.caption(f"Detectado: {hex_col}")
                            st.color_picker("Tono Aproximado", hex_col, disabled=True)
                        else: st.error("Link inválido")
                    except: pass
                
                col = st.selectbox("Color", ["01-Blanco", "02-Negro", "03-Gris", "04-Azul", "05-Verde", "06-Rojo", "07-Amarillo", "08-Beige", "09-Marron", "10-Denim", "11-Naranja", "12-Violeta", "99-Estampado"])[:2]
        
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
    st.header("📊 Estadísticas Avanzadas")
    if not df.empty:
        df['Uses'] = pd.to_numeric(df['Uses'], errors='coerce').fillna(0)
        
        st.subheader("1. Principio de Pareto")
        total_uses = df['Uses'].sum()
        if total_uses > 0:
            sorted_items = df.sort_values('Uses', ascending=False)
            sorted_items['Cumulative_Uses'] = sorted_items['Uses'].cumsum()
            top_80 = sorted_items[sorted_items['Cumulative_Uses'] <= (0.8 * total_uses)]
            count_80 = len(top_80)
            total_items = len(df)
            perc_items = (count_80 / total_items) * 100
            st.write(f"Usas el **{perc_items:.1f}%** de tu armario para el **80%** de tus outfits.")
            st.progress(min(1.0, perc_items/100))
        else: st.info("Falta data de uso.")

        st.subheader("2. Rueda de Colores")
        df['Color_Code'] = df['Code'].apply(lambda x: decodificar_sna(x)['color'] if decodificar_sna(x) else 'Unknown')
        
        color_data = df['Color_Code'].value_counts().reset_index()
        color_data.columns = ['ColorCode', 'Count']
        
        color_data['ColorName'] = color_data['ColorCode'].map(COLOR_NAMES).fillna('Otro')
        color_data['Hex'] = color_data['ColorCode'].map(COLOR_MAP).fillna('#808080')
        
        chart = alt.Chart(color_data).mark_bar().encode(
            x=alt.X('ColorName', sort='-y'),
            y='Count',
            color=alt.Color('ColorName', scale=alt.Scale(domain=list(color_data['ColorName']), range=list(color_data['Hex'])), legend=None),
            tooltip=['ColorName', 'Count']
        ).interactive()
        
        st.altair_chart(chart, use_container_width=True)
        
        st.subheader("3. MVP vs Suplentes")
        c_mvp1, c_mvp2 = st.columns(2)
        with c_mvp1:
            st.markdown("**🏆 MVP (Más usados)**")
            st.dataframe(df.sort_values('Uses', ascending=False).head(3)[['Code', 'Category', 'Uses']], hide_index=True)
        with c_mvp2:
            st.markdown("**💤 Suplentes (0 uso y limpios)**")
            suplentes = df[(df['Uses'] == 0) & (df['Status'] == 'Limpio')]
            st.dataframe(suplentes.head(3)[['Code', 'Category']], hide_index=True)

        st.subheader("4. Tendencia Flow vs Comfort")
        try:
            fb_trend = load_feedback_gsheet()
            if not fb_trend.empty:
                fb_trend['Date'] = pd.to_datetime(fb_trend['Date'], errors='coerce')
                fb_trend = fb_trend.dropna(subset=['Date'])
                fb_trend['Week'] = fb_trend['Date'].dt.to_period('W').apply(lambda r: r.start_time)
                
                weekly = fb_trend.groupby('Week')[['Rating_Comodidad', 'Rating_Seguridad']].mean().reset_index()
                
                chart_data = weekly.melt('Week', var_name='Metric', value_name='Rating')
                c = alt.Chart(chart_data).mark_line(point=True).encode(
                    x='Week',
                    y=alt.Y('Rating', scale=alt.Scale(domain=[1, 5])),
                    color='Metric'
                ).interactive()
                st.altair_chart(c, use_container_width=True)
        except: st.warning("Falta data histórica.")

    st.divider()
    st.subheader("📅 Calendario de Outfits")
    try:
        fb_cal = load_feedback_gsheet()
        if not fb_cal.empty:
            fb_cal['DateObj'] = pd.to_datetime(fb_cal['Date'], errors='coerce').dt.date
            
            now = get_mendoza_time() 
            current_month = now.month
            current_year = now.year
            cal = calendar.monthcalendar(current_year, current_month)
            
            cols_cal = st.columns(7)
            days = ["L", "M", "Mi", "J", "V", "S", "D"]
            for i, d in enumerate(days): cols_cal[i].write(f"**{d}**")
            
            for week in cal:
                cols = st.columns(7)
                for i, day in enumerate(week):
                    if day == 0:
                        cols[i].write(" ")
                    else:
                        d_str = datetime(current_year, current_month, day).date()
                        has_outfit = not fb_cal[(fb_cal['DateObj'] == d_str) & (fb_cal['Action'].astype(str).str.strip() == 'Accepted')].empty
                        marker = "🟢" if has_outfit else "⚪"
                        if d_str == now.date(): marker = "📍"
                        cols[i].write(f"{day} {marker}")
    except: st.error("No se pudo cargar el calendario.")

with tab6:
    st.header("✈️ Modo Viaje v3.2 (Smart - Lite)") 
    
    if 'active_trip' not in st.session_state: st.session_state['active_trip'] = False
    if 'trip_current_day' not in st.session_state: st.session_state['trip_current_day'] = 0

    if not st.session_state['active_trip']:
        # --- MODO PLANIFICACIÓN ---
        col_dest, col_days = st.columns([2, 1])
        with col_dest: dest_city = st.text_input("📍 Destino", value="Buenos Aires")
        with col_days: num_days = st.number_input("📅 Días Totales", min_value=1, max_value=30, value=5)

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
            packable = df[df['Status'] == 'Limpio'].copy()
            forecast = st.session_state.get('travel_weather')
            
            if packable.empty: 
                st.error("¡No tenés ropa limpia para viajar!")
            else:
                if num_days > 6:
                    st.info("🧺 Viaje largo: Te armo valija para 6 días, te va a tocar lavar allá para no cargar de más.")
                    req_daily_tops = 6
                    req_daily_bots = 4
                else:
                    req_daily_tops = num_days 
                    req_daily_bots = (num_days // 2) + 1 
                
                # --- FILTRO CLIMÁTICO ESTRICTO ---
                avg_max = st.session_state.get('travel_avg_max', 20) 
                if forecast:
                    def weather_filter(row):
                        sna = decodificar_sna(row['Code'])
                        if not sna: return True
                        cat, attr = row['Category'], sna['attr']
                        
                        if avg_max > 25: # Hace calor
                            if cat == 'Pantalón' and attr in ['Je', 'DL']: return False # No Jeans, no largo deportivo
                            if cat in ['Remera', 'Camisa'] and attr == '02': return False # No manga larga
                            if cat in ['Campera', 'Buzo'] and attr in ['03', '04', '05']: return False # No camperas pesadas
                        elif avg_max < 15: # Hace frío
                            if cat == 'Pantalón' and attr in ['Sh', 'DC']: return False # No shorts
                            if cat in ['Remera', 'Camisa'] and attr == '00': return False # No musculosas
                        return True
                    
                    packable = packable[packable.apply(weather_filter, axis=1)]

                pool_home = packable[packable['Occasion'] == 'C']
                pool_street = packable[packable['Occasion'].isin(['U', 'F'])]
                pool_outs = packable[(packable['Category'].isin(['Campera', 'Buzo'])) & (packable['Occasion'].isin(['U', 'F', 'D']))]

                final_pack = []
                try: final_pack.append(pool_home[pool_home['Category'].isin(['Remera', 'Camisa'])].sample(1))
                except: st.warning("No hay remeras de 'Casa' limpias/aptas para dormir.")
                try: final_pack.append(pool_home[pool_home['Category'] == 'Pantalón'].sample(1))
                except: st.warning("No hay pantalones de 'Casa' limpios/aptos para dormir.")

                avail_u_tops = pool_street[pool_street['Category'].isin(['Remera', 'Camisa'])]
                avail_u_bots = pool_street[pool_street['Category'] == 'Pantalón']
                
                if not avail_u_tops.empty: final_pack.append(avail_u_tops.sample(min(len(avail_u_tops), req_daily_tops)))
                if not avail_u_bots.empty: final_pack.append(avail_u_bots.sample(min(len(avail_u_bots), req_daily_bots)))
                if not pool_outs.empty: final_pack.append(pool_outs.sample(min(len(pool_outs), 2)))

                if final_pack:
                    # Usamos drop_duplicates subset Code para evitar el DuplicateElementKey
                    st.session_state['travel_pack'] = pd.concat(final_pack).drop_duplicates(subset=['Code'])
                    st.session_state['travel_selections'] = {} 
                    st.rerun()
                else: 
                    st.error("No se pudo generar la valija (falta stock limpio que cumpla con el clima).")

        if st.session_state.get('travel_pack') is not None:
            pack = st.session_state['travel_pack'].copy()
            packable = df[df['Status'] == 'Limpio']
            
            st.divider()
            st.subheader(f"🧳 Tu Valija ({len(pack)} prendas)")
            
            # --- NUEVO BOTÓN DE GUARDADO ---
            if st.button("💾 Guardar Valija en la Nube", type="primary", use_container_width=True):
                st.session_state['travel_pack'] = pack 
                save_travel_gsheet(st.session_state)
                st.success("¡Valija sincronizada! Ya podés abrir la app en el celular.")
            
            # --- SUGERENCIA ROPA INTERIOR ---
            st.info(f"🩲 **Sugerencia de Ropa Interior:** Llevá **{num_days + 2} pares** (1 para cada día + 2 de repuesto).")
            
            c_stats1, c_stats2 = st.columns(2)
            c_stats1.success(f"🏠 Casa (Dormir): {len(pack[pack['Occasion'] == 'C'])}")
            c_stats2.success(f"🎓 Universidad/Formal: {len(pack[pack['Occasion'].isin(['U', 'F'])] )}")
            
            cols = st.columns(3)
            for i, (index, row) in enumerate(pack.iterrows()):
                with cols[i % 3]:
                    with st.container(border=True):
                        emoji_occ = "🏠" if row['Occasion'] == 'C' else "🎓" if row['Occasion'] == 'U' else "👔"
                        img = cargar_imagen_desde_url(row['ImageURL'])
                        if img: st.image(img, use_container_width=True)
                        else: st.write("📷 Sin foto")
                        st.markdown(f"**{emoji_occ} {row['Category']}**")
                        st.caption(f"Code: `{row['Code']}`")
                        
                        # Checkboxes con Key segura sumándole "i"
                        c_ida, c_vuelta = st.columns(2)
                        is_ida = c_ida.checkbox("Ida", key=f"ida_{row['Code']}_{i}")
                        is_vuelta = c_vuelta.checkbox("Vuel", key=f"vuelta_{row['Code']}_{i}")
                        if 'travel_selections' not in st.session_state: st.session_state['travel_selections'] = {}
                        st.session_state['travel_selections'][row['Code']] = {'ida': is_ida, 'vuelta': is_vuelta}
                        
                        # --- DROPDOWN MEJORADO: Excluye repetidos, unifica Camisas/Remeras ---
                        if row['Category'] in ['Remera', 'Camisa']:
                            alt_pool = packable[packable['Category'].isin(['Remera', 'Camisa'])]
                        else:
                            alt_pool = packable[packable['Category'] == row['Category']]
                            
                        # Filtro por ocasión y exclusión de los que ya están en la valija
                        alt_pool = alt_pool[alt_pool['Occasion'] == row['Occasion']]
                        alt_pool = alt_pool[~alt_pool['Code'].isin(pack['Code'])]
                            
                        alts = alt_pool['Code'].head(7).tolist()
                        options = [f"Mantener {row['Code']}"] + alts + ["✍️ Ingresar código manual..."]
                        choice = st.selectbox("Cambiar por:", options, key=f"swap_{row['Code']}_{i}", label_visibility="collapsed")
                        
                        if choice == "✍️ Ingresar código manual...":
                            custom_code = st.text_input("Código a agregar:", key=f"custom_{row['Code']}_{i}")
                            if custom_code:
                                if custom_code in df['Code'].values:
                                    if st.button("Confirmar cambio", key=f"conf_{row['Code']}_{i}"):
                                        pack.loc[pack['Code'] == row['Code']] = df[df['Code'] == custom_code].iloc[0].values
                                        st.session_state['travel_pack'] = pack
                                        st.rerun()
                                else:
                                    st.error("El código no existe en BD.")
                        elif choice != f"Mantener {row['Code']}":
                            pack.loc[pack['Code'] == row['Code']] = df[df['Code'] == choice].iloc[0].values
                            st.session_state['travel_pack'] = pack
                            st.rerun()

            st.divider()
            sel = st.session_state.get('travel_selections', {})
            ida_items = [code for code, vals in sel.items() if vals.get('ida')]
            vuelta_items = [code for code, vals in sel.items() if vals.get('vuelta')]
            c1, c2 = st.columns(2)
            c1.info(f"🛫 **Ida:** {', '.join(ida_items) if ida_items else '---'}")
            c2.success(f"🛬 **Vuelta:** {', '.join(vuelta_items) if vuelta_items else '---'}")
            
            st.divider()
            col_start, col_del = st.columns(2)
            if col_start.button("✈️ INICIAR VIAJE", type="primary", use_container_width=True):
                if st.session_state.get('travel_weather'):
                    st.session_state['active_trip'] = True
                    st.session_state['trip_current_day'] = 0
                    save_travel_gsheet(st.session_state) # --- NUEVO GUARDADO ---
                    st.rerun()
                else:
                    st.error("Analizá el clima del destino primero.")
                    
            if col_del.button("🗑️ Borrar Valija", type="secondary", use_container_width=True):
                st.session_state['travel_pack'] = None; st.session_state['travel_selections'] = {}; st.session_state['active_trip'] = False
                save_travel_gsheet(st.session_state) # --- NUEVO GUARDADO ---
                st.rerun()

        st.divider()
        with st.expander("📋 Checklist de Supervivencia", expanded=False):
            essentials = ["DNI / Pasaporte", "Cargador", "Cepillo Dientes", "Desodorante", "Auriculares", "Medicamentos", "Lentes", "Billetera"]
            cols_ch = st.columns(2)
            for i, item in enumerate(essentials): cols_ch[i % 2].checkbox(item, key=f"check_{i}")

    else:
        # --- MODO VIAJE ACTIVO ---
        st.success("✈️ ¡Modo viaje activado! Te sugiero prendas solo con tu valija.")
        
        forecast = st.session_state.get('travel_weather')
        pack_df = st.session_state.get('travel_pack')
        curr_day = st.session_state['trip_current_day']
        
        if forecast and pack_df is not None:
            total_days = len(forecast['time'])
            
            if curr_day < total_days:
                date_str = forecast['time'][curr_day]
                max_t = forecast['temperature_2m_max'][curr_day]
                min_t = forecast['temperature_2m_min'][curr_day]
                w_code = forecast['weather_code'][curr_day]
                avg_t = (max_t + min_t) / 2
                
                st.markdown(f"### 📅 Día {curr_day + 1} de {total_days}")
                st.caption(f"{datetime.strptime(date_str, '%Y-%m-%d').strftime('%d/%m/%Y')}")
                st.metric("Clima en Destino", f"{int(avg_t)}°C Prom.", f"{int(min_t)}° min / {int(max_t)}° max {get_weather_emoji(w_code)}")
                
                trip_weather = {"temp": avg_t, "feels_like": avg_t, "min": min_t, "max": max_t, "desc": "Destino", "hourly_temp": [], "hourly_time": [], "wind": 10, "humidity": 50, "weather_code": w_code}
                
                occ_trip = st.selectbox("🎯 ¿Ocasión para hoy?", ["U (Turismo/Casual)", "F (Cena/Formal)", "C (Hotel/Dormir)"])
                code_occ_trip = occ_trip[0]
                
                recs_trip, t_feel_trip, advice_trip = recommend_outfit(pack_df, trip_weather, code_occ_trip, random.randint(1, 1000))
                
                if not recs_trip.empty:
                    t_row = recs_trip[recs_trip['Category'].isin(['Remera', 'Camisa'])]
                    b_row = recs_trip[recs_trip['Category'] == 'Pantalón']
                    o_row = recs_trip[recs_trip['Category'].isin(['Campera', 'Buzo'])]
                    
                    rec_top = t_row.iloc[0]['Code'] if not t_row.empty else "N/A"
                    rec_bot = b_row.iloc[0]['Code'] if not b_row.empty else "N/A"
                    rec_out = o_row.iloc[0]['Code'] if not o_row.empty else "N/A"
                    
                    canvas = create_outfit_canvas(rec_top, rec_bot, rec_out, pack_df)
                    if canvas:
                        st.image(canvas, use_column_width=True)
                        
                    c1, c2, c3 = st.columns(3)
                    c1.write(f"**Top:** {rec_top}")
                    c2.write(f"**Bot:** {rec_bot}")
                    c3.write(f"**Out:** {rec_out}")
                else:
                    st.warning("No pude combinar nada con lo que tenés en la valija para esta ocasión.")
                
                st.divider()
                col_next, col_end = st.columns(2)
                if col_next.button("⏭️ Avanzar de Día", use_container_width=True):
                    st.session_state['trip_current_day'] += 1
                    save_travel_gsheet(st.session_state) # --- NUEVO GUARDADO ---
                    st.rerun()
                if col_end.button("🛑 Terminar Viaje", type="secondary", use_container_width=True):
                    st.session_state['active_trip'] = False
                    st.session_state['travel_pack'] = None
                    save_travel_gsheet(st.session_state) # --- NUEVO GUARDADO ---
                    st.rerun()
            else:
                st.info("¡Fin del viaje programado!")
                if st.button("🏠 Desarmar Valija y volver", type="primary", use_container_width=True):
                    st.session_state['active_trip'] = False
                    st.session_state['travel_pack'] = None
                    save_travel_gsheet(st.session_state) # --- NUEVO GUARDADO ---
                    st.rerun()
        else:
            st.error("Datos del viaje perdidos. Volviendo a Planificación...")
            st.session_state['active_trip'] = False
            st.rerun()
with tab7:
    st.header("📅 Agenda de Eventos")
    st.info("Reserva outfits para fechas futuras. La IA no te sugerirá estas prendas antes de la fecha para que no se ensucien.")
    
    with st.form("agenda_form"):
        date_res = st.date_input("Fecha del Evento", min_value=datetime.now())
        codes_clean = df[df['Status'] == 'Limpio']['Code'].tolist()
        codes_res = st.multiselect("Prendas a reservar", codes_clean)
        if st.form_submit_button("Reservar Outfit"):
            if date_res and codes_res:
                st.session_state['agenda_reserves'][date_res] = codes_res
                st.success(f"Outfit reservado para el {date_res}")
                st.rerun()
            else: st.error("Selecciona fecha y prendas.")
            
    st.subheader("Tus Reservas Activas")
    if st.session_state['agenda_reserves']:
        for d, codes in st.session_state['agenda_reserves'].items():
            with st.container(border=True):
                st.markdown(f"**{d}**")
                st.write(f"Prendas: {', '.join(codes)}")
                if st.button("Cancelar", key=f"del_{d}"):
                    del st.session_state['agenda_reserves'][d]
                    st.rerun()
    else:
        st.caption("No hay eventos futuros.")

# ==========================================
# --- MÓDULO TELEGRAM INTERACTIVO (Telebot) ---
# ==========================================
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton

def generar_teclado_ocasiones():
    markup = InlineKeyboardMarkup()
    markup.row_width = 2
    markup.add(
        InlineKeyboardButton("🏠 Casa", callback_data="occ_C"),
        InlineKeyboardButton("🎓 Universidad", callback_data="occ_U"),
        InlineKeyboardButton("👔 Formal", callback_data="occ_F"),
        InlineKeyboardButton("⚽ Deporte", callback_data="occ_D")
    )
    return markup

def generar_teclado_acciones(occ_code):
    markup = InlineKeyboardMarkup()
    markup.add(
        InlineKeyboardButton("🔄 Cambiar", callback_data=f"reroll_{occ_code}"),
        InlineKeyboardButton("🛠️ Manual", callback_data="manual_mode")
    )
    return markup

if bot:
    @bot.callback_query_handler(func=lambda call: True)
    def callback_query(call):
        if call.data.startswith("occ_"):
            occ_code = call.data.split("_")[1]
            bot.answer_callback_query(call.id, "Generando outfit...")
            enviar_sugerencia_interactiva(occ_code, call.message.chat.id)

        elif call.data.startswith("reroll_"):
            occ_code = call.data.split("_")[1]
            bot.answer_callback_query(call.id, "Buscando otra opción...")
            try: bot.delete_message(call.message.chat.id, call.message.message_id)
            except: pass
            enviar_sugerencia_interactiva(occ_code, call.message.chat.id, force_new_seed=True)

        elif call.data == "manual_mode":
            bot.answer_callback_query(call.id)
            bot.send_message(call.message.chat.id, "🛠️ Entendido. Abre la app para editar manualmente: https://gdi-mendoza-ops-v21.streamlit.app")

def enviar_sugerencia_interactiva(occ_code, chat_id, force_new_seed=False):
    if not bot: return
    try:
        df_inv = load_data_gsheet()
        if df_inv.empty:
            bot.send_message(chat_id, "⚠️ Error leyendo base de datos.")
            return

        w_data = get_weather_open_meteo()
        seed = random.randint(1, 10000) if force_new_seed else int(datetime.now().timestamp())
        recs, temp_calc, advice = recommend_outfit(df_inv, w_data, occ_code, seed)
        
        if recs.empty:
            bot.send_message(chat_id, "⚠️ No hay ropa limpia disponible para esta ocasión.")
            return

        r_top = recs[recs['Category'].isin(['Remera', 'Camisa'])].iloc[0]['Code'] if not recs[recs['Category'].isin(['Remera', 'Camisa'])].empty else None
        r_bot = recs[recs['Category'] == 'Pantalón'].iloc[0]['Code'] if not recs[recs['Category'] == 'Pantalón'].empty else None
        r_out = recs[recs['Category'].isin(['Campera', 'Buzo'])].iloc[0]['Code'] if not recs[recs['Category'].isin(['Campera', 'Buzo'])].empty else None

        canvas = create_outfit_canvas(r_top, r_bot, r_out, df_inv)
        
        caption = (
            f"🧥 *Propuesta para {occ_code}*\n"
            f"🌡️ Clima: {w_data['temp']}°C (ST {w_data['feels_like']}°C)\n"
            f"💡 {advice}\n\n"
            f"• Top: {r_top}\n• Bot: {r_bot}\n• Out: {r_out}"
        )

        bio = BytesIO()
        canvas.save(bio, format='PNG')
        bio.seek(0)
        
        bot.send_photo(
            chat_id, 
            photo=bio, 
            caption=caption, 
            parse_mode='Markdown',
            reply_markup=generar_teclado_acciones(occ_code)
        )

    except Exception as e:
        bot.send_message(chat_id, f"Error generando: {str(e)}")

def tarea_manana():
    if not bot: return
    try:
        bot.send_message(
            TELEGRAM_CHAT_ID, 
            "🌅 *Buenos días.*\n¿Para qué ocasión te vistes hoy?", 
            parse_mode="Markdown",
            reply_markup=generar_teclado_ocasiones()
        )
    except Exception as e:
        print(f"Error tarea mañana: {e}")

def tarea_noche():
    if not bot: return
    try:
        msg = (
            "🌙 *Check de fin de día*\n"
            "No olvides calificar el outfit sugerido hoy para mejorar la IA.\n\n"
            "👉 [Abrir GDI: Mendoza Ops](https://gdi-mendoza-ops-v21.streamlit.app)"
        )
        bot.send_message(TELEGRAM_CHAT_ID, msg, parse_mode="Markdown")
    except Exception as e:
        print(f"Error tarea noche: {e}")

def crear_collage_lavanderia(codes, df_inv):
    try:
        imgs = []
        for c in codes:
            row = df_inv[df_inv['Code'] == c]
            if not row.empty:
                url = row.iloc[0]['ImageURL']
                pil_img = cargar_imagen_desde_url(url)
                if pil_img: 
                    base_width = 300
                    w_percent = (base_width / float(pil_img.size[0]))
                    h_size = int((float(pil_img.size[1]) * float(w_percent)))
                    img_resized = pil_img.resize((base_width, h_size), Image.Resampling.LANCZOS)
                    imgs.append(img_resized)
        
        if not imgs: return None

        total_height = sum([i.size[1] for i in imgs])
        canvas = Image.new('RGB', (300, total_height), (255, 255, 255))
        
        y_offset = 0
        for i in imgs:
            canvas.paste(i, (0, y_offset))
            y_offset += i.size[1]
            
        return canvas
    except: return None

def tarea_lavanderia():
    if not bot: return
    try:
        df = load_data_gsheet()
        if df.empty: return

        dirty_pool = df[df['Status'].isin(['Sucio', 'Lavando'])]
        if len(dirty_pool) < 4: 
            return 

        clean_pool = df[df['Status'] == 'Limpio']
        total_counts = df['Category'].value_counts()
        clean_counts = clean_pool['Category'].value_counts()
        
        recs = []
        for _, row in dirty_pool.iterrows():
            cat = row['Category']
            tot = total_counts.get(cat, 1)
            cln = clean_counts.get(cat, 0)
            scarcity = 1 - (cln / tot)
            recs.append({'Code': row['Code'], 'Cat': cat, 'Score': scarcity})
        
        top_urgente = sorted(recs, key=lambda x: x['Score'], reverse=True)[:5]
        codes_to_wash = [x['Code'] for x in top_urgente]
        
        msg = (
            "🧺 *¡Hora de Lavar!*\n"
            f"Tienes {len(dirty_pool)} prendas sucias acumuladas.\n\n"
            "🧠 *Sugerencia Smart (Prioridad):*\n"
        )
        for item in top_urgente:
            msg += f"• `{item['Code']}` ({item['Cat']})\n"
        
        msg += "\n_Entra a la app para marcar como 'Limpio'._"

        canvas = crear_collage_lavanderia(codes_to_wash, df)
        
        if canvas:
            bio = BytesIO()
            canvas.save(bio, format='PNG')
            bio.seek(0)
            bot.send_photo(TELEGRAM_CHAT_ID, photo=bio, caption=msg, parse_mode='Markdown')
        else:
            bot.send_message(TELEGRAM_CHAT_ID, msg, parse_mode='Markdown')

    except Exception as e:
        print(f"Error lavanderia: {e}")

# ==========================================
# --- THREAD DE CONTROL BLINDADO (SINGLETON) ---
# ==========================================
def run_scheduler():
    schedule.every().day.at("10:00").do(tarea_manana) # 07:00 AM AR
    schedule.every().day.at("01:00").do(tarea_noche)  # 22:00 PM AR
    schedule.every().friday.at("21:00").do(tarea_lavanderia) # 18:00 AR
    schedule.every().sunday.at("14:00").do(tarea_lavanderia) # 11:00 AR
    
    while True:
        try:
            schedule.run_pending()
            time.sleep(30)
        except Exception as e:
            print(f"⚠️ Error en scheduler: {e}")
            time.sleep(30)

def iniciar_bot_singleton():
    if bot:
        hilos_activos = [t.name for t in threading.enumerate()]
        
        if "Scheduler_GDI" not in hilos_activos:
            t_sched = threading.Thread(target=run_scheduler, name="Scheduler_GDI", daemon=True)
            t_sched.start()
            
        if "Bot_Polling_GDI" not in hilos_activos:
            def run_polling():
                while True:
                    try:
                        bot.polling(none_stop=True, interval=0, timeout=20)
                    except Exception as e:
                        print(f"Error polling: {e}")
                        time.sleep(15)
                        
            t_poll = threading.Thread(target=run_polling, name="Bot_Polling_GDI", daemon=True)
            t_poll.start()
            print("✅ Bot y Scheduler iniciados correctamente (Instancias Únicas).")

if __name__ == "__main__":
    iniciar_bot_singleton()
