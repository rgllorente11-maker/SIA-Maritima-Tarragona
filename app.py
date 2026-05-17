import streamlit as st
import numpy as np
import joblib
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# ── Configuración de página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Predictor de Oleaje · Tarragona",
    page_icon="🌊",
    layout="wide"
)

# ── CSS personalizado ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0a1628; color: white; }
    section[data-testid="stSidebar"] { background-color: #0d1f3c; border-right: 1px solid #1e4976; }
    section[data-testid="stSidebar"] * { color: white !important; }
    .titulo { font-size: 2.4em; font-weight: 900; color: white; text-align: center; padding: 10px 0; }
    .subtitulo { font-size: 1em; color: #7eb8d4; text-align: center; margin-bottom: 20px; }
    .card { background: linear-gradient(135deg, #112240, #1a3a5c); border-radius: 15px; padding: 20px; margin: 8px 0; border: 1px solid #1e4976; }
    .metrica-label { font-size: 0.8em; color: #7eb8d4; text-transform: uppercase; letter-spacing: 1px; }
    .metrica-valor { font-size: 2.6em; font-weight: 900; color: white; line-height: 1.1; }
    .seccion { font-size: 1.2em; font-weight: 700; color: white; margin: 18px 0 8px 0; border-left: 4px solid #4a9eff; padding-left: 10px; }
    .footer { text-align: center; color: #3a6080; font-size: 0.8em; margin-top: 30px; padding-top: 15px; border-top: 1px solid #1e4976; }
    div[data-testid="stMetricValue"] { color: white !important; font-size: 2em !important; }
    div[data-testid="stMetricLabel"] { color: #7eb8d4 !important; }
    h1, h2, h3, p, label { color: white !important; }
    .stButton button { background-color: #1e4976; color: white; border-radius: 8px; border: 1px solid #4a9eff; width: 100%; }
    .stButton button:hover { background-color: #4a9eff; }
</style>
""", unsafe_allow_html=True)

# ── Cargar modelo ─────────────────────────────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    model    = keras.models.load_model('modelo_tarragona.keras')
    scaler_X = joblib.load('scaler_X.pkl')
    scaler_y = joblib.load('scaler_y.pkl')
    return model, scaler_X, scaler_y

model, scaler_X, scaler_y = cargar_modelo()

# ── Historial en session state ────────────────────────────────────────────────
if 'historial' not in st.session_state:
    st.session_state.historial = []

# ── CABECERA ──────────────────────────────────────────────────────────────────
st.markdown('<div class="titulo">🌊 Predictor de Oleaje · Boya de Tarragona</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitulo">SIA Marítima I · ETSI Caminos, Canales y Puertos · Universidad Politécnica de Madrid</div>', unsafe_allow_html=True)
st.markdown("---")

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Condiciones actuales")
    st.markdown("---")
    altura  = st.slider("🌊 Altura de ola (m)",       0.1,  2.5,   0.9,  0.1)
    periodo = st.slider("⏱️ Período de ola (s)",      4.0,  8.0,   6.0,  0.1)
    viento  = st.slider("💨 Viento (m/s)",             0.0, 12.0,   5.0,  0.5)
    temp    = st.slider("🌡️ Temperatura agua (°C)",  14.0, 21.0,  17.0,  0.5)
    presion = st.slider("🔵 Presión atm. (hPa)",    1004.0,1020.0,1013.0, 0.5)
    st.markdown("---")
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        guardar = st.button("💾 Guardar")
    with col_btn2:
        limpiar = st.button("🗑️ Limpiar")
    st.markdown("---")
    st.markdown("**ℹ️ Modelo**")
    st.markdown("Red Neuronal FFNN · 3 capas ocultas · Adam optimizer · Datos sintéticos Boya Tarragona")

# ── PREDICCIÓN ────────────────────────────────────────────────────────────────
datos      = np.array([[altura, periodo, viento, temp, presion]])
datos_norm = scaler_X.transform(datos)
pred_norm  = model.predict(datos_norm, verbose=0)
pred       = float(scaler_y.inverse_transform(pred_norm)[0][0])
pred       = max(0.0, pred)

# Estado del mar
if pred < 0.5:
    estado = "Mar en calma"
    color  = "#27ae60"
    emoji  = "🟢"
    nivel  = 1
elif pred < 1.0:
    estado = "Mar poco agitado"
    color  = "#2ecc71"
    emoji  = "🟡"
    nivel  = 2
elif pred < 1.5:
    estado = "Mar agitado"
    color  = "#e67e22"
    emoji  = "🟠"
    nivel  = 3
elif pred < 2.0:
    estado = "Mar muy agitado"
    color  = "#e74c3c"
    emoji  = "🔴"
    nivel  = 4
else:
    estado = "⚠️ TEMPORAL"
    color  = "#8e44ad"
    emoji  = "🚨"
    nivel  = 5

# Guardar en historial
if guardar:
    st.session_state.historial.append({
        'hora'    : datetime.now().strftime('%H:%M:%S'),
        'ola_m'   : altura,
        'viento'  : viento,
        'pred_m'  : round(pred, 2),
        'estado'  : estado
    })

if limpiar:
    st.session_state.historial = []

# ══════════════════════════════════════════════════════════════════════════════
# FILA 1: Métricas principales
# ══════════════════════════════════════════════════════════════════════════════
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="metrica-label">Hm0 predicha en t+1h</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metrica-valor">{pred:.2f} <span style="font-size:0.4em;color:#7eb8d4">m</span></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="metrica-label">Estado del mar</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:1.6em;font-weight:800;color:{color};margin-top:5px">{emoji} {estado}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="metrica-label">Ola actual introducida</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metrica-valor">{altura:.1f} <span style="font-size:0.4em;color:#7eb8d4">m</span></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="metrica-label">Variación predicha</div>', unsafe_allow_html=True)
    diff = pred - altura
    signo = "+" if diff >= 0 else ""
    color_diff = "#e74c3c" if diff > 0.1 else "#27ae60" if diff < -0.1 else "#f1c40f"
    st.markdown(f'<div style="font-size:2.2em;font-weight:900;color:{color_diff}">{signo}{diff:.2f} <span style="font-size:0.4em;color:#7eb8d4">m</span></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# FILA 2: Alerta + Nivel de peligro
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="seccion">🚨 Panel de alertas</div>', unsafe_allow_html=True)

col_alerta, col_nivel = st.columns([2, 1])

with col_alerta:
    st.markdown(f"""
    <div style="background:{color};border-radius:15px;padding:25px;text-align:center">
        <div style="font-size:3em">{emoji}</div>
        <div style="font-size:1.8em;font-weight:900;color:white;margin:5px 0">{estado}</div>
        <div style="font-size:1em;color:rgba(255,255,255,0.8)">Altura predicha: {pred:.2f} m en la próxima hora</div>
    </div>
    """, unsafe_allow_html=True)

with col_nivel:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="metrica-label">Escala de peligrosidad</div>', unsafe_allow_html=True)
    niveles = ["🟢 Calma", "🟡 Poco agitado", "🟠 Agitado", "🔴 Muy agitado", "🚨 Temporal"]
    for i, niv in enumerate(niveles):
        bg = color if i == nivel - 1 else "rgba(255,255,255,0.05)"
        borde = f"2px solid {color}" if i == nivel - 1 else "1px solid rgba(255,255,255,0.1)"
        st.markdown(f'<div style="background:{bg};border:{borde};border-radius:8px;padding:6px 10px;margin:4px 0;font-size:0.9em">{niv}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# FILA 3: Gráfico de ola + Condiciones
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="seccion">📈 Simulación visual de la ola</div>', unsafe_allow_html=True)

col_ola, col_cond = st.columns([3, 1])

with col_ola:
    x = np.linspace(0, 4 * np.pi, 500)
    y_ola = pred * np.sin(x)
    y_act = altura * np.sin(x + 0.3)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    fig.patch.set_facecolor('#112240')
    ax.set_facecolor('#112240')

    ax.plot(x, y_act, color='#4a9eff', linewidth=1.5, linestyle='--', alpha=0.6, label=f'Ola actual ({altura:.1f} m)')
    ax.fill_between(x, y_ola, alpha=0.25, color=color)
    ax.plot(x, y_ola, color=color, linewidth=2.5, label=f'Ola predicha ({pred:.2f} m)')
    ax.axhline(0, color='rgba(255,255,255,0.2)', linewidth=0.8)
    ax.set_ylim(-3, 3)
    ax.set_ylabel("Altura (m)", color='#7eb8d4')
    ax.set_xlabel("Longitud de onda", color='#7eb8d4')
    ax.tick_params(colors='#7eb8d4')
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e4976')
    ax.legend(facecolor='#112240', edgecolor='#1e4976', labelcolor='white')
    ax.grid(alpha=0.15, color='white')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col_cond:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="metrica-label">Condiciones introducidas</div>', unsafe_allow_html=True)
    condiciones = {
        "🌊 Ola actual" : f"{altura} m",
        "⏱️ Período"    : f"{periodo} s",
        "💨 Viento"     : f"{viento} m/s",
        "🌡️ Temperatura": f"{temp} °C",
        "🔵 Presión"    : f"{presion} hPa"
    }
    for k, v in condiciones.items():
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #1e4976">
            <span style="color:#7eb8d4;font-size:0.9em">{k}</span>
            <span style="color:white;font-weight:600">{v}</span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# FILA 4: Comparativa de escenarios
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="seccion">🔄 Comparativa de escenarios</div>', unsafe_allow_html=True)

escenarios = {
    "Mar en calma":     [0.3, 5.5, 2.0, 17.0, 1015.0],
    "Mar moderado":     [0.9, 6.0, 6.0, 17.0, 1012.0],
    "Mar agitado":      [1.4, 6.5, 9.0, 16.0, 1008.0],
    "Temporal":         [2.2, 7.5,12.0, 15.0, 1005.0],
    "Condición actual": [altura, periodo, viento, temp, presion]
}

preds_escenarios = {}
for nombre, vals in escenarios.items():
    d = np.array([vals])
    dn = scaler_X.transform(d)
    pn = model.predict(dn, verbose=0)
    preds_escenarios[nombre] = float(scaler_y.inverse_transform(pn)[0][0])

fig2, ax2 = plt.subplots(figsize=(10, 3.5))
fig2.patch.set_facecolor('#112240')
ax2.set_facecolor('#112240')

nombres = list(preds_escenarios.keys())
valores = [max(0, v) for v in preds_escenarios.values()]
colores_barra = ['#27ae60','#2ecc71','#e67e22','#8e44ad','#4a9eff']

bars = ax2.bar(nombres, valores, color=colores_barra, edgecolor='#0a1628', linewidth=1.5, width=0.6)
ax2.axhline(pred, color='#4a9eff', linewidth=1.5, linestyle='--', alpha=0.7, label=f'Predicción actual ({pred:.2f} m)')

for bar, val in zip(bars, valores):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
             f'{val:.2f}m', ha='center', va='bottom', color='white', fontsize=10, fontweight='bold')

ax2.set_ylabel("Hm0 predicha (m)", color='#7eb8d4')
ax2.set_ylim(0, max(valores) * 1.25 + 0.2)
ax2.tick_params(colors='#7eb8d4')
for spine in ax2.spines.values():
    spine.set_edgecolor('#1e4976')
ax2.legend(facecolor='#112240', edgecolor='#1e4976', labelcolor='white')
ax2.grid(axis='y', alpha=0.15, color='white')
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
st.pyplot(fig2)
plt.close()

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# FILA 5: Historial de predicciones
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="seccion">📋 Historial de predicciones</div>', unsafe_allow_html=True)

if len(st.session_state.historial) == 0:
    st.markdown('<div style="color:#3a6080;text-align:center;padding:20px">Usa el botón 💾 Guardar del panel lateral para registrar predicciones aquí.</div>', unsafe_allow_html=True)
else:
    df_hist = pd.DataFrame(st.session_state.historial)
    df_hist.columns = ['Hora', 'Ola actual (m)', 'Viento (m/s)', 'Predicción (m)', 'Estado']

    # Gráfico del historial
    fig3, ax3 = plt.subplots(figsize=(10, 2.5))
    fig3.patch.set_facecolor('#112240')
    ax3.set_facecolor('#112240')
    ax3.plot(range(len(df_hist)), df_hist['Predicción (m)'], color='#4a9eff', linewidth=2, marker='o', markersize=6)
    ax3.fill_between(range(len(df_hist)), df_hist['Predicción (m)'], alpha=0.2, color='#4a9eff')
    ax3.set_ylabel("Hm0 (m)", color='#7eb8d4')
    ax3.set_xticks(range(len(df_hist)))
    ax3.set_xticklabels(df_hist['Hora'], rotation=30, ha='right', fontsize=8)
    ax3.tick_params(colors='#7eb8d4')
    for spine in ax3.spines.values():
        spine.set_edgecolor('#1e4976')
    ax3.grid(alpha=0.15, color='white')
    ax3.set_title("Evolución de predicciones guardadas", color='white', fontsize=10)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

    st.dataframe(
        df_hist,
        use_container_width=True,
        hide_index=True
    )

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="footer">
    🌊 Predictor de Oleaje · Boya de Tarragona · Puertos del Estado<br>
    SIA Marítima I · ETSI Caminos, Canales y Puertos · UPM · {datetime.now().year}<br>
    Modelo: ANN FFNN · TensorFlow/Keras · Datos: sintéticos (pendiente datos reales Puertos del Estado)
</div>
""", unsafe_allow_html=True)
