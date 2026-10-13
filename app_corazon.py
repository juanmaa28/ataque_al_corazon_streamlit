# -*- coding: utf-8 -*-
"""Despliegue del modelo de predicción de ataque al corazón (Streamlit)."""

import pickle

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Pulso · Riesgo de ataque al corazón", page_icon="🫀", layout="centered")

# ---------------------------------------------------------------- Modelo
@st.cache_resource
def cargar_modelo():
    # Se guardó como [modelo_final, labelencoder, variables, min_max_scaler]
    with open("modelo-class-ata.pkl", "rb") as f:
        return pickle.load(f)


modelo, labelencoder, variables, min_max_scaler = cargar_modelo()

# ---------------------------------------------------------------- Estilos
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,800&family=Figtree:wght@400;500;600&display=swap');

:root{
  --paper:#FBF3F3; --grid:#F0D9DB; --ink:#17203A; --muted:#5E6679;
  --signal:#D92B3F; --ok:#127A6F; --card:#FFFFFF; --line:#E6D0D3;
}
html, body, [class*="css"], .stApp { font-family:'Figtree',sans-serif; color:var(--ink); }
.stApp{
  background-color:var(--paper);
  background-image:
    linear-gradient(var(--grid) 1px, transparent 1px),
    linear-gradient(90deg, var(--grid) 1px, transparent 1px);
  background-size:24px 24px;
}
#MainMenu, footer, header[data-testid="stHeader"]{visibility:hidden; height:0;}
.block-container{max-width:760px; padding-top:2.2rem; padding-bottom:3rem;}

.hero h1{
  font-family:'Bricolage Grotesque',sans-serif; font-weight:800;
  font-size:clamp(2.2rem,6vw,3.4rem); line-height:1.02; letter-spacing:-.02em;
  margin:0 0 .6rem; color:var(--ink);
}
.hero p{color:var(--muted); font-size:1.05rem; max-width:52ch; margin:0 0 .8rem;}
.ecg{width:100%; height:70px; display:block; margin:.2rem 0 1rem;}
.ecg path{fill:none; stroke:var(--signal); stroke-width:2.5; stroke-linecap:round; stroke-linejoin:round;
  stroke-dasharray:900; stroke-dashoffset:900; animation:trace 2.4s ease-out forwards;}
@keyframes trace{to{stroke-dashoffset:0}}

[data-testid="stForm"]{
  background:var(--card); border:1.5px solid var(--ink); border-radius:14px;
  padding:1.4rem 1.4rem 1rem; box-shadow:6px 6px 0 var(--ink);
}
label, [data-testid="stWidgetLabel"] p{font-weight:600 !important; color:var(--ink) !important;}
div[data-baseweb="select"] > div{border-radius:8px; border-color:var(--line); background:#FFFCFC;}
[data-testid="stSlider"] [role="slider"]{background:var(--signal) !important; box-shadow:none !important;}
[data-testid="stFormSubmitButton"] button{
  width:100%; background:var(--ink); color:#fff; border:none; border-radius:10px;
  padding:.7rem 1rem; font-weight:600; font-size:1.05rem; transition:background .15s;
}
[data-testid="stFormSubmitButton"] button:hover{background:var(--signal); color:#fff;}
[data-testid="stFormSubmitButton"] button:focus-visible, div[data-baseweb="select"]:focus-within > div{
  outline:3px solid var(--signal); outline-offset:2px;
}

.result{
  margin-top:1.6rem; background:var(--card); border:1.5px solid var(--ink); border-radius:14px;
  padding:1.3rem 1.4rem 1.2rem; box-shadow:6px 6px 0 var(--ink);
}
.result h2{font-family:'Bricolage Grotesque',sans-serif; font-weight:800; font-size:1.7rem; margin:.1rem 0 .3rem; letter-spacing:-.01em;}
.result.alto h2{color:var(--signal);} .result.bajo h2{color:var(--ok);}
.result p{color:var(--muted); margin:.2rem 0 .8rem;}
.result svg{width:100%; height:64px; display:block;}
.result.alto svg path{stroke:var(--signal);} .result.bajo svg path{stroke:var(--ok);}
.result svg path{fill:none; stroke-width:2.5; stroke-linecap:round; stroke-linejoin:round;
  stroke-dasharray:900; stroke-dashoffset:900; animation:trace 1.6s ease-out forwards;}
.chips{display:flex; flex-wrap:wrap; gap:.4rem; margin-top:.9rem;}
.chip{font-size:.85rem; border:1px solid var(--line); background:var(--paper); border-radius:999px; padding:.2rem .65rem; color:var(--ink);}
.aviso{font-size:.82rem; color:var(--muted); margin-top:2rem;}

@media (prefers-reduced-motion: reduce){
  .ecg path, .result svg path{animation:none; stroke-dashoffset:0;}
}
</style>
""",
    unsafe_allow_html=True,
)

# Trazo de electrocardiograma: ritmo regular y ritmo irregular
ECG_NORMAL = "M0 40 H60 l8 -6 l8 6 H110 l6 6 l8 -38 l8 54 l6 -22 H190 l10 -10 l10 10 H300 l6 6 l8 -38 l8 54 l6 -22 H380 l10 -10 l10 10 H490 l6 6 l8 -38 l8 54 l6 -22 H570 l10 -10 l10 10 H700"
ECG_ALTO = "M0 40 H40 l6 -4 l6 4 H80 l5 8 l6 -34 l6 58 l5 -32 H130 l8 -8 l8 8 H170 l5 8 l5 -20 l5 40 l5 -28 H215 l5 -6 l5 6 H250 l6 8 l9 -38 l9 56 l6 -26 H330 l8 -8 l8 8 H380 l5 8 l5 -18 l5 34 l5 -24 H430 l12 -10 l12 10 H700"

st.markdown(
    f"""
<div class="hero">
  <h1>Pulso</h1>
  <p>Ingresa los datos de la persona y el modelo estimará si presenta riesgo de sufrir un ataque al corazón.</p>
  <svg class="ecg" viewBox="0 0 700 80" preserveAspectRatio="none" aria-hidden="true"><path d="{ECG_NORMAL}"/></svg>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- Formulario
SI_NO = {"No": "No", "Sí": "Yes"}
FUMADOR = {
    "Nunca ha fumado": "'never smoked'",
    "Fumó antes": "'formerly smoked'",
    "Fuma actualmente": "smokes",
    "Sin información": "Unknown",
}

with st.form("formulario"):
    c1, c2 = st.columns(2)
    with c1:
        edad = st.slider("Edad (años)", min_value=1, max_value=82, value=45)
        hipertension = st.selectbox("Hipertensión", list(SI_NO))
        casado = st.selectbox("¿Ha estado casado/a?", list(SI_NO))
    with c2:
        glucosa = st.slider("Glucosa promedio (mg/dL)", min_value=55.0, max_value=272.0, value=100.0, step=0.5)
        enfermedad = st.selectbox("Enfermedad del corazón", list(SI_NO))
        fumador = st.selectbox("Hábito de fumar", list(FUMADOR))
    enviar = st.form_submit_button("Evaluar riesgo")

# ---------------------------------------------------------------- Predicción
if enviar:
    data = pd.DataFrame(
        [[edad, SI_NO[hipertension], SI_NO[enfermedad], SI_NO[casado], glucosa, FUMADOR[fumador]]],
        columns=["age", "hypertension", "heart_disease", "ever_married", "avg_glucose_level", "smoking_status"],
    )

    # Misma preparación del entrenamiento: dummies (drop_first=True), mismas columnas y escalado
    prep = pd.get_dummies(
        data, columns=["smoking_status", "hypertension", "heart_disease", "ever_married"],
        drop_first=False, dtype=int,
    )
    prep = prep.reindex(columns=variables, fill_value=0)
    prep[["age", "avg_glucose_level"]] = min_max_scaler.transform(prep[["age", "avg_glucose_level"]])

    pred = labelencoder.inverse_transform(modelo.predict(prep))[0]
    riesgo = pred == "Yes"

    chips = "".join(
        f'<span class="chip">{t}</span>'
        for t in [
            f"{edad} años",
            f"Glucosa {glucosa:g}",
            f"Hipertensión: {hipertension}",
            f"Corazón: {enfermedad}",
            fumador,
        ]
    )
    st.markdown(
        f"""
<div class="result {'alto' if riesgo else 'bajo'}" role="status">
  <svg viewBox="0 0 700 80" preserveAspectRatio="none" aria-hidden="true"><path d="{ECG_ALTO if riesgo else ECG_NORMAL}"/></svg>
  <h2>{'Riesgo alto de ataque al corazón' if riesgo else 'Riesgo bajo de ataque al corazón'}</h2>
  <p>{'El perfil ingresado se parece al de personas que sufrieron un ataque. Conviene consultar a un profesional de la salud.'
      if riesgo else 'El perfil ingresado se parece al de personas que no sufrieron un ataque.'}</p>
  <div class="chips">{chips}</div>
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown(
    '<p class="aviso">Modelo educativo (KNN, k=1) entrenado con 5.110 registros. No reemplaza un diagnóstico médico.</p>',
    unsafe_allow_html=True,
)
