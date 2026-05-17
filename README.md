# SIA-Maritima-Tarragona
Proyecto predicción oleaje
# 🌊 Predictor de Oleaje · Boya de Tarragona

**Seminario de Inteligencia Artificial en Ingeniería Marítima I**  
ETSI Caminos, Canales y Puertos · Universidad Politécnica de Madrid · 2026

---

## Descripción

Desarrollo de una **Red Neuronal Artificial (ANN)** de tipo Feedforward para la 
predicción de la altura significante de ola (Hm0) en t+1h a partir de datos 
oceanográficos de la Boya de Tarragona (Puertos del Estado / Copernicus Marine Service).

## Datos

- **Fuente:** Copernicus Marine Service (CMEMS) · Modelo mediterráneo de oleaje
- **Boya:** Tarragona · Mediterráneo occidental
- **Periodo:** Mayo 2022 – Mayo 2026
- **Resolución:** Horaria · 35.065 registros

## Variables del modelo

| Variable | Descripción | Unidad |
|---|---|---|
| VHM0 | Altura significante de ola (target) | m |
| VTPK | Período pico | s |
| VMDR | Dirección media del oleaje | grados |
| VHM0_SW1 | Altura de swell | m |
| VHM0_WW | Altura de oleaje de viento | m |

## Pipeline

1. Carga y exploración de datos
2. Preprocesamiento (NaN, duplicados, normalización Standard)
3. División Hold-out 70-15-15%
4. Arquitectura ANN: 3 capas ocultas · ReLU · Dropout 20% · Adam
5. Entrenamiento con EarlyStopping
6. Evaluación: MSE, RMSE, MAE, R², Bias
7. Validación cruzada K-Fold K=10
8. Aplicación web interactiva (Streamlit)

## Tecnologías

Python · TensorFlow/Keras · scikit-learn · pandas · xarray · Streamlit

## Autor

Rodrigo Gil Llorente · ETSI Caminos, Canales y Puertos · UPM
