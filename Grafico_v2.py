import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import cv2 as cv
from tkinter import filedialog
from pathlib import Path

# ==========================================================================
# 0. CONFIGURACIÓN — AJUSTAR ESTOS NOMBRES A LOS ARCHIVOS REALES DE ESTE BARRIDO
# ==========================================================================

# Csv con los dutycycles
ruta_archivo = Path(filedialog.askopenfilename(
    title="Abrir archivo con los dutycycles usados",
    defaultextension=".csv",
    initialdir=Path.home(),        
    filetypes=[("CSV","*.csv*"),("All files", "*.*")]
))

if not ruta_archivo:
    print("No se seleccionó ningún archivo. Saliendo.")
    exit()

dutys_csv = ruta_archivo.name 
nombre_dutys_csv = ruta_archivo.stem   
ruta_dutys_csv = ruta_archivo.parent        

dutys_path = rf'{ruta_dutys_csv}\{dutys_csv}'

# CSV de la adquisición (Tiempo, Monitoreo_X_CH1, Scattering_CH2)
ruta_archivo = Path(filedialog.askopenfilename(
    title="Abrir archivo con los datos adquiridos",
    defaultextension=".csv",
    initialdir=Path.home(),        
    filetypes=[("CSV","*.csv*"),("All files", "*.*")]
))

if not ruta_archivo:
    print("No se seleccionó ningún archivo. Saliendo.")
    exit()

adquisicion_csv = ruta_archivo.name 
nombre_adquisicion_csv = ruta_archivo.stem   
ruta_adquisicion_csv = ruta_archivo.parent     

adquisicion_path = rf'{ruta_adquisicion_csv}\{adquisicion_csv}'

# Umbral para descartar puntos saturados/espurios de Scattering_CH2 (V)
UMBRAL_SATURACION = 1.0

# Resolución de la grilla de salida (píxeles de la imagen reconstruida)
NUM_PIXELS_X = 500
NUM_PIXELS_Y = 500


# ==========================================================================
# 1. RECONSTRUIR EL MODELO CINEMÁTICO (idéntico a calculo_dcs.py)
# ==========================================================================

promedio_cross_x = pd.read_csv(fr'C:\Users\Usuario\Desktop\Facu\Labo 6\Laboratorio6\Calibración\Datos_ajuste\ajuste_lin_x_calv1_0107.csv')
promedio_cross_y = pd.read_csv(fr'C:\Users\Usuario\Desktop\Facu\Labo 6\Laboratorio6\Calibración\Datos_ajuste\ajuste_lin_y_calv1_0107.csv')
                                    
promedio_cross_x = [promedio_cross_x['m'].mean(), 0]
promedio_cross_y = [promedio_cross_y['m'].mean(), 0]

datos_ajuste_x = pd.read_csv(fr'C:\Users\Usuario\Desktop\Facu\Labo 6\Laboratorio6\Calibración\Datos_ajuste\ajuste_cubico_x_calv1_0107.csv')
datos_ajuste_y = pd.read_csv(fr'C:\Users\Usuario\Desktop\Facu\Labo 6\Laboratorio6\Calibración\Datos_ajuste\ajuste_cubico_y_calv1_0107.csv')

promedio_x = datos_ajuste_x.mean()[1::]
coefs_x = [coef for coef in promedio_x]
coefs_x[-1] = 0

promedio_y = datos_ajuste_y.mean()[1::]
coefs_y = [coef for coef in promedio_y]
coefs_y[-1] = 0

cross_x = np.poly1d(promedio_cross_x)
cross_y = np.poly1d(promedio_cross_y)

polinomio_x = np.poly1d(coefs_x)
polinomio_y = np.poly1d(coefs_y)


def cinematica_directa(dcx, dcy):
    """
    Convierte duty cycles (dcx, dcy) a posición física real (X, Y) en micrones,
    usando el mismo modelo (no linealidad + cross-talk) que calculo_dcs.py.
    """
    X_fisico = polinomio_x(dcx) + cross_y(polinomio_y(dcy))
    Y_fisico = polinomio_y(dcy) + cross_x(polinomio_x(dcx))
    return X_fisico, Y_fisico


# ==========================================================================
# 2. CARGAR LA SECUENCIA DE DUTY CYCLES PROGRAMADA (posición real objetivo)
# ==========================================================================

df_dutys = pd.read_csv(dutys_path)
# El CSV se guardó con to_csv() sin index=False -> puede traer una columna "Unnamed: 0"
df_dutys = df_dutys.loc[:, ~df_dutys.columns.str.contains('^Unnamed')]

dcx_seq = df_dutys['Dcx'].values
dcy_seq = df_dutys['Dcy'].values

x_real, y_real = cinematica_directa(dcx_seq, dcy_seq)


# ==========================================================================
# 3. CARGAR LOS DATOS ADQUIRIDOS (Scattering)
# ==========================================================================

df_adq = pd.read_csv(adquisicion_path)
scattering = df_adq['Scattering_CH2 (V)'].values
v_x_medido = df_adq['Monitoreo_X_CH1 (V)'].values  # solo para chequeo de consistencia

# ---- Chequeo de sincronización 1 a 1 ----
n_dutys = len(dcx_seq)
n_adq = len(scattering)

if n_dutys != n_adq:
    print(f"[!] ADVERTENCIA: la secuencia de duty cycles tiene {n_dutys} puntos "
          f"pero la adquisición tiene {n_adq} puntos.")
    print("Se recorta al mínimo común para no desalinear el resto del barrido.")
    n_min = min(n_dutys, n_adq)
    dcx_seq, dcy_seq = dcx_seq[:n_min], dcy_seq[:n_min]
    x_real, y_real = x_real[:n_min], y_real[:n_min]
    scattering = scattering[:n_min]
    v_x_medido = v_x_medido[:n_min]
else:
    print(f"Sincronización OK: {n_dutys} puntos en ambos archivos.")

# ---- (Opcional) chequeo de consistencia CH1 vs duty cycle programado ----
# Si el monitoreo de X es proporcional al duty cycle, esta correlación debería
# ser fuerte. Sirve como diagnóstico, no se usa para la reconstrucción.
correlacion = np.corrcoef(v_x_medido, dcx_seq)[0, 1]
print(f"Correlación entre CH1 medido y Dcx programado: {correlacion:.4f} "
      f"(valores bajos pueden indicar pérdida de sincronismo)")

# ---- Descartar puntos saturados, manteniendo la correspondencia con X,Y ----
mascara_valida = scattering <= UMBRAL_SATURACION
x_real = x_real[mascara_valida]
y_real = y_real[mascara_valida]
scattering = scattering[mascara_valida]


# ==========================================================================
# 4. INTERPOLAR A UNA GRILLA REGULAR Y GRAFICAR
# ==========================================================================

grid_x, grid_y = np.mgrid[
    x_real.min():x_real.max():complex(NUM_PIXELS_X),
    y_real.min():y_real.max():complex(NUM_PIXELS_Y)
]

grid_luz = griddata(
    points=(x_real, y_real),
    values=scattering,
    xi=(grid_x, grid_y),
    method='linear'
)

plt.figure(figsize=(10, 6))
plt.imshow(
    grid_luz.T,
    extent=[x_real.min(), x_real.max(), y_real.min(), y_real.max()],
    origin='lower',
    cmap='viridis',
    aspect='equal'
)

plt.colorbar(label='Intensidad de Scattering (V) - CH2')
plt.title('Reconstrucción 2D corregida (modelo cinemático directo)')
plt.xlabel('Posición X ($\\mu$m)')
plt.ylabel('Posición Y ($\\mu$m)')
plt.grid(False)
plt.show()