import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import os
os.chdir(r'c:\Users\LEC\Desktop\Laboratorio6')
# ---------------------------------------------------------
# 1. CARGA Y DEFINICIÓN DEL MODELO DIRECTO
# ---------------------------------------------------------

promedio_cross_x = [-0.03461811951503092, 0]
promedio_cross_y = [-0.06852970480292754, 0]

# Intenta leer tus datos de ajuste. 
# Al usar poly1d, si las columnas de promedio_x tienen 4 elementos, arma grado 3 solo.
datos_ajuste_x = pd.read_csv(r'Calibración\Aproach_NR\ajuste_cubico_x_calv1_1906.csv') 
datos_ajuste_y = pd.read_csv(r'Calibración\Aproach_NR\ajuste_cubico_y_calv1_1906.csv')

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

d_polinomio_x = np.polyder(polinomio_x)
d_polinomio_y = np.polyder(polinomio_y)

os.chdir(r'C:\Users\LEC\Desktop\AdquisicionRigol')
# =========================================================
# 1. CARGA DE DATOS (Asegurate de poner los nombres correctos)
# =========================================================
# El archivo que guardó el RIGOL con los promedios
df_medicion = pd.read_csv('mapeo_scattering_punto_a_punto_v3.csv')

# El archivo que generó tu script NR_testeo_v2.py con los Duty Cycles
# (Buscá el nombre exacto con la hora que te generó, ej: 'dutys_barrido_discreto_x_2906_1811.csv')
df_dutys = pd.read_csv('dutys_barrido_discreto_x_2906_1714.csv') 

# =========================================================
# 2. ALINEACIÓN Y CONVERSIÓN A DUTY CYCLES
# =========================================================
# Por si el osciloscopio capturó algún punto de menos o de más, igualamos longitudes
min_len = min(len(df_medicion), len(df_dutys))
print(f'Puntos recorridos: {len(df_dutys)}')
print(f'Datos tomados: {len(df_medicion)}')


# Voltaje máximo real de tu señal PWM (típicamente 3.3V para un ESP32)
# Ajustá este valor si tu cuadrada tiene un pico distinto en el osciloscopio
V_MAX_PWM = 0.32458 

# Calculamos el Duty Cycle REAL en X midiendo el Vavg del Canal 1
dcx_real_medido = (df_medicion['Monitoreo_X_CH1 (V)'].values[:min_len]) / V_MAX_PWM

# Usamos el Duty Cycle TEÓRICO en Y (asumimos que en Y el ESP no pierde pasos)
dcy_teorico = df_dutys['Dcy'].values[:min_len]

# Extraemos la intensidad de scattering
intensidad_luz = df_medicion['Scattering_CH2 (V)'].values[:min_len]

# =========================================================
# 3. CINEMÁTICA DIRECTA (TU MODELO FÍSICO)
# =========================================================
# Usamos tus mismos polinomios para pasar de DC a Micrones reales compensando Cross-Talk
X_fisico = polinomio_x(dcx_real_medido) + cross_x(polinomio_y(dcy_teorico))
Y_fisico = polinomio_y(dcy_teorico) + cross_y(polinomio_x(dcx_real_medido))

# =========================================================
# 4. INTERPOLACIÓN EN GRILLA RECTANGULAR
# =========================================================
# Definimos los límites físicos del mapa en micrones
x_min, x_max = X_fisico.min(), X_fisico.max()
y_min, y_max = Y_fisico.min(), Y_fisico.max()

# Creamos una grilla perfecta de "píxeles" (ej: 300x300 de resolución)
grid_x, grid_y = np.mgrid[x_min:x_max:300j, y_min:y_max:300j]

# Interpolamos la nube de puntos distorsionada hacia la grilla perfecta
grid_luz = griddata(
    points=(X_fisico, Y_fisico), 
    values=intensidad_luz, 
    xi=(grid_x, grid_y), 
    method='cubic' # 'cubic' suaviza bien las formas circulares
)

# =========================================================
# 5. GRAFICACIÓN CON RELACIÓN DE ASPECTO CORREGIDA
# =========================================================
plt.figure(figsize=(8, 6))

# Ploteamos la imagen transpuesta (T) porque np.mgrid orienta diferente los ejes
plt.imshow(
    grid_luz.T, 
    extent=[x_min, x_max, y_min, y_max], 
    origin='lower', 
    cmap='viridis',
    aspect='equal' # ¡MAGIA ACÁ! Esto fuerza a que 1um en X mida igual que 1um en Y
)

plt.colorbar(label='Intensidad de Scattering (V)')
plt.title('Mapa de Discos de Silicio (1 $\mu$m) - Corregido por Cross-Talk')
plt.xlabel('Posición Real X ($\mu$m)')
plt.ylabel('Posición Real Y ($\mu$m)')
plt.grid(False)

plt.tight_layout()
plt.savefig('mapa_scattering_corregido.png', dpi=300)
plt.show()