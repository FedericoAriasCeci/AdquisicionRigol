import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# 1. Cargar los datos del barrido recién adquirido
version = 'v5_0107'

archivo_csv = f'mapeo_scattering_punto_a_punto_{version}.csv'
df = pd.read_csv(archivo_csv)

# Extraemos las columnas de interés
voltaje_x = df['Monitoreo_X_CH1 (V)'][df['Scattering_CH2 (V)'] <= 1].values
intensidad_luz = df['Scattering_CH2 (V)'][df['Scattering_CH2 (V)'] <= 1].values
tiempo = df['Tiempo (s)'][df['Scattering_CH2 (V)'] <= 1].values


# 2. Reconstrucción del eje Y (Discreto)
# Como el osciloscopio solo midió X (CH1) y Luz (CH2), necesitamos reconstruir 
# el eje Y en base al tiempo. Sabemos que el barrido es una serie de filas en X.
# Vamos a estimar cuántos puntos entran por cada fila de X.
# Con un rango de 8 um a pasos de 0.05 um, cada fila tiene ~160 puntos. 
# Sumando tiempos de retorno, estimemos unas 25 filas para cubrir los 2.5 um (saltos de 0.1 um).
total_puntos = len(voltaje_x)
pasos_x_por_fila = 233  # Ajustar este número si conocen el tamaño exacto de su array de ida en X
cant_filas_y = int(np.ceil(total_puntos / pasos_x_por_fila))

# Creamos un eje Y teórico que asocie cada punto a su fila correspondiente
y_discreto = np.zeros(total_puntos)
for i in range(total_puntos):
    nro_fila = i // pasos_x_por_fila
    y_discreto[i] = nro_fila * 0.05 # Saltos de 0.1 micrones

# 3. Escalar el Voltaje de X a Micrones reales (0 a 8 um)
v_min, v_max = voltaje_x.min(), voltaje_x.max()
x_micrones = 8.0 * (voltaje_x - v_min) / (v_max - v_min)
y_micrones = y_discreto

# 4. Crear una grilla regular (perfecta) para la imagen 2D
# Definimos la resolución de los "píxeles" de nuestra imagen final
num_pixels_x = pasos_x_por_fila 
num_pixels_y = cant_filas_y

grid_x, grid_y = np.mgrid[0:8:complex(num_pixels_x), 0:y_micrones.max():complex(num_pixels_y)]

# 5. Interpolación bidimensional (ataca no-linealidades y desfases de tiempo)
grid_luz = griddata(
    points=(x_micrones, y_micrones), 
    values=intensidad_luz, 
    xi=(grid_x, grid_y), 
    method='linear' # Puede ser 'linear', 'nearest' o 'cubic'
)

# 6. Graficar el mapa 2D de Scattering
plt.figure(figsize=(10, 6))

# Graficamos usando imshow (transponemos la grilla para orientar bien los ejes)
plt.imshow(
    grid_luz.T, 
    extent=[0, 8, 0, y_micrones.max()],
    origin='lower', 
    cmap='viridis'
)

plt.colorbar(label='Intensidad de Scattering (V) - CH2')
plt.title('Reconstrucción 2D: Array de Discos de Silicio (1 $\mu$m)')
plt.xlabel('Posición X ($\mu$m)')
plt.ylabel('Posición Y ($\mu$m)')
plt.grid(False)

# Guardar la imagen del mapa
plt.savefig(f'reconstruccion_muestra_2D_{version}.png', dpi=300)
plt.show()