import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# ==========================================================================
# 0. CONFIGURACIÓN — AJUSTAR ESTOS NOMBRES A LOS ARCHIVOS REALES DE ESTE BARRIDO
# ==========================================================================

# Fecha de la calibración usada para ESTE barrido (misma que en calculo_dcs.py)
fecha_calibracion = '0107'

# CSV con la secuencia de duty cycles programada en el ESP32 para este barrido
# (el que generó calculo_dcs.py con dutys_csv.to_csv(...))
archivo_dutys = 'dutys_scanning_xy_XXXX_XXXX.csv'   # <-- completar con el nombre real

# CSV de la adquisición (Tiempo, Monitoreo_X_CH1, Scattering_CH2)
version = 'v5_0107'
archivo_csv_adquisicion = f'mapeo_scattering_punto_a_punto_{version}.csv'

# Umbral para descartar puntos saturados/espurios de Scattering_CH2 (V)
UMBRAL_SATURACION = 1.0

# Resolución de la grilla de salida (píxeles de la imagen reconstruida)
NUM_PIXELS_X = 300
NUM_PIXELS_Y = 300


# ==========================================================================
# 1. RECONSTRUIR EL MODELO CINEMÁTICO (idéntico a calculo_dcs.py)
# ==========================================================================

promedio_cross_x = pd.read_csv(fr'Calibración\Datos_ajuste\ajuste_lin_x_calv1_{fecha_calibracion}.csv')
promedio_cross_y = pd.read_csv(fr'Calibración\Datos_ajuste\ajuste_lin_y_calv1_{fecha_calibracion}.csv')

promedio_cross_x = [promedio_cross_x['m'].mean(), 0]
promedio_cross_y = [promedio_cross_y['m'].mean(), 0]

datos_ajuste_x = pd.read_csv(fr'Calibración\Datos_ajuste\ajuste_cubico_x_calv1_{fecha_calibracion}.csv')
datos_ajuste_y = pd.read_csv(fr'Calibración\Datos_ajuste\ajuste_cubico_y_calv1_{fecha_calibracion}.csv')

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

df_dutys = pd.read_csv(archivo_dutys)
# El CSV se guardó con to_csv() sin index=False -> puede traer una columna "Unnamed: 0"
df_dutys = df_dutys.loc[:, ~df_dutys.columns.str.contains('^Unnamed')]

dcx_seq = df_dutys['Dcx'].values
dcy_seq = df_dutys['Dcy'].values

x_real, y_real = cinematica_directa(dcx_seq, dcy_seq)


# ==========================================================================
# 3. CARGAR LOS DATOS ADQUIRIDOS (Scattering)
# ==========================================================================

df_adq = pd.read_csv(archivo_csv_adquisicion)
scattering_raw = df_adq['Scattering_CH2 (V)'].values
v_x_medido_raw = df_adq['Monitoreo_X_CH1 (V)'].values

n_dutys = len(dcx_seq)
n_adq = len(scattering_raw)
print(f"Puntos en secuencia de duty cycles: {n_dutys} | Puntos adquiridos: {n_adq}")


# ==========================================================================
# 3.5 SINCRONIZACIÓN POR CORRELACIÓN CRUZADA (CH1 medido vs Dcx programado)
# ==========================================================================
# CH1 mide el duty cycle eléctrico (post filtro RC), que es proporcional al
# Dcx programado -- NO a la posición física no lineal. Por eso es la señal
# correcta para encontrar el desfasaje de muestras entre lo adquirido y lo
# programado (p.ej. muestras "muertas" tomadas antes de que arranque el
# movimiento, o después de que termine).


# IMPORTANTE: el patrón de Dcx se repite (aprox) en cada fila del barrido, así que
# buscar en una ventana MÁS ANCHA que el largo de una fila puede confundir el
# offset real con un múltiplo del período de fila (aliasing). Usar un valor
# bastante MENOR al número de puntos por fila (ver 'pasos_x_por_fila' en tu
# Grafico.py original / el largo de desp_x en calculo_dcs.py). Si esperás
# solo unas pocas muestras muertas al principio/final, con 60 alcanza y sobra.
MAX_OFFSET = 60

def normalizar(v):
    return (v - np.mean(v)) / np.std(v)

def encontrar_offset(v_medido, referencia, max_offset=MAX_OFFSET, offset_centro=0):
    """
    Busca el corrimiento (en muestras) que mejor alinea v_medido con
    referencia, maximizando la correlación de Pearson, en el rango
    [offset_centro - max_offset, offset_centro + max_offset].

    offset > 0  => v_medido tiene 'offset' muestras de sobra al principio
                   (hay que descartarlas).
    offset < 0  => referencia tiene 'offset' muestras de sobra al principio.

    IMPORTANTE: si el patrón de X se repite igual en cada fila (periódico),
    buscar en un rango amplio puede confundir offsets separados por múltiplos
    del largo de fila (todos con correlación casi idéntica). Por eso el
    diagnóstico por tramos busca con max_offset chico, centrado en el offset
    global ya encontrado (que sí se resuelve bien porque usa toda la señal,
    incluyendo las puntas no periódicas del principio/final).
    """
    v_norm = normalizar(v_medido)
    ref_norm = normalizar(referencia)
    n_min = min(len(v_norm), len(ref_norm))

    mejor_offset, mejor_corr = offset_centro, -np.inf
    for offset in range(offset_centro - max_offset, offset_centro + max_offset + 1):
        if offset >= 0:
            a = v_norm[offset:offset + n_min]
            b = ref_norm[:len(a)]
        else:
            b = ref_norm[-offset:-offset + n_min]
            a = v_norm[:len(b)]
        if len(a) < max(50, n_min // 10):
            continue
        corr = np.corrcoef(a, b)[0, 1]
        if corr > mejor_corr:
            mejor_corr, mejor_offset = corr, offset

    return mejor_offset, mejor_corr


corr_sin_alinear = np.corrcoef(
    normalizar(v_x_medido_raw[:min(n_dutys, n_adq)]),
    normalizar(dcx_seq[:min(n_dutys, n_adq)])
)[0, 1]

offset, corr_alineada = encontrar_offset(v_x_medido_raw, dcx_seq)
print(f"Correlación SIN alinear (offset=0): {corr_sin_alinear:.4f}")
print(f"Offset encontrado: {offset} muestras | Correlación alineada: {corr_alineada:.4f}")

if corr_alineada < 0.95:
    print("[!] ADVERTENCIA: la correlación alineada no es tan alta como se esperaría "
          "de un offset fijo simple (>0.99 típicamente). Es probable que haya DRIFT "
          "real a lo largo del barrido y no solo un corrimiento constante -- revisar "
          "el diagnóstico por tramos de abajo con atención.")

# ---- Diagnóstico de deriva: ¿el offset es constante a lo largo del barrido? ----
# Si el offset cambia de forma sistemática entre el principio y el final,
# no alcanza con un corrimiento global constante (indicaría drift de reloj
# entre el osciloscopio y el ESP32, no solo muestras muertas al inicio/final).
# Se busca en una ventana CHICA centrada en el offset global (ver docstring de
# encontrar_offset) para no confundirse con la periodicidad de fila.
MAX_OFFSET_LOCAL = 25
N_SEGMENTOS = 6
largo_seg = min(n_dutys, n_adq) // (N_SEGMENTOS + 1)
print("\nDiagnóstico de deriva por tramos (desvío del offset local respecto al global):")
for k in range(N_SEGMENTOS):
    ini = k * largo_seg
    fin = ini + largo_seg
    if fin >= min(n_dutys, n_adq) or fin - ini < 100:
        continue

    # Pre-desplazamos la ventana de v_medido por el offset global ya encontrado,
    # y buscamos solo un ajuste fino (+/- MAX_OFFSET_LOCAL) alrededor de eso.
    ini_v = max(0, ini + offset - MAX_OFFSET_LOCAL)
    fin_v = min(len(v_x_medido_raw), ini + offset + largo_seg + MAX_OFFSET_LOCAL)
    v_seg = v_x_medido_raw[ini_v:fin_v]
    ref_seg = dcx_seq[ini:fin]

    if len(v_seg) < 100:
        continue

    desvio_rel, corr_local = encontrar_offset(
        v_seg, ref_seg, max_offset=MAX_OFFSET_LOCAL, offset_centro=(ini + offset - ini_v)
    )
    desvio_abs = desvio_rel - (ini + offset - ini_v)  # desvío respecto al offset global
    off_local_absoluto = offset + desvio_abs
    print(f"  Tramo {k+1} (muestras {ini}-{fin}): offset local = {off_local_absoluto} "
          f"(desvío = {desvio_abs:+d}, corr={corr_local:.3f})")

# ---- Aplicar el corrimiento global encontrado ----
if offset >= 0:
    v_x_medido = v_x_medido_raw[offset:]
    scattering = scattering_raw[offset:]
    dcx_ref, dcy_ref = dcx_seq, dcy_seq
    x_real_ref, y_real_ref = x_real, y_real
else:
    v_x_medido = v_x_medido_raw
    scattering = scattering_raw
    dcx_ref, dcy_ref = dcx_seq[-offset:], dcy_seq[-offset:]
    x_real_ref, y_real_ref = x_real[-offset:], y_real[-offset:]

n_final = min(len(scattering), len(dcx_ref))
scattering = scattering[:n_final]
v_x_medido = v_x_medido[:n_final]
x_real = x_real_ref[:n_final]
y_real = y_real_ref[:n_final]
print(f"\nPuntos usados tras alinear y recortar: {n_final}")

# ---- Descartar puntos saturados, manteniendo la correspondencia con X,Y ----
mascara_valida = scattering <= UMBRAL_SATURACION
x_real = x_real[mascara_valida]
y_real = y_real[mascara_valida]
scattering = scattering[mascara_valida]


# ==========================================================================
# 3.6 GRÁFICO DE DIAGNÓSTICO: CH1 medido vs Dcx programado, antes/después
# ==========================================================================

n_muestra_diag = min(600, n_final)  # ventana chica para que se vea el detalle

fig_diag, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

ax1.set_title('Antes de alinear (offset = 0)')
ax1.plot(normalizar(v_x_medido_raw[:n_muestra_diag]), label='CH1 medido (norm.)')
ax1.plot(normalizar(dcx_seq[:n_muestra_diag]), label='Dcx programado (norm.)', alpha=0.7)
ax1.legend()
ax1.grid(alpha=0.3)

ax2.set_title(f'Después de alinear (offset = {offset} muestras)')
ax2.plot(normalizar(v_x_medido[:n_muestra_diag]), label='CH1 medido (norm.)')
ax2.plot(normalizar(dcx_ref[:n_muestra_diag]), label='Dcx programado (norm.)', alpha=0.7)
ax2.legend()
ax2.grid(alpha=0.3)
ax2.set_xlabel('Muestra')

plt.tight_layout()
plt.savefig(f'diagnostico_sincronizacion_{version}.png', dpi=200)
plt.show()


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

plt.savefig(f'reconstruccion_muestra_2D_{version}_corregida.png', dpi=300)
plt.show()
