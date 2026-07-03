import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.optimize import minimize

# ==========================================================================
# 0. CONFIGURACIÓN — AJUSTAR ESTOS NOMBRES A LOS ARCHIVOS REALES DE ESTE BARRIDO
# ==========================================================================

fecha_calibracion = '0107'

# CSV con la secuencia de duty cycles programada en el ESP32 para este barrido
archivo_dutys = fr'C:\Users\LEC\Desktop\AdquisicionRigol\dutys_scanning_xy_0107_2000.csv'    # <-- completar con el nombre real

version = 'v5_0107'
archivo_csv_adquisicion = fr'C:\Users\LEC\Desktop\AdquisicionRigol\mapeo_scattering_punto_a_punto_v5_0107.csv'

# Umbral para descartar puntos saturados/espurios de Scattering_CH2 (V)
UMBRAL_SATURACION = 1.0

# Resolución de la grilla de salida (píxeles de la imagen reconstruida)
NUM_PIXELS_X = 300
NUM_PIXELS_Y = 300


# ==========================================================================
# 1. RECONSTRUIR EL MODELO CINEMÁTICO (idéntico a calculo_dcs.py)
# ==========================================================================

promedio_cross_x = pd.read_csv(fr'C:\Users\LEC\Desktop\Laboratorio6\Calibración\Datos_ajuste\ajuste_lin_x_calv1_0107.csv')
promedio_cross_y = pd.read_csv(fr'C:\Users\LEC\Desktop\Laboratorio6\Calibración\Datos_ajuste\ajuste_lin_y_calv1_0107.csv')
                                    
promedio_cross_x = [promedio_cross_x['m'].mean(), 0]
promedio_cross_y = [promedio_cross_y['m'].mean(), 0]

datos_ajuste_x = pd.read_csv(fr'C:\Users\LEC\Desktop\Laboratorio6\Calibración\Datos_ajuste\ajuste_cubico_x_calv1_0107.csv')
datos_ajuste_y = pd.read_csv(fr'C:\Users\LEC\Desktop\Laboratorio6\Calibración\Datos_ajuste\ajuste_cubico_y_calv1_0107.csv')

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
    """Convierte duty cycles (dcx, dcy) a posición física real (X, Y) en micrones."""
    X_fisico = polinomio_x(dcx) + cross_y(polinomio_y(dcy))
    Y_fisico = polinomio_y(dcy) + cross_x(polinomio_x(dcx))
    return X_fisico, Y_fisico


# ==========================================================================
# 2. CARGAR LA SECUENCIA DE DUTY CYCLES PROGRAMADA (posición real objetivo)
# ==========================================================================

df_dutys = pd.read_csv(archivo_dutys)
df_dutys = df_dutys.loc[:, ~df_dutys.columns.str.contains('^Unnamed')]

dcx_seq = df_dutys['Dcx'].values
dcy_seq = df_dutys['Dcy'].values
n_dutys = len(dcx_seq)

x_real_prog, y_real_prog = cinematica_directa(dcx_seq, dcy_seq)


# ==========================================================================
# 3. CARGAR LOS DATOS ADQUIRIDOS (Scattering)
# ==========================================================================

df_adq = pd.read_csv(archivo_csv_adquisicion)
scattering_raw = df_adq['Scattering_CH2 (V)'].values
v_x_medido_raw = df_adq['Monitoreo_X_CH1 (V)'].values
n_adq = len(scattering_raw)

print(f"Puntos en secuencia de duty cycles: {n_dutys} | Puntos adquiridos: {n_adq}")


# ==========================================================================
# 3.5 SINCRONIZACIÓN — MODELO AFÍN (offset + DERIVA DE RELOJ), NO SOLO OFFSET
# ==========================================================================
# *** ESTE ES EL BUG QUE SE DEPURÓ ***
#
# El código original (Grafico_corregido.py) buscaba un único corrimiento
# ENTERO y CONSTANTE entre las muestras adquiridas y la secuencia programada
# (encontrar_offset, con MAX_OFFSET=60). Ese modelo asume que el reloj del
# ESP32 (que define el dwell time de cada duty cycle) y el reloj de
# adquisición (que define cada cuánto se muestrea CH1/CH2) están
# perfectamente sincronizados, y que el único desfasaje posible es un
# corrimiento fijo al principio/final del barrido.
#
# Con los datos reales de este barrido (dutys_scanning_xy_0107_2000.csv +
# mapeo_scattering_punto_a_punto_v5_0107.csv) esa hipótesis es FALSA:
#
#   - El barrido tiene 97 filas de 223 muestras cada una (97*223 = 21631,
#     coincide exacto con el largo de dutys).
#   - Un análisis espectral (FFT / autocorrelación) de CH1 muestra que el
#     período real de fila en la señal ADQUIRIDA es ~225.3 muestras, no 223.
#   - Es decir: hay un desfasaje relativo de reloj de ~1.3% entre ambos
#     sistemas. Sobre el barrido completo (~21630 muestras) eso acumula
#     ~284 muestras de deriva — MÁS QUE UNA FILA ENTERA (223 muestras).
#
# Por eso, con el código original:
#   - El offset hallado (60) queda pegado al borde de MAX_OFFSET (bug: el
#     verdadero óptimo de "mejor offset constante" ni siquiera está dentro
#     del rango de búsqueda).
#   - Incluso buscando sin límite de rango, el mejor offset CONSTANTE posible
#     da una correlación de ~0.15 (se espera >0.95): ningún offset fijo
#     puede alinear a la vez el principio y el final del barrido cuando la
#     deriva acumulada supera una fila completa.
#   - El diagnóstico "por tramos" del código original tampoco lo detecta,
#     porque center a cada tramo en el offset GLOBAL (no arrastra la deriva
#     acumulada de tramos anteriores) y busca en una ventana (±25) más
#     chica que la deriva real por tramo (~32 muestras) -> dan valores
#     erráticos y no monótonos que parecen "ruido" pero en realidad es la
#     búsqueda chocando contra su propio límite.
#
# LA CORRECCIÓN: modelar el índice de adquisición correspondiente a la
# muestra k de la secuencia programada como una función AFÍN,
#     idx_adquisicion(k) = offset0 + escala * k
# con dos parámetros (offset0, escala) en vez de uno solo, y usar
# interpolación (no un simple corte de arrays) para reconstruir el valor de
# CH1/Scattering en cada punto programado. Con esto la correlación CH1-Dcx
# sube de ~0.10 a ~0.97, y el residuo local (revisado en tramos de a ~2000
# muestras) queda dentro de ±4 muestras en todo el barrido — confirma que
# un modelo lineal (offset+escala) alcanza para explicar toda la deriva.

def normalizar(v):
    return (v - np.mean(v)) / np.std(v)


def _correlacion_afin(params, v_medido, referencia, n_ref, n_medido):
    offset0, escala = params
    k = np.arange(n_ref)
    warped = offset0 + escala * k
    valido = (warped >= 0) & (warped <= n_medido - 1)
    if valido.sum() < max(50, n_ref // 20):
        return 0.0
    interp = np.interp(warped[valido], np.arange(n_medido), v_medido)
    return np.corrcoef(normalizar(interp), normalizar(referencia[valido]))[0, 1]


def buscar_alineacion_afin(v_medido, referencia,
                            offset_grid=range(0, 320, 4),
                            escala_grid=np.linspace(0.985, 1.05, 261)):
    """
    Encuentra (offset0, escala) que maximizan la correlación de Pearson entre
    'referencia' (Dcx programado) y 'v_medido' (CH1) bajo el modelo afín
    idx_medido(k) = offset0 + escala*k. Primero grilla gruesa (robusta contra
    máximos locales/aliasing de fila) y después refina con Nelder-Mead.
    """
    n_ref, n_medido = len(referencia), len(v_medido)

    mejor = (-np.inf, None, None)
    for offset0 in offset_grid:
        for escala in escala_grid:
            c = _correlacion_afin((offset0, escala), v_medido, referencia, n_ref, n_medido)
            if c > mejor[0]:
                mejor = (c, offset0, escala)

    _, offset0_ini, escala_ini = mejor

    res = minimize(
        lambda p: -_correlacion_afin(p, v_medido, referencia, n_ref, n_medido),
        x0=[offset0_ini, escala_ini], method='Nelder-Mead',
        options={'xatol': 1e-4, 'fatol': 1e-6, 'maxiter': 2000}
    )
    offset0, escala = res.x
    corr = -res.fun
    return offset0, escala, corr


corr_sin_alinear = np.corrcoef(
    normalizar(v_x_medido_raw[:min(n_dutys, n_adq)]),
    normalizar(dcx_seq[:min(n_dutys, n_adq)])
)[0, 1]
print(f"Correlación SIN alinear (offset=0, escala=1): {corr_sin_alinear:.4f}")

offset0, escala, corr_alineada = buscar_alineacion_afin(v_x_medido_raw, dcx_seq)
print(f"Alineación afín hallada: offset0={offset0:.2f} muestras, escala={escala:.6f} "
      f"({(escala-1)*100:+.2f}% de deriva de reloj)")
print(f"Correlación alineada (modelo afín): {corr_alineada:.4f}")
print(f"Deriva acumulada estimada en todo el barrido: {(escala-1)*n_dutys:.1f} muestras")

if corr_alineada < 0.90:
    print("[!] ADVERTENCIA: incluso con el modelo afín (offset+deriva) la correlación "
          "sigue baja. Puede haber deriva NO lineal (ver diagnóstico por tramos abajo) "
          "o un problema distinto en la señal de monitoreo CH1.")

# ---- Diagnóstico de residuo: ¿el modelo afín deja algún residuo sistemático? ----
# Si el modelo lineal (offset+escala) es correcto, el desfasaje LOCAL residual
# (después de aplicar offset0+escala*k) debería quedar chico y estable en
# todos los tramos del barrido. Si en cambio crece o decrece de forma
# sistemática entre tramos, hay deriva no lineal y este modelo no alcanza.
N_SEGMENTOS = 10
k_prog = np.arange(n_dutys)
warped_global = offset0 + escala * k_prog
valido_global = (warped_global >= 0) & (warped_global <= n_adq - 1)
ch1_interp_global = np.interp(warped_global[valido_global], np.arange(n_adq), v_x_medido_raw)
dcx_valido = dcx_seq[valido_global]

n_val = len(ch1_interp_global)
largo_seg = n_val // N_SEGMENTOS
MAX_OFFSET_LOCAL = 20
print("\nDiagnóstico de residuo por tramos (desfasaje local tras la corrección afín; "
      "debería quedar chico y sin tendencia sistemática):")
for s in range(N_SEGMENTOS):
    ini, fin = s * largo_seg, (s + 1) * largo_seg
    a = normalizar(ch1_interp_global[ini:fin])
    b = normalizar(dcx_valido[ini:fin])
    mejor_c, mejor_l = -np.inf, 0
    for lag in range(-MAX_OFFSET_LOCAL, MAX_OFFSET_LOCAL + 1):
        if lag >= 0:
            aa, bb = a[lag:], b[:len(a[lag:])]
        else:
            bb, aa = b[-lag:], a[:len(b[-lag:])]
        if len(aa) < 50:
            continue
        c = np.corrcoef(aa, bb)[0, 1]
        if c > mejor_c:
            mejor_c, mejor_l = c, lag
    print(f"  Tramo {s+1} (muestras {ini}-{fin} en índice programado): "
          f"desfasaje local residual = {mejor_l:+d}  corr={mejor_c:.3f}")

# ---- Construir las series alineadas por interpolación (reemplaza el corte por offset) ----
scattering_interp = np.interp(warped_global[valido_global], np.arange(n_adq), scattering_raw)
v_x_medido = ch1_interp_global
scattering = scattering_interp
x_real = x_real_prog[valido_global]
y_real = y_real_prog[valido_global]
dcx_ref = dcx_valido

n_final = len(scattering)
print(f"\nPuntos usados tras alinear (interpolación afín): {n_final}")

# ---- Descartar puntos saturados/espurios, manteniendo la correspondencia con X,Y ----
mascara_valida = scattering <= UMBRAL_SATURACION
n_descartados = (~mascara_valida).sum()
if n_descartados:
    print(f"Puntos descartados por saturación/espurios (> {UMBRAL_SATURACION} V): {n_descartados}")
x_real = x_real[mascara_valida]
y_real = y_real[mascara_valida]
scattering = scattering[mascara_valida]
v_x_medido = v_x_medido[mascara_valida]
dcx_ref = dcx_ref[mascara_valida]


# ==========================================================================
# 3.6 GRÁFICO DE DIAGNÓSTICO: CH1 medido vs Dcx programado, antes/después
# ==========================================================================

n_muestra_diag = min(600, n_final)

fig_diag, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

ax1.set_title('Antes de alinear (offset=0, escala=1)')
ax1.plot(normalizar(v_x_medido_raw[:n_muestra_diag]), label='CH1 medido (norm.)')
ax1.plot(normalizar(dcx_seq[:n_muestra_diag]), label='Dcx programado (norm.)', alpha=0.7)
ax1.legend()
ax1.grid(alpha=0.3)

ax2.set_title(f'Después de alinear (offset0={offset0:.1f}, escala={escala:.5f})')
ax2.plot(normalizar(v_x_medido[:n_muestra_diag]), label='CH1 interpolado (norm.)')
ax2.plot(normalizar(dcx_ref[:n_muestra_diag]), label='Dcx programado (norm.)', alpha=0.7)
ax2.legend()
ax2.grid(alpha=0.3)
ax2.set_xlabel('Muestra (índice programado)')

plt.tight_layout()
plt.savefig(f'diagnostico_sincronizacion_{version}.png', dpi=200)
plt.show()


# ==========================================================================
# 4. INTERPOLAR A UNA GRILLA REGULAR Y GRAFICAR — SIN CAMBIOS
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
plt.title('Reconstrucción 2D corregida (modelo cinemático directo + sync afín)')
plt.xlabel('Posición X ($\\mu$m)')
plt.ylabel('Posición Y ($\\mu$m)')
plt.grid(False)

plt.savefig(f'reconstruccion_muestra_2D_{version}_corregida.png', dpi=300)
plt.show()
