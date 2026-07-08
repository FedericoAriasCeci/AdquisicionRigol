import pyvisa
import time
import pandas as pd

# --- Configuración Básica ---
RESOURCE_NAME = 'USB0::0x1AB1::0x044C::DHO9S264705563::INSTR'
CH_X = 1        # Canal 1: Monitoreo de Duty Cycles enviado a X
CH_LUZ = 2      # Canal 2: Intensidad del fotodiodo (Scattering)

# --- Configuración del Muestreo Coincidente con el ESP32 ---
CANTIDAD_PUNTOS = 30000     # Total de puntos del barrido 2D
INTERVALO_SEG = 0.05         # Tiempo estimado entre puntos (pueden ajustarlo)

hora = time.strftime("%d%m_%H%M")

ARCHIVO_CSV = f'mapeo_scattering_punto_a_punto_{hora}.csv'

def adquisicion_punto_a_punto():
    rm = pyvisa.ResourceManager()
    tiempos = []
    voltajes_x = []
    voltajes_luz = []
    scope = None
    
    try:
        # 1. Conexión y limpieza
        scope = rm.open_resource(RESOURCE_NAME)
        scope.timeout = 5000  
        scope.write('*CLS')
        
        # 2. Habilitamos los canales y activamos la medición de promedio (VAVG) en pantalla
        scope.write(f':MEASure:ITEM VAVG,CHANnel{CH_X}')
        scope.write(f':MEASure:ITEM VAVG,CHANnel{CH_LUZ}')
        time.sleep(0.5) # Pausa técnica para que el firmware del RIGOL procese los comandos
        
        print(f"Preparado para adquirir {CANTIDAD_PUNTOS} puntos.")
        print("--> ¡Iniciá el barrido desde Thonny / ESP32 AHORA! <--")
        print("*(Podés interrumpir en cualquier momento con Ctrl+C y se guardará lo acumulado)*\n")
        
        # Guardamos el instante exacto de inicio
        tiempo_cero = time.time()
        
        # 3. Bucle de muestreo punto a punto
        for k in range(CANTIDAD_PUNTOS):
            t_inicio_punto = time.time()
            
            # Consultamos el voltaje promedio de la posición X (CH1)
            resp_x = scope.query(f':MEASure:ITEM? VAVG,CHANnel{CH_X}')
            v_x = float(resp_x.strip())
            
            # Consultamos el voltaje promedio de la luz (CH2)
            resp_luz = scope.query(f':MEASure:ITEM? VAVG,CHANnel{CH_LUZ}')
            v_luz = float(resp_luz.strip())
            
            # Registramos el tiempo transcurrido
            tiempo_actual = time.time() - tiempo_cero
            
            # Guardamos en las listas
            tiempos.append(tiempo_actual)
            voltajes_x.append(v_x)
            voltajes_luz.append(v_luz)
            
            # Imprimimos progreso cada 100 puntos para saber que sigue vivo
            if (k + 1) % 100 == 0:
                print(f"Progreso: {k+1}/{CANTIDAD_PUNTOS} puntos | X: {v_x:.4f}V | Luz: {v_luz:.5f}V")
            
            # Sincronización temporal activa
            tiempo_transcurrido_punto = time.time() - t_inicio_punto
            tiempo_espera = INTERVALO_SEG - tiempo_transcurrido_punto

            if tiempo_espera > 0:
                time.sleep(tiempo_espera)
            
    except KeyboardInterrupt:
        print(f"\n[!] Adquisición interrumpida por el usuario (Ctrl+C).")
    except Exception as e:
        print(f"\n[!] Error en la comunicación durante la adquisición: {e}")
    finally:
        # 4. Cerramos conexión de forma segura si llegó a abrirse
        if scope is not None:
            try:
                scope.close()
            except:
                pass
        
    return tiempos, voltajes_x, voltajes_luz

if __name__ == '__main__':
    t_datos, v_x_datos, v_luz_datos = adquisicion_punto_a_punto()
    
    if len(t_datos) > 0:
        # Guardamos la matriz completa (o lo que se haya llegado a rescatar)
        df = pd.DataFrame({
            'Tiempo (s)': t_datos, 
            'Monitoreo_X_CH1 (V)': v_x_datos, 
            'Scattering_CH2 (V)': v_luz_datos
        })
        df.to_csv(ARCHIVO_CSV, index=False)
        print(f"\n¡Proceso finalizado! Se guardaron {len(t_datos)} puntos en '{ARCHIVO_CSV}'")
    else:
        print("No se pudieron recolectar datos.")