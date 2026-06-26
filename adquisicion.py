import pyvisa
import time
import pandas as pd
import matplotlib.pyplot as plt

# --- Configuración Básica ---
RESOURCE_NAME = 'USB0::0x1AB1::0x044C::DHO9S264705563::INSTR'
CANAL = 2

# --- Configuración del Datalogger ---
INTERVALO_SEG = 0.001       # Tiempo de espera entre mediciones (en segundos)
CANTIDAD_PUNTOS = 1000      # Cuántas mediciones tomar en total
ARCHIVO_CSV = 'mapeo_temporal_2.csv'

def adquisicion_continua():
    rm = pyvisa.ResourceManager()
    tiempos = []
    voltajes = []
    
    try:
        # 1. Conexión y purga
        scope = rm.open_resource(RESOURCE_NAME)
        scope.timeout = 5000  
        scope.write('*CLS')
        
        # 2. Activamos la medición explícitamente en pantalla
        scope.write(f':MEASure:ITEM VAVG,CHAN{CANAL}')
        time.sleep(0.5) # Le damos changüí al hardware para que procese la primera vez
        
        print(f"Arrancando adquisición: {CANTIDAD_PUNTOS} puntos cada {INTERVALO_SEG} seg...")
        
        # Guardamos el tiempo exacto en el que arranca el cronómetro
        tiempo_cero = time.time()
        
        # 3. Bucle de muestreo
        for k in range(CANTIDAD_PUNTOS):
            # Le pedimos el promedio al osciloscopio
            respuesta = scope.query(f':MEASure:ITEM? VAVG,CHAN{CANAL}')
            valor_v = float(respuesta.strip())
            
            # Calculamos cuántos segundos pasaron desde que arrancamos
            tiempo_actual = time.time() - tiempo_cero
            
            # Guardamos los datos en las listas
            tiempos.append(tiempo_actual)
            voltajes.append(valor_v)
            
            #print(f"Punto {k+1}/{CANTIDAD_PUNTOS} | Tiempo: {tiempo_actual:.2f} s | V_Promedio: {valor_v:.5f} V")
            
            # Esperamos el intervalo definido antes de la próxima medición
            #time.sleep(INTERVALO_SEG)
            
        # 4. Cerramos conexión al terminar el bucle
        scope.close()
        return tiempos, voltajes

    except Exception as e:
        print(f"Error en la comunicación: {e}")
        return tiempos, voltajes # Devolvemos lo que haya llegado a medir antes del error

if __name__ == '__main__':
    t_datos, v_datos = adquisicion_continua()
    
    if len(t_datos) > 0:
        # Guardamos todo en un archivo CSV
        df = pd.DataFrame({'Tiempo (s)': t_datos, f'Voltaje Medio CH{CANAL} (V)': v_datos})
        df.to_csv(ARCHIVO_CSV, index=False)
        print(f"\n¡Listo! Los {len(t_datos)} datos se guardaron en '{ARCHIVO_CSV}'")
        
        
    else:
        print("No se pudieron recolectar datos.")

