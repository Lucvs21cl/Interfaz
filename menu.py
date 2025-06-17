import pandas as pd
import numpy as np
import joblib
import os

def obtener_datos_pasajero():
    print("\n=== Ingrese los datos del pasajero ===")
    
    # Datos básicos
    edad = int(input("Edad: "))
    genero = input("Género (Male/Female): ").lower()
    tipo_cliente = input("Tipo de Cliente (Loyal Customer/disloyal Customer): ").lower()
    tipo_viaje = input("Tipo de Viaje (Business travel/Personal Travel): ").lower()
    clase = input("Clase (Business/Eco): ").lower()
    distancia_vuelo = float(input("Distancia del vuelo: "))
    
    # Servicios (0-5)
    print("\nCalifique los siguientes servicios (0-5):")
    wifi = int(input("Servicio de WiFi a bordo: "))
    tiempo_conveniente = int(input("Conveniencia de horario de salida/llegada: "))
    reserva_online = int(input("Facilidad de reserva online: "))
    ubicacion_puerta = int(input("Ubicación de la puerta: "))
    comida = int(input("Comida y bebida: "))
    embarque_online = int(input("Embarque online: "))
    comodidad_asiento = int(input("Comodidad del asiento: "))
    entretenimiento = int(input("Entretenimiento a bordo: "))
    servicio_bordo = int(input("Servicio a bordo: "))
    espacio_piernas = int(input("Servicio de espacio para piernas: "))
    equipaje = int(input("Manejo de equipaje: "))
    checkin = int(input("Servicio de check-in: "))
    servicio_vuelo = int(input("Servicio durante el vuelo: "))
    limpieza = int(input("Limpieza: "))
    
    # Retrasos
    retraso_salida = float(input("Retraso en la salida (minutos): "))
    
    # Crear diccionario con los datos en el orden correcto
    datos = {
        'Age': edad,
        'Gender': 1 if genero == 'male' else 0,
        'Customer Type': 0 if tipo_cliente == 'loyal customer' else 1,
        'Type of Travel': 0 if tipo_viaje == 'business travel' else 1,
        'Class': 0 if clase == 'business' else 1,
        'Flight Distance': distancia_vuelo,
        'Inflight wifi service': wifi,
        'Departure/Arrival time convenient': tiempo_conveniente,
        'Ease of Online booking': reserva_online,
        'Gate location': ubicacion_puerta,
        'Food and drink': comida,
        'Online boarding': embarque_online,
        'Seat comfort': comodidad_asiento,
        'Inflight entertainment': entretenimiento,
        'On-board service': servicio_bordo,
        'Leg room service': espacio_piernas,
        'Baggage handling': equipaje,
        'Checkin service': checkin,
        'Inflight service': servicio_vuelo,
        'Cleanliness': limpieza,
        'Departure Delay in Minutes': retraso_salida,
        'satisfaction': 1
    }
    
    # Crear DataFrame y asegurar el orden de las columnas
    df = pd.DataFrame([datos])
    
    # Ordenar las columnas según el orden del modelo
    columnas_ordenadas = [
        'Age', 'Gender', 'Customer Type', 'Type of Travel', 'Class',
        'Flight Distance', 'Inflight wifi service', 'Departure/Arrival time convenient',
        'Ease of Online booking', 'Gate location', 'Food and drink',
        'Online boarding', 'Seat comfort', 'Inflight entertainment',
        'On-board service', 'Leg room service', 'Baggage handling',
        'Checkin service', 'Inflight service', 'Cleanliness',
        'Departure Delay in Minutes', 'satisfaction'
    ]
    
    return df[columnas_ordenadas]

def clasificar_pasajero():
    try:
        # Obtener la ruta del directorio actual
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Cargar el modelo y el scaler
        model = joblib.load(os.path.join(current_dir, 'kmeans_model.joblib'))
        scaler = joblib.load(os.path.join(current_dir, 'scaler.joblib'))
        pca = joblib.load(os.path.join(current_dir, 'pca.joblib'))
        columnas = joblib.load(os.path.join(current_dir, 'columnas.joblib'))
        
        # Obtener datos del pasajero
        datos_pasajero = obtener_datos_pasajero()
        
        # Asegurar que las columnas estén en el orden correcto
        datos_pasajero = datos_pasajero[columnas]
        
        # Preprocesar los datos
        datos_escalados = scaler.transform(datos_pasajero)
        datos_pca = pca.transform(datos_escalados)
        
        # Predecir el cluster
        cluster = model.predict(datos_pca)[0]
        
        # Mostrar resultado
        print("\n=== Resultado de la clasificación ===")
        print(f"El pasajero pertenece al Grupo {cluster}")
        
        # Mostrar descripción del grupo
        descripciones = {
            0: "Viajeros de negocios frecuentes/leales - Mayor satisfacción general, menos retrasos",
            1: "Público joven - Más crítico con los servicios, más retrasos",
            2: "Viajeros de negocios infrecuentes - Crítico con aspectos logísticos y tecnológicos",
            3: "Viajeros de vacaciones familiares infrecuentes - Enfocado en servicios y comodidades"
        }
        print(f"\nDescripción del grupo: {descripciones[cluster]}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Asegúrese de que el archivo knn_classifier.py se haya ejecutado primero para generar los modelos.")

def main():
    while True:
        print("\n=== Menú de Clasificación de Pasajeros ===")
        print("1. Clasificar nuevo pasajero")
        print("2. Salir")
        
        opcion = input("\nSeleccione una opción: ")
        
        if opcion == "1":
            clasificar_pasajero()
        elif opcion == "2":
            print("¡Hasta luego!")
            break
        else:
            print("Opción no válida. Por favor, intente de nuevo.")

if __name__ == "__main__":
    main() 