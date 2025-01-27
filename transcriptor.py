import whisper
import os

def transcribir_audio(ruta_audio, modelo="base"):
    try:
        # Verificar si el archivo de audio existe
        if not os.path.exists(ruta_audio):
            raise FileNotFoundError(f"Archivo de audio no encontrado: {ruta_audio}")

        # Cargar modelo
        model = whisper.load_model(modelo)

        # Transcribir
        resultado = model.transcribe(ruta_audio)
        texto = resultado.get("text", "").strip()

        if not texto:
            raise ValueError("La transcripción está vacía. ¿El audio tiene voz clara?")

        # Guardar en el mismo directorio del script
        output_path = os.path.join(os.path.dirname(__file__), "transcripcion.txt")

        with open(output_path, "w", encoding="utf-8") as archivo:
            archivo.write(texto)

        print(f"Transcripción exitosa. Archivo guardado en: {output_path}")
        print("Texto extraído (primeros 100 caracteres):", texto[:100] + "...")

    except Exception as e:
        print(f"Error: {str(e)}")

# Ejemplo de uso (¡cambia la ruta!)
ruta_audio = os.path.join(os.path.dirname(__file__), "sin nombre.wav")  # o .wav
transcribir_audio(ruta_audio, modelo="small")  # Usar modelo 'small' para mejor precisión
