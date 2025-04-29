# Usar una imagen base de Python
FROM python:3.9

# Establecer el directorio de trabajo en el contenedor
WORKDIR /app

# Copiar los archivos necesarios al contenedor
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt


# Comando para iniciar la aplicación (dependiendo del proyecto)
CMD ["python", "app.py"]
