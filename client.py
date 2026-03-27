# Cliente Python de ejemplo
import requests, base64, PIL.Image, io

r = requests.post(
    "http://localhost:8000/comparar",
    files={"imagen": open("Captura de pantalla 2026-03-24 165005.png", "rb")}
)
data = r.json()
print(data["metricas"])

# Decodificar y mostrar las imágenes
for key, b64 in data["imagenes"].items():
    img = PIL.Image.open(io.BytesIO(base64.b64decode(b64)))
    img.show(title=key)