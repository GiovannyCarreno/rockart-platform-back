# API de Generación de Imágenes con StyleGAN2

Servicio REST basado en FastAPI para generar imágenes usando un modelo StyleGAN2 entrenado (pictos512). El proyecto utiliza PyTorch y la arquitectura StyleGAN2 con Adaptive Discriminator Augmentation (ADA) de NVIDIA.

## Características

- **Generación única**: Produce una imagen a partir de una semilla (seed) específica
- **Generación múltiple**: Genera varias imágenes con semillas aleatorias en una sola petición
- **Configurable**: Parámetros de truncación y modo de ruido ajustables
- **GPU acelerado**: Usa CUDA automáticamente si está disponible

## Requisitos previos

- Python 3.8 o superior
- CUDA (opcional, para aceleración con GPU)
- Modelo preentrenado en `modelo/pictos512.pkl`

## Instalación

1. **Clonar o descargar el proyecto** y entrar en el directorio:

```bash
cd version 3
```

2. **Crear un entorno virtual** (recomendado):

```bash
python -m venv .venv
.venv\Scripts\activate    # Windows
# source .venv/bin/activate  # Linux/macOS
```

3. **Instalar dependencias**:

```bash
pip install -r requirements.txt
```

4. **Verificar el modelo**: Asegúrate de que el archivo del modelo esté en `modelo/pictos512.pkl`. Puedes ajustar la ruta en `service.py` (línea 53) si usas otra ubicación.

## Ejecución con Docker

Si prefieres ejecutar el proyecto en un contenedor, necesitas [Docker](https://docs.docker.com/get-docker/) instalado y, para GPU, [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

### Construir la imagen

```bash
docker build -t pictos-gan-api .
```

> **Nota**: Asegúrate de tener el archivo `modelo/pictos512.pkl` en la carpeta del proyecto antes del build.

### Ejecutar el contenedor

**Con GPU (recomendado):**
```bash
docker run --gpus all -p 8000:8000 pictos-gan-api
```

**Solo CPU:**
```bash
docker run -p 8000:8000 pictos-gan-api
```

**Montar el modelo como volumen** (si no está incluido en la imagen):
```bash
docker run --gpus all -p 8000:8000 -v ./modelo:/app/modelo pictos-gan-api
```

El servidor quedará disponible en `http://localhost:8000`.

## Uso

### Iniciar el servidor

```bash
uvicorn service:app --reload --host 0.0.0.0 --port 8000
```

El servidor estará disponible en `http://localhost:8000`.

### Documentación interactiva

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Endpoints de la API

### POST `/generateSingle`

Genera una única imagen con los parámetros indicados.

**Cuerpo de la petición (JSON):**

```json
{
  "seed": 456,
  "truncation_psi": 1.0,
  "noise_mode": "const",
  "number": 1
}
```

| Parámetro        | Tipo   | Default  | Descripción                                      |
|------------------|--------|----------|--------------------------------------------------|
| seed             | float  | 456      | Semilla para reproducibilidad                    |
| truncation_psi   | float  | 1.0      | Factor de truncación (afecta la variación)       |
| noise_mode       | string | "const"  | Modo de ruido: `"const"`, `"random"`, `"none"`   |
| number           | int    | 1        | (No usado en este endpoint)                      |

**Respuesta:**

```json
{
  "image": "<base64_encoded_png>",
  "seed": 456,
  "truncation_psi": 1.0,
  "noise_mode": "const"
}
```

### POST `/generateSeveral`

Genera varias imágenes con semillas aleatorias.

**Cuerpo de la petición (JSON):**

```json
{
  "seed": 456,
  "truncation_psi": 1.0,
  "noise_mode": "const",
  "number": 5
}
```

| Parámetro        | Tipo   | Descripción                                 |
|------------------|--------|---------------------------------------------|
| number           | int    | **Requerido.** Cantidad de imágenes a generar |
| truncation_psi   | float  | Factor de truncación                        |
| noise_mode       | string | Modo de ruido                               |

**Respuesta:**

```json
{
  "number": 5,
  "images": ["<base64_1>", "<base64_2>", ...],
  "seeds": [123456, 789012, ...],
  "truncation_psi": 1.0,
  "noise_mode": "const"
}
```

## Ejemplo de uso con cURL

```bash
# Generar una imagen con semilla fija
curl -X POST "http://localhost:8000/generateSingle" \
  -H "Content-Type: application/json" \
  -d '{"seed": 12345}'

# Generar 3 imágenes
curl -X POST "http://localhost:8000/generateSeveral" \
  -H "Content-Type: application/json" \
  -d '{"number": 3}'
```

## Estructura del proyecto

```
version 3/
├── service.py        # Servidor FastAPI principal
├── legacy.py         # Carga de modelos StyleGAN2
├── dnnlib/           # Librería interna de StyleGAN2
├── torch_utils/      # Utilidades PyTorch
├── modelo/           # Carpeta para el modelo (.pkl)
├── Dockerfile        # Definición de la imagen Docker
├── .dockerignore     # Archivos excluidos al construir la imagen
├── requirements.txt
├── LICENSE.txt
└── README.md
```

## Tecnologías

- **FastAPI** – Framework web
- **PyTorch** – Motor de inferencia
- **StyleGAN2 (ADA)** – Arquitectura del modelo
- **PIL/Pillow** – Procesamiento de imágenes

## Licencia

Este proyecto utiliza código bajo la [NVIDIA Source Code License for StyleGAN2 ADA](LICENSE.txt).