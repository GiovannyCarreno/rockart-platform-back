# API pic-generator-back

Servicio REST con **FastAPI** que combina:

1. **Generación de pictogramas** con **StyleGAN2 (ADA)** y el checkpoint `pictos512.pkl` (PyTorch).
2. **Segmentación y simulación visual** con un modelo en **ONNX** a dos resoluciones (256×256 y 512×512), útil para comparar cobertura y resultado sobre un fondo de referencia.

## Características

- **POST `/generateSingle`**: una imagen generada a partir de una **semilla** fija.
- **POST `/generateSeveral`**: varias imágenes con **semillas aleatorias** en una sola petición.
- **POST `/comparar`**: sube una imagen (`multipart/form-data`) y devuelve métricas de cobertura y PNG en base64 (comparación 256 vs 512, máscaras y simulaciones).
- **GPU**: PyTorch (GAN) y **ONNX Runtime** usan CUDA cuando está disponible.
- **CORS** configurado para `http://localhost:5174` (ajústalo en `service.py` si tu front usa otro origen).

## Requisitos previos

- **Python 3.10+** recomendado (el `requirements.txt` fija versiones concretas; en Windows suele usarse un venv dedicado).
- **CUDA** opcional para el GAN y para ONNX GPU (`onnxruntime-gpu`).
- Archivos presentes antes de arrancar (o montados en Docker):
  - `modelo/pictos512.pkl` — StyleGAN2.
  - `modelo/mejor_modelo_dinamico.onnx` — segmentación.
  - `roca/roca_3.jpg` — imagen de fondo usada en la simulación.

### Repositorio con Git LFS

Los pesos en `modelo/` pueden estar versionados con **Git LFS**. Tras clonar:

```bash
git lfs install
git lfs pull
```

## Instalación

1. Entrar en el directorio del proyecto (por ejemplo la carpeta `version 3`).

2. Crear y activar un entorno virtual:

```bash
python -m venv .venv
.venv\Scripts\activate    # Windows
# source .venv/bin/activate  # Linux/macOS
```

3. Instalar dependencias:

```bash
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

> En CPU sin NVIDIA puedes instalar solo `onnxruntime` y omitir el paquete GPU según tu entorno; el código intenta `CUDAExecutionProvider` y cae a CPU si no está.

Las rutas de modelo y fondo se definen en `service.py` (`ONNX_MODEL_PATH`, `RUTA_FONDO` y carga del `.pkl` en el evento `startup`).

## Ejecución local

```bash
uvicorn service:app --reload --host 0.0.0.0 --port 8000
```

Documentación interactiva: **http://localhost:8000/docs**

## Docker

Requisitos: [Docker](https://docs.docker.com/get-docker/) y, para GPU, [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

Construir (incluye `modelo/` y `roca/` en la imagen):

```bash
docker build -t pictos-gan-api .
```

Ejecutar:

```bash
# Con GPU
docker run --gpus all -p 8000:8000 pictos-gan-api

# Solo CPU
docker run -p 8000:8000 pictos-gan-api
```

Montar modelos o fondo desde el host si no van dentro de la imagen:

```bash
docker run --gpus all -p 8000:8000 -v ./modelo:/app/modelo -v ./roca:/app/roca pictos-gan-api
```

## Endpoints

### POST `/generateSingle`

Cuerpo JSON (ejemplo):

```json
{
  "seed": 456,
  "truncation_psi": 1.0,
  "noise_mode": "const",
  "number": 1
}
```

| Parámetro        | Tipo   | Default  | Descripción |
|------------------|--------|----------|-------------|
| `seed`           | number | 456      | Semilla (se convierte a entero internamente). |
| `truncation_psi` | float  | 1.0      | Truncación StyleGAN2. |
| `noise_mode`     | string | `"const"`| `"const"`, `"random"` o `"none"`. |
| `number`         | int    | 1        | No aplica a este endpoint. |

Respuesta: `image` (PNG en base64), `seed`, `truncation_psi`, `noise_mode`.

### POST `/generateSeveral`

Mismo esquema de cuerpo; **`number`** es la cantidad de imágenes (por defecto `1`). Las semillas se eligen al azar por imagen.

Respuesta: `images`, `seeds`, `number`, `truncation_psi`, `noise_mode`.

### POST `/comparar`

- **Entrada**: `multipart/form-data`, campo de archivo **`imagen`** (`png`, `jpg`, `jpeg`, `bmp`, `webp`).
- **Salida** (JSON):
  - `metricas`: `cobertura_modelo_1` / `cobertura_modelo_2` (256 vs 512), umbrales usados.
  - `imagenes`: `comparacion` (figura 2×3 en base64), `simulacion_modelo_1`, `simulacion_modelo_2`.

Ejemplo con **cURL**:

```bash
curl -X POST "http://localhost:8000/comparar" \
  -F "imagen=@ruta/a/tu/imagen.png"
```

En el repo, `client.py` es un ejemplo mínimo que llama a `/comparar` y muestra las imágenes.

## Estructura del proyecto

```
version 3/
├── service.py          # FastAPI: arranque, GAN, ONNX, /comparar
├── legacy.py           # Carga del .pkl StyleGAN2
├── client.py           # Ejemplo de cliente para /comparar
├── dnnlib/             # Utilidades StyleGAN2
├── torch_utils/        # Utilidades PyTorch (StyleGAN2)
├── modelo/             # pictos512.pkl, mejor_modelo_dinamico.onnx
├── roca/               # Recursos de fondo (p. ej. roca_3.jpg)
├── Dockerfile
├── .dockerignore
├── .gitattributes      # Reglas Git LFS para *.pkl, *.onnx, etc.
├── requirements.txt
├── LICENSE.txt
└── README.md
```

## Tecnologías

- **FastAPI** / **Uvicorn** — API HTTP.
- **PyTorch** — inferencia del generador StyleGAN2.
- **ONNX Runtime** — segmentación dinámica (GPU/CPU según instalación).
- **OpenCV**, **Pillow**, **NumPy**, **SciPy** — imagen y postprocesado.
- **Matplotlib** — figura de comparación en `/comparar`.

Código y modelo GAN sujetos a la licencia de **StyleGAN2 ADA**; ver `LICENSE.txt`.
