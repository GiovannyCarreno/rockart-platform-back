# API pic-generator-back

Servicio REST con **FastAPI** que integra generación, segmentación y clasificación de arte rupestre:

1. **Generación de pictogramas** con **StyleGAN2 (ADA)** (PyTorch).
2. **Segmentación y simulación visual** con modelos **ONNX** a 256×256 y 512×512.
3. **Clasificación** Petroglifo / Pictograma con **MobileNetV3** (timm).

## Características

| Endpoint | Descripción |
|----------|-------------|
| **POST `/generateSingle`** | Una imagen generada con semilla fija. |
| **POST `/generateSeveral`** | Varias imágenes con semillas aleatorias. |
| **POST `/comparar`** | Segmenta una imagen subida y compara resultados a 256 vs 512 px. |
| **POST `/clasificar`** | Clasifica una imagen como Petroglifo o Pictograma. |

- **Selección de modelo** en generación (`pictos512`, `pictos512_2`) y segmentación (`mejor_modelo_dinamico`, `modelo_dinamico_gab`).
- **GPU**: PyTorch (GAN y clasificador) y ONNX Runtime (CUDA → DirectML → CPU).
- **CORS** configurado para `http://localhost:5174` (ajústalo en `service.py` si tu front usa otro origen).

## Requisitos previos

- **Python 3.10+** recomendado.
- **CUDA** opcional para PyTorch y ONNX GPU (`onnxruntime-gpu`).
- Archivos necesarios antes de arrancar (o montados en Docker):

| Archivo | Uso |
|---------|-----|
| `modelo/pictos512.pkl` | StyleGAN2 (generación, default) |
| `modelo/pictos512_2.pkl` | StyleGAN2 alternativo |
| `modelo/mejor_modelo_dinamico.onnx` | Segmentación ONNX (default) |
| `modelo/modelo_dinamico_gab.onnx` | Segmentación ONNX con postprocesado avanzado |
| `modelo/best_model_fine_mobilenet.pth` | Clasificador Petroglifo / Pictograma |
| `roca/roca_3.jpg` | Fondo para `mejor_modelo_dinamico` |
| `roca/roca_5.jpg` | Fondo para `modelo_dinamico_gab` |

### Repositorio con Git LFS

Los pesos en `modelo/` pueden estar versionados con **Git LFS**. Tras clonar:

```bash
git lfs install
git lfs pull
```

## Instalación

1. Entrar en el directorio del proyecto.

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

> En CPU sin NVIDIA puedes usar solo `onnxruntime` (sin `-gpu`). El código intenta `CUDAExecutionProvider`, luego `DmlExecutionProvider` (Windows) y cae a CPU.

Las rutas de modelos y fondos se definen en `service.py` (constantes al inicio del archivo y carga en el evento `startup`).

## Ejecución local

```bash
uvicorn service:app --reload --host 0.0.0.0 --port 8000
```

Al iniciar, la consola indica qué modelos se cargaron y qué proveedor ONNX se usa (p. ej. `CUDAExecutionProvider`).

## Docker

Requisitos: [Docker](https://docs.docker.com/get-docker/) y, para GPU, [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

Construir (incluye `modelo/`, `roca/` y código en la imagen):

```bash
docker build -t pictos-gan-api .
```

Construir sin caché:

```bash
docker build --no-cache -t pictos-gan-api .
```

Ejecutar:

```bash
# Con GPU
docker run --gpus all -p 8000:8000 pictos-gan-api

# Solo CPU
docker run -p 8000:8000 pictos-gan-api
```

Montar modelos o fondos desde el host:

```bash
docker run --gpus all -p 8000:8000 -v ./modelo:/app/modelo -v ./roca:/app/roca pictos-gan-api
```

## Endpoints

### POST `/generateSingle`

Cuerpo **JSON**:

```json
{
  "seed": 456,
  "truncation_psi": 1.0,
  "noise_mode": "const",
  "model": "pictos512"
}
```

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `seed` | number | 456 | Semilla (entero internamente). |
| `truncation_psi` | float | 1.0 | Truncación StyleGAN2. |
| `noise_mode` | string | `"const"` | `"const"`, `"random"` o `"none"`. |
| `model` | string | `"pictos512"` | `"pictos512"` o `"pictos512_2"`. |

Respuesta: `image` (PNG base64), `seed`, `truncation_psi`, `noise_mode`, `model`.

### POST `/generateSeveral`

Mismo esquema JSON; **`number`** indica cuántas imágenes generar (default `1`). Las semillas son aleatorias.

Respuesta: `images`, `seeds`, `number`, `truncation_psi`, `noise_mode`, `model`.

### POST `/clasificar`

Entrada **`multipart/form-data`**:

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `imagen` | File | Imagen a clasificar (`png`, `jpg`, `jpeg`, `bmp`, `webp`). |

Respuesta (JSON):

```json
{
  "filename": "ejemplo.png",
  "clase": "Petroglifo",
  "confianza": 0.9234,
  "confianza_porcentaje": 92.34,
  "alta_confianza": true,
  "probabilidades": {
    "Petroglifo": 0.9234,
    "Pictograma": 0.0766
  },
  "clases": ["Petroglifo", "Pictograma"]
}
```

`alta_confianza` es `true` si la confianza es ≥ 75 %.

Ejemplo **cURL**:

```bash
curl -X POST "http://localhost:8000/clasificar" \
  -F "imagen=@ruta/a/tu/imagen.png"
```

### POST `/comparar`

Entrada **`multipart/form-data`**:

| Campo | Tipo | Default | Descripción |
|-------|------|---------|-------------|
| `imagen` | File | — | Imagen a segmentar. |
| `model` | string | `mejor_modelo_dinamico` | `mejor_modelo_dinamico` o `modelo_dinamico_gab`. |

**Comportamiento por modelo:**

- **`mejor_modelo_dinamico`**: inferencia ONNX directa, umbral 0.7, simulación con tono y desgaste sobre `roca_3.jpg`.
- **`modelo_dinamico_gab`**: preprocesado CLAHE, TTA, postprocesado de máscara (`segmentar_petroglifo.py`), render en relieve sobre `roca_5.jpg`.

Salida (JSON):

- `model`: modelo ONNX usado.
- `metricas`: `cobertura_modelo_1` / `cobertura_modelo_2` (256 vs 512), umbrales.
- `imagenes`: `comparacion` (figura 2×3 en base64), `simulacion_modelo_1`, `simulacion_modelo_2`.

Ejemplo **cURL**:

```bash
curl -X POST "http://localhost:8000/comparar" \
  -F "imagen=@ruta/a/tu/imagen.png" \
  -F "model=mejor_modelo_dinamico"
```

```bash
curl -X POST "http://localhost:8000/comparar" \
  -F "imagen=@ruta/a/tu/imagen.png" \
  -F "model=modelo_dinamico_gab"
```

En el repo, `client.py` es un ejemplo mínimo que llama a `/comparar`.

## Estructura del proyecto

```
pic-generator-back/
├── service.py              # FastAPI: endpoints y orquestación
├── segmentar_petroglifo.py # Postprocesado de segmentación (modelo GAB)
├── legacy.py               # Carga de checkpoints StyleGAN2 (.pkl)
├── client.py               # Ejemplo de cliente para /comparar
├── dnnlib/                 # Utilidades StyleGAN2
├── torch_utils/            # Utilidades PyTorch (StyleGAN2)
├── modelo/                 # Pesos (.pkl, .onnx, .pth)
├── roca/                   # Fondos para simulación
├── Dockerfile
├── .dockerignore
├── .gitattributes          # Git LFS (*.pkl, *.onnx, etc.)
├── requirements.txt
├── LICENSE.txt
└── README.md
```

## Tecnologías

- **FastAPI** / **Uvicorn** — API HTTP.
- **PyTorch** / **timm** — StyleGAN2, clasificador MobileNetV3.
- **ONNX Runtime** — segmentación (CUDA / DirectML / CPU).
- **OpenCV**, **Pillow**, **NumPy**, **SciPy**, **scikit-image** — imagen y postprocesado.
- **Matplotlib** — figura de comparación en `/comparar`.

Código y modelo GAN sujetos a la licencia de **StyleGAN2 ADA**; ver `LICENSE.txt`.