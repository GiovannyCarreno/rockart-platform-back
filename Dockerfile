# Imagen base con PyTorch y CUDA 11.8 (incluye torch, torchvision, torchaudio)
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# Variables de entorno para evitar prompts interactivos
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    IOPAINT_DEVICE=cuda

# Instalar dependencias del sistema necesarias para algunos paquetes Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements y excluir torch/torchvision/torchaudio (ya vienen en la imagen base)
COPY requirements.txt .
RUN grep -vE '^(torch|torchvision|torchaudio)' requirements.txt > requirements-docker.txt && \
    pip install --no-cache-dir -r requirements-docker.txt

# Instalar iopaint
RUN pip install iopaint==1.6.0

# Copiar código de la aplicación
COPY legacy.py .
COPY service.py .
COPY segmentar_petroglifo.py .
COPY docker-entrypoint.sh .
COPY dnnlib/ ./dnnlib/
COPY torch_utils/ ./torch_utils/
COPY modelo/ ./modelo/
COPY roca/ ./roca/

RUN chmod +x docker-entrypoint.sh

EXPOSE 8000 8080

# FastAPI (8000) + IOPaint LaMa (8080)
ENTRYPOINT ["./docker-entrypoint.sh"]
