# Imagen base con PyTorch y CUDA 11.8 (incluye torch, torchvision, torchaudio)
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# Variables de entorno para evitar prompts interactivos
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Instalar dependencias del sistema necesarias para algunos paquetes Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements y excluir torch/torchvision/torchaudio (ya vienen en la imagen base)
COPY requirements.txt .
RUN grep -vE '^(torch|torchvision|torchaudio)' requirements.txt > requirements-docker.txt && \
    pip install --no-cache-dir -r requirements-docker.txt

# Copiar código de la aplicación
COPY legacy.py .
COPY service.py .
COPY dnnlib/ ./dnnlib/
COPY torch_utils/ ./torch_utils/
COPY modelo/ ./modelo/
COPY roca/ ./roca/

EXPOSE 8000

# Iniciar el servidor FastAPI
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000"]