# Sử dụng PyTorch phù hợp với Mac M1/M2 (ARM64) hoặc Intel (AMD64)
FROM pytorch/pytorch:latest

# Tắt prompt yêu cầu nhập múi giờ khi cài đặt tzdata
ENV DEBIAN_FRONTEND=noninteractive

# Cài đặt các thư viện hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    gcc \
    g++ \
    python3-opencv \
    libgl1-mesa-glx \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt PyTorch và các thư viện cần thiết từ requirements.txt
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# Cài đặt Detectron2 TÙY THEO kiến trúc CPU/GPU
RUN pip install --no-cache-dir torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Thiết lập thư mục làm việc trong container
WORKDIR /workspace

# Sao chép toàn bộ mã nguồn vào container
COPY . /workspace

# Mở cổng API
EXPOSE 8000

# Chạy API khi container khởi động
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
