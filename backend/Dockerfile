# Sử dụng PyTorch chính thức, hỗ trợ cả Mac M1/M2 (ARM64) và Intel (AMD64)
FROM pytorch/pytorch:latest

# Tắt prompt khi cài tzdata
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

# Sao chép danh sách thư viện cần cài đặt
COPY requirements.txt /workspace/requirements.txt

# Cài đặt Python dependencies
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# Cài đặt Detectron2 từ nguồn chính thức (CPU version)
RUN pip install --no-cache-dir torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Thiết lập thư mục làm việc trong container
WORKDIR /workspace

# Sao chép toàn bộ mã nguồn vào container
COPY . /workspace

# Mở cổng API
EXPOSE 8080

# Chạy API khi container khởi động
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]