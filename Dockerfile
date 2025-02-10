# Sử dụng image PyTorch phù hợp với Mac M1/M2 (ARM64) hoặc Intel (AMD64)
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

# Kiểm tra Git đã được cài đặt chưa
RUN git --version

# Cài đặt PyTorch và các thư viện cần thiết
RUN pip install --no-cache-dir torch torchvision torchaudio numpy tqdm opencv-python pillow matplotlib albumentations ultralytics

# Cài đặt Detectron2 từ GitHub
RUN pip install --no-cache-dir 'git+https://github.com/facebookresearch/detectron2.git'

# Thiết lập thư mục làm việc trong container
WORKDIR /workspace

# Sao chép toàn bộ mã nguồn vào container
COPY . /workspace

# Thiết lập lệnh mặc định khi chạy container
CMD ["python", "run_script.py"]
