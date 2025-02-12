# Sử dụng Node.js 18
FROM node:18

# Thiết lập thư mục làm việc
WORKDIR /app

# Sao chép file package.json vào container
COPY package.json package-lock.json ./

# Cài đặt dependencies
RUN npm install

# Sao chép toàn bộ mã nguồn vào container
COPY . .

# Biến môi trường cho Vite
ENV VITE_BACKEND_URL=http://localhost:8080

# Build ứng dụng
RUN npm run build

# Dùng nginx để serve file frontend
FROM nginx:alpine
COPY --from=0 /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]