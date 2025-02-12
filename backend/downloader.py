import os
import time
import aiofiles
import httpx

# Định nghĩa thư mục lưu ảnh
IMAGE_DIR = "images"
os.makedirs(IMAGE_DIR, exist_ok=True)

async def download_images(image_urls):
    """
    Tải ảnh từ danh sách URL và lưu vào thư mục `images/`.

    Args:
        image_urls (List[str]): Danh sách URL ảnh.

    Returns:
        List[str]: Danh sách đường dẫn ảnh đã tải.
    """
    image_paths = []

    async with httpx.AsyncClient() as client:
        for idx, link in enumerate(image_urls):
            try:
                response = await client.get(link)
                response.raise_for_status()
                file_name = f"image_{idx}_{int(time.time())}.jpg"
                file_path = os.path.join(IMAGE_DIR, file_name)

                async with aiofiles.open(file_path, "wb") as file:
                    await file.write(response.content)

                image_paths.append(file_path)
                print(f"✔ Ảnh đã tải: {file_path}")

            except httpx.HTTPStatusError as e:
                print(f"❌ HTTP error khi tải {link}: {e}")
            except httpx.RequestError as e:
                print(f"⚠️ Lỗi request {link}: {e}")

    return image_paths
