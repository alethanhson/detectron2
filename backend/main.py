import os
import shutil
import time
import logging
from fastapi import FastAPI, Form
from fastapi.responses import FileResponse, JSONResponse
from google_sheets import get_image_links
from downloader import download_images
from detectron2_to_labelme import process_images
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

# Cấu hình logging
logging.basicConfig(
    filename="process.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

error_logger = logging.getLogger("error")
error_logger.setLevel(logging.ERROR)
error_handler = logging.FileHandler("error.log")
error_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
error_logger.addHandler(error_handler)

@app.post("/process/")
async def process_images_from_sheets(sheet_id: str = Form(...), worksheet_name: str = Form(...)):
    """
    API nhập link Google Sheet -> Tải ảnh -> Chạy Detectron2 -> Tạo file ZIP -> Trả về file ZIP.
    """
    # truncate folder `images/` and `outputs/`
    shutil.rmtree("images", ignore_errors=True)
    shutil.rmtree("outputs", ignore_errors=True)

    try:
        logging.info("🔹 Bắt đầu quá trình xử lý...")
        logging.info(f"📌 Sheet ID: {sheet_id}, Worksheet Name: {worksheet_name}")

        print("🔹 Lấy danh sách ảnh từ Google Sheet...")
        image_urls = get_image_links(sheet_id, worksheet_name)

        print("🔹 Tải ảnh về thư mục `images/`...")
        await download_images(image_urls)

        print("🔹 Chạy Detectron2 để tạo dữ liệu LabelMe...")
        process_images()

        zip_filename = "outputs.zip"
        zip_filepath = os.path.join(os.getcwd(), zip_filename)

        print("🔹 Nén thư mục `outputs/` thành file ZIP...")
        shutil.make_archive("outputs", "zip", "outputs")

        logging.info(f"✔ Xử lý hoàn tất! Trả về file ZIP: {zip_filepath}")
        print(f"✔ Xử lý hoàn tất! Trả về file ZIP: {zip_filepath}")

        return FileResponse(zip_filepath, filename=zip_filename, media_type="application/zip")
    except Exception as e:
        error_logger.error(f"❌ Lỗi trong quá trình xử lý: {e}")
        print(f"❌ Lỗi xảy ra: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
