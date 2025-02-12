import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Định nghĩa phạm vi truy cập Google Sheets API
scope = ["https://spreadsheets.google.com/feeds",
         "https://www.googleapis.com/auth/drive"]

# Xác thực tài khoản Google
creds = ServiceAccountCredentials.from_json_keyfile_name(
    "./client_secret.json", scope)

def get_gspread_client():
    """Trả về client gspread đã xác thực."""
    return gspread.authorize(creds)

def get_image_links(sheet_id: str, worksheet_name: str):
    """
    Lấy danh sách link ảnh từ Google Sheets.

    Args:
        sheet_id (str): ID của Google Sheet.
        worksheet_name (str): Tên worksheet.

    Returns:
        List[str]: Danh sách URL ảnh.
    """
    client = get_gspread_client()
    sheet = client.open_by_key(sheet_id)
    worksheet = sheet.worksheet(worksheet_name)

    # Đọc dữ liệu từ cột đầu tiên (bỏ qua dòng tiêu đề)
    links = worksheet.col_values(1)[1:]

    # Đảm bảo tất cả link có tiền tố HTTP/S hợp lệ
    links = [link if link.startswith("http://") or link.startswith("https://") else "http://" + link for link in links]

    return links
