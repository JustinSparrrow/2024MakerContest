import requests
from PIL import Image
from io import BytesIO

def upload_image(image_path, server_url):
    with open(image_path, 'rb') as img_file:
        files = {'file': img_file}
        
        response = requests.post(f"{server_url}/upload", files=files)
        
        if response.status_code == 200:
            data = response.json()
            download_link = data['download_link']
            qr_code_url = data['qr_code']
            
            print(f"Download Link: { download_link}")
            
            # 获取并且显示二维码图片
            qr_code_response = requests.get(qr_code_url)
            qr_code_image = Image.open(BytesIO(qr_code_response.content))
            qr_code_image.show()
            
            qr_code_image.save("qr_code.png")
            
            print("QR code saved as qr_code.png")
        else:
            print(f"Failed to upload image. Status code: {response.status_code}")
            print(response.json)
            
if __name__ == '__main__':
    image_path = "/Users/moqi/Desktop/竞赛/2024创客赛/me.jpg"
    server_url = "http://127.0.0.1:8080"
    
    upload_image(image_path, server_url)