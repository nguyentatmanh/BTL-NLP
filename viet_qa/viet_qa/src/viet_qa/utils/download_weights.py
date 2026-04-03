import gdown
import os
import zipfile
import shutil

def download_model():
    # Google Drive File ID
    file_id = '1viAih2eZk7X8C1BO7YW4zh_fusDfrzcA'
    url = f'https://drive.google.com/uc?id={file_id}'
    
    # Path to extract
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dir = os.path.join(base_dir, "checkpoints", "extractive")
    tmp_zip = os.path.join(base_dir, "model_weights.zip")
    
    if not os.path.exists(os.path.dirname(target_dir)):
        os.makedirs(os.path.dirname(target_dir))
        
    print(f"🚀 Đang tải Model Weights từ Google Drive...")
    try:
        gdown.download(url, tmp_zip, quiet=False)
        
        print(f"📦 Đang giải nén vào {target_dir}...")
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir) # Clear old weights if any
            
        with zipfile.ZipFile(tmp_zip, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
            
        os.remove(tmp_zip)
        print("✅ Hoàn tất! Model đã sẵn sàng để chạy.")
        
    except Exception as e:
        print(f"❌ Lỗi khi tải model: {e}")
        print("Hãy đảm bảo bạn đã bật quyền truy cập công khai cho Link Google Drive!")

if __name__ == "__main__":
    download_model()
