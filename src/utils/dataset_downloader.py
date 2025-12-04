import kagglehub
import shutil
import os

def download_flower_dataset():
    try:
        print("Baixando dataset de flores do Kaggle...")
        path = kagglehub.dataset_download("alxmamaev/flowers-recognition")
        
        # Move para estrutura do projeto
        target_path = "data/raw"
        os.makedirs("data/raw", exist_ok=True)
        
        if os.path.exists(target_path):
            shutil.rmtree(target_path)
        
        shutil.move(path, target_path)
        print(f"Dataset baixado para: {target_path}")
        return target_path + '/flowers'
        
    except Exception as e:
        print(f"Erro ao baixar dataset: {e}")
        return None