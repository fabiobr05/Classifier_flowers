import os
import cv2
import numpy as np
from pathlib import Path

def load_flower_dataset(data_path):
    images = []
    labels = []
    image_paths = []
    class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
    
    # Carrega imagens de cada classe
    for class_idx, class_name in enumerate(class_names):
        class_path = os.path.join(data_path, class_name)
        if not os.path.exists(class_path):
            continue
            
        for img_file in os.listdir(class_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_path, img_file)
                image = cv2.imread(img_path)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    # Redimensiona para tamanho fixo
                    image = cv2.resize(image, (224, 224))
                    images.append(image)
                    labels.append(class_idx)
                    image_paths.append(img_path)
    
    return images, np.array(labels), class_names, image_paths

def get_image_paths(data_path):
    # Retorna apenas os caminhos das imagens
    image_paths = []
    class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
    
    for class_name in class_names:
        class_path = os.path.join(data_path, class_name)
        if not os.path.exists(class_path):
            continue
            
        for img_file in os.listdir(class_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_path, img_file)
                image_paths.append(img_path)
    
    return image_paths

def safe_image_load(image_path):
    # Carrega imagem com tratamento de erro
    if not os.path.exists(image_path):
        return None
    
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)