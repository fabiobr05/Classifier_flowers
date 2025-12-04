import cv2
import numpy as np

class ImagePreprocessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
    
    def preprocess(self, image):
        # Redimensiona e normaliza imagem
        resized = cv2.resize(image, self.target_size)
        normalized = resized.astype(np.float32) / 255.0
        return normalized
    
    def extract_features(self, image):
        # Converte para uint8 para operações OpenCV
        img_uint8 = (image * 255).astype(np.uint8)
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        
        # Histograma de cores
        hist_r = cv2.calcHist([img_uint8], [0], None, [32], [0, 256])
        hist_g = cv2.calcHist([img_uint8], [1], None, [32], [0, 256])
        hist_b = cv2.calcHist([img_uint8], [2], None, [32], [0, 256])
        
        # Características de textura
        texture = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Combina todas as características
        features = np.concatenate([hist_r.flatten(), hist_g.flatten(), hist_b.flatten(), [texture]])
        return features