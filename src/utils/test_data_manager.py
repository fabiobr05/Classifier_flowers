import numpy as np
import os
import json

class TestDataManager:
    def __init__(self):
        self.test_indices_file = "data/results/test_indices.json"
    
    def save_test_indices(self, test_indices, image_paths):
        # Salva índices e caminhos do conjunto de teste
        test_data = {
            'test_indices': test_indices.tolist(),
            'test_image_paths': [image_paths[i] for i in test_indices]
        }
        
        os.makedirs(os.path.dirname(self.test_indices_file), exist_ok=True)
        with open(self.test_indices_file, 'w') as f:
            json.dump(test_data, f)
    
    def load_test_indices(self):
        # Carrega índices do conjunto de teste
        if os.path.exists(self.test_indices_file):
            with open(self.test_indices_file, 'r') as f:
                test_data = json.load(f)
            return test_data['test_indices'], test_data['test_image_paths']
        return None, None
    
    def get_test_images_by_class(self, class_name, max_samples=3):
        # Retorna imagens de teste para uma classe específica
        test_indices, test_paths = self.load_test_indices()
        if test_paths is None:
            return []
        
        # Filtra imagens da classe específica
        class_test_images = []
        for path in test_paths:
            if class_name in path and len(class_test_images) < max_samples:
                class_test_images.append(path)
        
        return class_test_images