import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import os

class PlantClassifier:
    def __init__(self):
        self.model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
        self.scaler = StandardScaler()
        self.classes = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
    
    def train(self, features, labels):
        # Divide dados em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )
        
        # Normaliza características
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Treina o modelo
        self.model.fit(X_train_scaled, y_train)
        
        # Avalia performance
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy, classification_report(y_test, y_pred, target_names=self.classes)
    
    def predict(self, features):
        # Normaliza características e calcula probabilidades
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        probabilities = self.model.predict_proba(features_scaled)[0]
        predicted_class = self.classes[np.argmax(probabilities)]
        confidence = np.max(probabilities)
        
        return predicted_class, confidence
    
    def save_model(self, path):
        # Salva modelo e scaler treinados
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model_data = {'model': self.model, 'scaler': self.scaler}
        joblib.dump(model_data, path)
    
    def load_model(self, path):
        # Carrega modelo e scaler salvos
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']