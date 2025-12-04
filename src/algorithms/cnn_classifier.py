import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt

class CNNClassifier:
    def __init__(self, num_classes=5, input_shape=(224, 224, 3)):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.classes = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
        self.model = None
        self.history = None
    
    def create_simple_cnn(self):
        # CNN simples
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        return model
    
    def create_transfer_learning_model(self):
        # Transfer learning com MobileNetV2
        base_model = MobileNetV2(input_shape=self.input_shape, 
                                include_top=False, 
                                weights='imagenet')
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.2),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        return model
    
    def create_data_augmentation(self):
        # Data augmentation
        return tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1)
        ])
    
    def prepare_data(self, images, labels, validation_split=0.2, test_split=0.1):
        # Converte para arrays numpy e normaliza
        X = np.array(images, dtype=np.float32) / 255.0
        y = tf.keras.utils.to_categorical(labels, self.num_classes)
        
        # Divide dados: treino, validação, teste
        n_samples = len(X)
        n_test = int(n_samples * test_split)
        n_val = int(n_samples * validation_split)
        
        # Embaralha dados com seed fixo para reprodutibilidade
        np.random.seed(42)
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        # Divide conjuntos
        X_test = X_shuffled[:n_test]
        y_test = y_shuffled[:n_test]
        
        X_val = X_shuffled[n_test:n_test+n_val]
        y_val = y_shuffled[n_test:n_test+n_val]
        
        X_train = X_shuffled[n_test+n_val:]
        y_train = y_shuffled[n_test+n_val:]
        
        # Salva índices do conjunto de teste para uso posterior
        self.test_indices = indices[:n_test]
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def train(self, images, labels, model_type='transfer', use_augmentation=True, epochs=20):
        print(f"Treinando modelo CNN ({model_type})...")
        
        # Prepara dados
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.prepare_data(images, labels)
        
        print(f"Treino: {len(X_train)}, Validação: {len(X_val)}, Teste: {len(X_test)}")
        
        # Cria modelo
        if model_type == 'simple':
            self.model = self.create_simple_cnn()
        else:
            self.model = self.create_transfer_learning_model()
        
        # Compila modelo
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Data augmentation
        if use_augmentation:
            augmentation = self.create_data_augmentation()
            X_train_aug = augmentation(X_train, training=True)
        else:
            X_train_aug = X_train
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.2, patience=3)
        ]
        
        # Treina modelo
        self.history = self.model.fit(
            X_train_aug, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Avalia no conjunto de teste
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        print(f"Acurácia no teste: {test_accuracy:.3f}")
        
        return test_accuracy, self.history
    
    def plot_training_history(self):
        # Plota evolução do treinamento
        if self.history is None:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Acurácia
        ax1.plot(self.history.history['accuracy'], label='Treino')
        ax1.plot(self.history.history['val_accuracy'], label='Validação')
        ax1.set_title('Acurácia do Modelo')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Acurácia')
        ax1.legend()
        
        # Loss
        ax2.plot(self.history.history['loss'], label='Treino')
        ax2.plot(self.history.history['val_loss'], label='Validação')
        ax2.set_title('Loss do Modelo')
        ax2.set_xlabel('Época')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def predict(self, image):
        # Redimensiona e normaliza imagem para predição
        import cv2
        resized = cv2.resize(image, (224, 224))
        
        if len(resized.shape) == 3:
            resized = np.expand_dims(resized, axis=0)
        
        resized = resized.astype(np.float32) / 255.0
        predictions = self.model.predict(resized, verbose=0)
        
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        
        return self.classes[predicted_class_idx], confidence
    
    def save_model(self, path):
        # Salva modelo treinado
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
    
    def load_model(self, path):
        # Carrega modelo salvo
        self.model = tf.keras.models.load_model(path)