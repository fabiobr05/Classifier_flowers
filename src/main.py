import os
import sys
sys.path.append('src')

from preprocessing.image_processor import ImagePreprocessor
from algorithms.classifier import PlantClassifier
from algorithms.cnn_classifier import CNNClassifier
from utils.data_loader import load_flower_dataset, safe_image_load, get_image_paths
from utils.dataset_downloader import download_flower_dataset
from utils.test_data_manager import TestDataManager
from visualization.display import display_prediction, display_multiple_predictions

def train_cnn_model():
    print("Treinando classificador CNN de espécies de plantas...")
    
    # Verifica se dataset existe, senão baixa
    dataset_path = "data/raw/flowers"
    if not os.path.exists(dataset_path):
        dataset_path = download_flower_dataset()
        if dataset_path is None:
            return None
    
    # Carrega dataset
    images, labels, class_names, _ = load_flower_dataset(dataset_path)
    
    if len(images) == 0:
        print("Dataset vazio ou corrompido")
        return None
    
    print(f"Dataset carregado: {len(images)} imagens")
    
    # Experimenta diferentes arquiteturas
    results = {}
    
    # 1. CNN Simples sem augmentation
    print("\n=== Testando CNN Simples (sem augmentation) ===")
    cnn_simple = CNNClassifier()
    acc1, hist1 = cnn_simple.train(images, labels, model_type='simple', use_augmentation=False, epochs=10)
    results['CNN Simples'] = acc1
    
    # 2. CNN Simples com augmentation
    print("\n=== Testando CNN Simples (com augmentation) ===")
    cnn_aug = CNNClassifier()
    acc2, hist2 = cnn_aug.train(images, labels, model_type='simple', use_augmentation=True, epochs=10)
    results['CNN + Augmentation'] = acc2
    
    # 3. Transfer Learning
    print("\n=== Testando Transfer Learning (MobileNetV2) ===")
    cnn_transfer = CNNClassifier()
    acc3, hist3 = cnn_transfer.train(images, labels, model_type='transfer', use_augmentation=True, epochs=15)
    results['Transfer Learning'] = acc3
    
    # Mostra resultados comparativos
    print("\n=== RESULTADOS COMPARATIVOS ===")
    for model_name, accuracy in results.items():
        print(f"{model_name}: {accuracy:.3f}")
    
    # Salva melhor modelo
    best_model_name = max(results, key=results.get)
    print(f"\nMelhor modelo: {best_model_name} ({results[best_model_name]:.3f})")
    
    if best_model_name == 'Transfer Learning':
        best_classifier = cnn_transfer
    elif best_model_name == 'CNN + Augmentation':
        best_classifier = cnn_aug
    else:
        best_classifier = cnn_simple
    
    # Salva modelo e plota histórico
    best_classifier.save_model("data/results/cnn_classifier.h5")
    best_classifier.plot_training_history()
    
    return best_classifier

def train_traditional_model():
    print("Treinando classificador tradicional...")
    
    # Verifica se dataset existe, senão baixa
    dataset_path = "data/raw/flowers"
    if not os.path.exists(dataset_path):
        dataset_path = download_flower_dataset()
        if dataset_path is None:
            return None
    
    # Carrega dataset
    images, labels, class_names, _ = load_flower_dataset(dataset_path)
    
    if len(images) == 0:
        print("Dataset vazio ou corrompido")
        return None
    
    # Pré-processa e extrai características
    preprocessor = ImagePreprocessor()
    features = []
    
    for image in images:
        processed = preprocessor.preprocess(image)
        feature_vector = preprocessor.extract_features(processed)
        features.append(feature_vector)
    
    # Treina classificador
    classifier = PlantClassifier()
    accuracy, report = classifier.train(features, labels)
    
    print(f"Acurácia de treinamento: {accuracy:.3f}")
    print(report)
    
    # Salva modelo
    classifier.save_model("data/results/plant_classifier.pkl")
    return classifier, preprocessor

def predict_single_image_cnn(image_path, cnn_classifier):
    image = safe_image_load(image_path)
    if image is None:
        print(f"Não foi possível carregar imagem: {image_path}")
        return
    
    predicted_class, confidence = cnn_classifier.predict(image)
    display_prediction(image, predicted_class, confidence)
    return predicted_class, confidence

def predict_single_image_traditional(image_path, classifier, preprocessor):
    image = safe_image_load(image_path)
    if image is None:
        print(f"Não foi possível carregar imagem: {image_path}")
        return
    
    processed = preprocessor.preprocess(image)
    features = preprocessor.extract_features(processed)
    
    predicted_class, confidence = classifier.predict(features)
    display_prediction(image, predicted_class, confidence)
    return predicted_class, confidence

def main():
    print("=== SISTEMA DE CLASSIFICAÇÃO DE PLANTAS ===")
    print("Escolha o tipo de modelo:")
    print("1. CNN (Deep Learning) - Recomendado")
    print("2. Modelo Tradicional (GradientBoosting)")
    
    choice = input("Digite sua escolha (1 ou 2): ").strip()
    
    if choice == "1":
        # Modelo CNN
        cnn_model_path = "data/results/cnn_classifier.h5"
        
        if os.path.exists(cnn_model_path):
            print("Carregando modelo CNN existente...")
            cnn_classifier = CNNClassifier()
            cnn_classifier.load_model(cnn_model_path)
        else:
            cnn_classifier = train_cnn_model()
            if cnn_classifier is None:
                return
        
        # Testa com CNN
        test_with_cnn(cnn_classifier)
        
    else:
        # Modelo tradicional
        traditional_model_path = "data/results/plant_classifier.pkl"
        
        if os.path.exists(traditional_model_path):
            print("Carregando modelo tradicional existente...")
            classifier = PlantClassifier()
            classifier.load_model(traditional_model_path)
            preprocessor = ImagePreprocessor()
        else:
            result = train_traditional_model()
            if result is None:
                return
            classifier, preprocessor = result
        
        # Testa com modelo tradicional
        test_with_traditional(classifier, preprocessor)

def test_with_cnn(cnn_classifier):
    dataset_path = "data/raw/flowers"
    if os.path.exists(dataset_path):
        print("\nTestando com CNN - imagens de amostra do dataset...")
        classes = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
        
        for class_name in classes:
            class_path = os.path.join(dataset_path, class_name)
            if os.path.exists(class_path):
                files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if files:
                    test_img = os.path.join(class_path, files[0])
                    print(f"\nTestando {class_name}: {files[0]}")
                    predict_single_image_cnn(test_img, cnn_classifier)

def test_with_traditional(classifier, preprocessor):
    dataset_path = "data/raw/flowers"
    if os.path.exists(dataset_path):
        print("\nTestando com modelo tradicional - imagens de amostra do dataset...")
        classes = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
        
        for class_name in classes:
            class_path = os.path.join(dataset_path, class_name)
            if os.path.exists(class_path):
                files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if files:
                    test_img = os.path.join(class_path, files[0])
                    print(f"\nTestando {class_name}: {files[0]}")
                    predict_single_image_traditional(test_img, classifier, preprocessor)

if __name__ == "__main__":
    main()