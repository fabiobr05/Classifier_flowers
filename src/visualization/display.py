import matplotlib.pyplot as plt
import numpy as np

def display_prediction(image, predicted_class, confidence):
    # Exibe predição de uma única imagem
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.title(f'Predição: {predicted_class} (Confiança: {confidence:.2f})')
    plt.axis('off')
    plt.show()

def display_multiple_predictions(images, predictions, confidences, titles=None):
    # Exibe múltiplas predições em grade
    n = len(images)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i in range(n):
        axes[i].imshow(images[i])
        title = f'{predictions[i]} ({confidences[i]:.2f})'
        if titles:
            title = f'{titles[i]}: {title}'
        axes[i].set_title(title)
        axes[i].axis('off')
    
    # Oculta eixos não utilizados
    for i in range(n, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()