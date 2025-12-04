# Plant Species Recognition

Sistema de reconhecimento de espécies de plantas usando visão computacional.

## Dataset
Download do dataset Kaggle: https://www.kaggle.com/datasets/alxmamaev/flowers-recognition

Extrair para: `data/raw/flowers/`

## Estrutura
```
visao_computacional/
├── data/
│   ├── raw/flowers/          # Dataset Kaggle
│   ├── processed/            # Imagens processadas
│   └── results/              # Modelos e resultados
├── src/
│   ├── preprocessing/        # Pré-processamento
│   ├── algorithms/           # Algoritmos de classificação
│   ├── utils/               # Utilitários
│   └── visualization/       # Visualização
└── main.py                  # Script principal
```

## Uso
```bash
pip install -r requirements.txt
python src/main.py
```

## Classes Suportadas
- Margarida (daisy)
- Dente-de-leão (dandelion) 
- Rosa (rose)
- Girassol (sunflower)
- Tulipa (tulip)