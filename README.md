# Transfer Learning com MobileNetV2 – Cats vs Dogs

Projeto de Transfer Learning aplicado à classificação de gatos e cachorros utilizando a rede **MobileNetV2** pré‑treinada no ImageNet. Desenvolvido em Python com TensorFlow/Keras no ambiente Kaggle como parte do desafio prático da [Digital Innovation One](https://www.dio.me/).

## 📁 Estrutura do Repositório

- `transfer_learning_cats_dogs.ipynb` – Notebook com todo o pipeline de treinamento.
- `images/training_curves.png` – Gráfico de acurácia e perda durante o treinamento.
- `model_summary.txt` – Resumo da arquitetura do modelo.

## 🧠 Metodologia

- **Modelo base**: MobileNetV2 (include_top=False, weights='imagenet')
- **Cabeça de classificação**: GlobalAveragePooling2D + Dropout(0.2) + Dense(1, sigmoid)
- **Congelamento**: Camadas convolucionais do modelo base mantidas não treináveis.
- **Treinamento**: 5 épocas, otimizador Adam (lr=1e-4), binary crossentropy.
- **Tratamento de dados**: Normalização (1/255) e exclusão de imagens corrompidas com `tf.data.experimental.ignore_errors()`.

## 📊 Resultados

| Época | Acurácia Treino | Acurácia Validação | Perda Treino | Perda Validação |
|-------|-----------------|---------------------|--------------|-----------------|
| 1     | 96,3%           | 100%                | 0,162        | 0,024           |
| 2     | 85,5%           | 100%                | 0,324        | 0,021           |
| 3     | 94,7%           | 99,94%              | 0,148        | 0,017           |
| 4     | 96,3%           | 99,90%              | 0,111        | 0,014           |
| 5     | **96,8%**       | **99,86%**          | **0,087**    | **0,013**       |

![Curvas de treinamento](images/training_curves.png)

### ⚠️ Observação sobre a validação

A acurácia de validação excepcionalmente alta desde a primeira época sugere um possível **vazamento de dados** entre os conjuntos de treino e validação, provavelmente devido à divisão sequencial dos arquivos sem embaralhamento. Apesar disso, o modelo demonstrou a eficácia do Transfer Learning.

## 🚀 Como reproduzir

1. Clone este repositório.
2. Instale as dependências: TensorFlow, Matplotlib.
3. Execute o notebook `transfer_learning_cats_dogs.ipynb` em um ambiente com GPU (opcional).

## 📚 Referências

- Dataset: [Microsoft Cats vs Dogs](https://www.microsoft.com/en-us/download/details.aspx?id=54765)
- MobileNetV2: [TensorFlow Applications](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2)

## 👤 Autor

Jadson Jose  
[GitHub](https://github.com/jadson-jose)
