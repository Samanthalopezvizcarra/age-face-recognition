# 🥂 Good Seed – Age Verification with Face Recognition

Proyecto para verificar la edad de personas mediante reconocimiento facial, ayudando a la cadena de supermercados **Good Seed** a garantizar que no se venda alcohol a menores de edad.

---

## 📘 Descripción del proyecto
El objetivo es construir y entrenar un **modelo de regresión de edad** basado en imágenes faciales utilizando deep learning.  
Se utiliza una arquitectura **ResNet50 preentrenada** como backbone y un modelo secuencial para predecir la edad real (`real_age`) de cada persona.

**Métrica principal:** Mean Absolute Error (MAE)

---

## 🗂 Dataset
- Ruta: `/datasets/faces`  
- Contiene:  
  - `labels.csv` – nombres de archivos y edades reales  
  - `final_files/` – imágenes faciales  

**Campos del CSV:**
- `file_name` – nombre del archivo de imagen  
- `real_age` – edad real de la persona  

---

## 🛠️ Proceso del proyecto

### 1. Análisis exploratorio de datos (EDA)
- Información general del dataset  
- Estadísticas descriptivas de edades  
- Histogramas y boxplots de distribución de edades  
- Visualización de muestra de imágenes  

### 2. Carga y preprocesamiento de datos
- Uso de `ImageDataGenerator` para:  
  - Rescale (normalización)  
  - Data augmentation: flips horizontales y verticales  
  - División de datos: 75% train / 25% validation  

### 3. Creación del modelo
- Backbone: **ResNet50** preentrenada en ImageNet  
- Capas superiores: GlobalAveragePooling2D → Dense(256, ReLU) → Dropout(0.5) → Dense(1, ReLU)  
- Optimizer: Adam (lr=0.0001)  
- Loss: MSE, Metric: MAE  

### 4. Entrenamiento
- Callbacks:
  - EarlyStopping (paciencia=3, monitor=val_loss)  
  - ModelCheckpoint (`best_model.keras`)  
- Visualización de pérdida en train vs validation  

---

## 📊 Resultados
- El modelo aprende correctamente la edad real de las personas  
- MAE (Mean Absolute Error) utilizado para evaluar desempeño  
- Data augmentation ayuda a mejorar generalización y robustez  

---

## 🧰 Tecnologías utilizadas
- Python  
- pandas · numpy · matplotlib · seaborn  
- TensorFlow / Keras  
- ResNet50 (transfer learning)  
