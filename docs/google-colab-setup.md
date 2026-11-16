# Entrenar en Google Colab

> Guía rápida para correr el pipeline en Colab:


## 1. Prender la GPU

**Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU**.

## 2. Bajar el repo
Pegar esto en una celda:

```python
!git clone https://github.com/SebasAndres/TP2-Aprendizaje-Automatico-UBA.git
%cd TP2-Aprendizaje-Automatico-UBA
!pip install -q uv
!uv sync
```

Para entrenar una rama que no es `main` (si el repo ya estaba clonado, alcanza con esta celda):

```python
!git fetch origin
!git checkout <rama>
!git pull
```

## 3. Guardar los modelos en Drive

La máquina de Colab se borra cuando se cierra la sesión (y en la versión gratuita se cierra si
la pestaña queda desconectada un rato, por ejemplo si la computadora se suspende). Para no
perder lo entrenado, `models/` apunta a una carpeta de Drive. Correr antes del pipeline:

```python
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/tp2-models
!rm -rf models && ln -s /content/drive/MyDrive/tp2-models models
```

Cada modelo queda en Drive apenas termina de entrenarse; el `metrics.json`, al terminar cada
problema. Igual conviene dejar la computadora sin suspenderse (ej: con Caffeine) y la pestaña
abierta.

## 4. Correr el pipeline

```python
!uv run python -m src.pipeline
```

Con todas las fuentes chicas completas (tatoeba aporta 3000 instancias, ~22% del total),
más épocas y solo algunos experimentos:

```python
!uv run python -m src.pipeline --capitalization-size 3000 --punctuation-size 3000 --epochs 15 \
    --experiments bidirectional_joint_lstm,joint_projected_bigru,joint_residual_stacked_bilstm,joint_conv_bilstm
```

Para el notebook del póster conviene correr todos los experimentos (sin `--experiments`): lee el
`metrics.json` de la última corrida, y sin los demás no hay contra qué comparar.

El dataset ya viene en el repo y el modelo de BERT se baja solo la primera vez. Los scores salen en el log.
