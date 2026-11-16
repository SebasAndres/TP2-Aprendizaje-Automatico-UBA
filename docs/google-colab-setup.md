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

## 3. Correr el pipeline

```python
!uv run python -m src.pipeline
```

Con todas las fuentes chicas completas (tatoeba aporta 3000 instancias, ~22% del total),
más épocas y solo algunos experimentos:

```python
!uv run python -m src.pipeline --capitalization-size 3000 --punctuation-size 3000 --epochs 15 \
    --experiments bidirectional_joint_lstm,joint_projected_bigru,joint_residual_stacked_bilstm,joint_conv_bilstm
```

Dejar la pestaña abierta: Colab corta la sesión gratuita si queda inactiva un rato. Los modelos y
el `metrics.json` quedan en `models/<día>__<hora>/` de la máquina de Colab: bajarlos (o copiarlos
a Drive) antes de que se cierre la sesión.

El dataset ya viene en el repo y el modelo de BERT se baja solo la primera vez. Los scores salen en el log.
