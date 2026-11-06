# Entrenar en Google Colab

> Guía rápida para correr el pipeline en Colab:

* Obs: Todavía no implementamos las redes igual, pero para futuro :eyes:


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

Si el repo es privado, el clone te va a pedir autenticarte: lo más fácil es usar un token de GitHub en la URL. No lo dejes escrito en un notebook que después compartas.

Usá `%cd` y no `!cd`, porque con `!` el cambio de carpeta se pierde en la línea siguiente.

## 3. Correr el pipeline

```python
!uv run python -m src.pipeline 
```

El dataset ya viene en el repo y el modelo de BERT se baja solo la primera vez. Los scores salen en el log.