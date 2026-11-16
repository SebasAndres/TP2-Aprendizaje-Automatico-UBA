"""Lectores de cada fuente de texto.

Cada fuente viene en un formato distinto. Cada lector limpia su formato y devuelve una lista
de documentos, donde un documento es una lista de oraciones de texto continuo:

- Oraciones sueltas (Tatoeba en es.txt.gz, rioplatense): no hay contexto entre líneas, así que todo
  el archivo es un único documento con una oración por línea.
- Charlas TED: una charla por línea, con un encabezado "Traductor: X Y Revisor: Z W".
  Cada charla es un documento.
- Subtítulos de MrBeast: un archivo por video, con cada subtítulo repetido y las oraciones
  cortadas entre líneas. Cada video es un documento.
- Subtítulos .srt (películas y series argentinas): carpetas con .srt sueltos o dentro de
  .zip. Cada archivo es un documento.

Los documentos guardan el texto con su puntuación y capitalización originales: las etiquetas
las calculan después los datasets.
"""
import gzip
import hashlib
import re
import zipfile
from collections.abc import Iterator
from pathlib import Path

Document = list[str]  # oraciones de un mismo texto continuo, en orden

# Caracteres invisibles que aparecen en los subtítulos y en algunas oraciones
_INVISIBLE = re.compile(r"[​‌‍⁠﻿]")
# Fin de oración: un signo de cierre seguido de espacio
_SENTENCE_END = re.compile(r"(?<=[.?!…])\s+")
# Encabezado de las charlas TED: siempre nombre y apellido de cada uno
_TED_HEADER = re.compile(r"^(Traductor|Transcriptor):\s+\S+\s+\S+\s+Revisor:\s+\S+\s+\S+\s+")
# Guión de diálogo al principio de un subtítulo: "-¿Dónde?"
_DIALOGUE_DASH = re.compile(r"(^|\s)-(?=\S)")
# Anotaciones que no son habla: sonidos y hablantes "[música suave]", "[Ricardo]"
_BRACKETS = re.compile(r"\[[^\]]*\]")
# Marcas de formato de los subtítulos: "<i>", "</i>", "{\an8}"
_MARKUP = re.compile(r"<[^>]*>|\{[^}]*\}")
# Líneas de control de un bloque .srt: el número y "00:00:28,780 --> 00:00:31,210"
_SRT_INDEX = re.compile(r"^\d+$")
_SRT_TIMESTAMP = re.compile(r"-->")
# Guión de diálogo al principio de una línea .srt, con o sin espacio: "- Sí." / "-Sí."
_SRT_DASH = re.compile(r"^-\s*")
# Guión de un segundo hablante en la misma línea, después de un fin de oración: "Hola. - Sí."
_SRT_INNER_DASH = re.compile(r"(?<=[.?!])\s+-\s*(?=\S)")


def clean_text(text: str) -> str:
    """Limpieza común a todas las fuentes: saca caracteres invisibles, anotaciones entre
    corchetes (sonidos y hablantes, no son habla) y marcas de formato, unifica los puntos
    suspensivos y colapsa los espacios"""
    text = _INVISIBLE.sub("", text)
    text = _MARKUP.sub(" ", _BRACKETS.sub(" ", text))
    text = text.replace("…", "...").replace("♪", " ")
    return " ".join(text.split())


def split_sentences(text: str) -> list[str]:
    """Separa un texto continuo en oraciones, cortando después de . ? ! o ..."""
    return [sentence for sentence in _SENTENCE_END.split(text) if sentence]


def _read_text(path: Path) -> str:
    """Lee un archivo de texto, comprimido o no"""
    if path.suffix == ".gz":
        with gzip.open(path, mode="rt", encoding="utf-8") as f:
            return f.read()
    return path.read_text(encoding="utf-8")


def read_line_sentences(path: str | Path) -> list[Document]:
    """Archivo con una oración por línea (.txt o .txt.gz). Las oraciones no están relacionadas
    entre sí, pero se devuelven como un solo documento para poder armar ventanas que crucen
    de una oración a la siguiente"""
    lines = (clean_text(line) for line in _read_text(Path(path)).splitlines())
    return [[line for line in lines if line]]


def read_ted(path: str | Path) -> list[Document]:
    """Archivo con una charla TED por línea. Saca el encabezado de traductor/revisor y las
    charlas repetidas"""
    documents, seen = [], set()
    for line in _read_text(Path(path)).splitlines():
        talk = clean_text(_TED_HEADER.sub("", line.strip()))
        if talk and talk not in seen:
            seen.add(talk)
            documents.append(split_sentences(talk))
    return documents


def _read_subtitle_file(path: Path) -> Document:
    """Un archivo de subtítulos: bloques separados por líneas en blanco, cada uno repetido
    una o más veces seguidas. Se quedan los bloques sin repetir, se unen en un texto
    continuo y se separa en oraciones"""
    text = _read_text(path).replace("\r", "")
    blocks = [clean_text(block) for block in re.split(r"\n\s*\n", _INVISIBLE.sub("", text))]
    unique_blocks = [
        block for i, block in enumerate(blocks)
        if block and (i == 0 or block != blocks[i - 1])
    ]
    talk = _DIALOGUE_DASH.sub(r"\1", " ".join(unique_blocks))
    return split_sentences(talk)


def read_subtitles(directory: str | Path) -> list[Document]:
    """Carpeta con un archivo de subtítulos (.txt) por video, en orden alfabético"""
    files = sorted(Path(directory).glob("*.txt"))
    return [document for document in map(_read_subtitle_file, files) if document]


def _decode(raw: bytes) -> str:
    """Los .srt vienen en UTF-8 (a veces con BOM) o en Windows-1252"""
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1252")


def _srt_files(directory: Path) -> Iterator[tuple[str, bytes]]:
    """(nombre, contenido) de cada .srt de la carpeta y sus subcarpetas, sueltos o dentro de
    un .zip, en orden alfabético"""
    for path in sorted(directory.rglob("*")):
        if path.suffix.lower() == ".srt":
            yield str(path), path.read_bytes()
        elif path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                for name in sorted(archive.namelist()):
                    if name.lower().endswith(".srt"):
                        yield f"{path}/{name}", archive.read(name)


def _srt_subtitles(text: str) -> list[str]:
    """El texto de cada subtítulo de un .srt, limpio. Saca el número y el tiempo de cada
    bloque, los guiones de diálogo y las canciones (líneas con ♪, que no tienen la
    puntuación del habla)"""
    subtitles = []
    for block in re.split(r"\n\s*\n", text.replace("\r", "")):
        lines = [
            _SRT_DASH.sub("", clean_text(line)) for line in block.split("\n")
            if line.strip() and not _SRT_INDEX.match(line.strip()) and not _SRT_TIMESTAMP.search(line)
        ]
        subtitle = _SRT_INNER_DASH.sub(" ", clean_text(" ".join(line for line in lines if "♪" not in line)))
        if subtitle:
            subtitles.append(subtitle)
    return subtitles


def _join_subtitles(subtitles: list[str]) -> str:
    """Une los subtítulos en un texto continuo. Unos puntos suspensivos al final de un
    subtítulo que sigue en minúscula (o al principio del que sigue) solo indican que la
    oración continúa en el próximo subtítulo, así que se sacan: si no, la palabra quedaría
    etiquetada con punto. Ej: "vendo mi departamento..." + "que al final" """
    joined = []
    for i, subtitle in enumerate(subtitles):
        following = subtitles[i + 1] if i + 1 < len(subtitles) else ""
        if subtitle.endswith("...") and (following[:1].islower() or following.startswith("...")):
            subtitle = subtitle[:-3].rstrip()
        if subtitle.startswith("...") and joined:
            subtitle = subtitle[3:].lstrip()
        if subtitle:
            joined.append(subtitle)
    return " ".join(joined)


def read_srt(directory: str | Path) -> list[Document]:
    """Carpeta con subtítulos .srt, sueltos o en .zip (también en subcarpetas). Cada archivo
    es un documento; los archivos con el mismo contenido se leen una sola vez"""
    documents, seen = [], set()
    for _, raw in _srt_files(Path(directory)):
        digest = hashlib.sha256(raw).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        document = split_sentences(_join_subtitles(_srt_subtitles(_decode(raw))))
        if document:
            documents.append(document)
    return documents
