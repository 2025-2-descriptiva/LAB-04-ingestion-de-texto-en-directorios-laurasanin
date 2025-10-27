# pylint: disable=import-outside-toplevel
# pylint: disable=line-too-long
# flake8: noqa
"""
Escriba el codigo que ejecute la accion solicitada en cada pregunta.
"""

# Importaciones
from pathlib import Path
import zipfile
import os
import pandas as pd

# Funciones auxiliares
def _leer_texto(ruta: Path) -> str:
    """Lee un archivo de texto como una sola frase (sin saltos de línea)."""
    with ruta.open("r", encoding="utf-8", errors="ignore") as f:
        contenido = f.read()
    return contenido.replace("\n", " ").strip()


def _construir_df(split_dir: Path) -> pd.DataFrame:
    """
    Recorre split_dir con subcarpetas negative/positive/neutral
    y construye un DataFrame con columnas: phrase, target.
    """
    registros = []
    for target in ("negative", "positive", "neutral"):
        subdir = split_dir / target
        if not subdir.exists():
            # Si falta alguna etiqueta, simplemente continúa
            continue
        for txt_path in subdir.glob("*.txt"):
            phrase = _leer_texto(txt_path)
            registros.append({"phrase": phrase, "target": target})
    return pd.DataFrame(registros, columns=["phrase", "target"])


def pregunta_01():
    """
    La información requerida para este laboratio esta almacenada en el
    archivo "files/input.zip" ubicado en la carpeta raíz.
    Descomprima este archivo.

    Como resultado se creara la carpeta "input" en la raiz del
    repositorio, la cual contiene la siguiente estructura de archivos:


    ```
    train/
        negative/
            0000.txt
            0001.txt
            ...
        positive/
            0000.txt
            0001.txt
            ...
        neutral/
            0000.txt
            0001.txt
            ...
    test/
        negative/
            0000.txt
            0001.txt
            ...
        positive/
            0000.txt
            0001.txt
            ...
        neutral/
            0000.txt
            0001.txt
            ...
    ```

    A partir de esta informacion escriba el código que permita generar
    dos archivos llamados "train_dataset.csv" y "test_dataset.csv". Estos
    archivos deben estar ubicados en la carpeta "output" ubicada en la raiz
    del repositorio.

    Estos archivos deben tener la siguiente estructura:

    * phrase: Texto de la frase. hay una frase por cada archivo de texto.
    * sentiment: Sentimiento de la frase. Puede ser "positive", "negative"
      o "neutral". Este corresponde al nombre del directorio donde se
      encuentra ubicado el archivo.

    Cada archivo tendria una estructura similar a la siguiente:

    ```
    |    | phrase                                                                                                                                                                 | target   |
    |---:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------|
    |  0 | Cardona slowed her vehicle , turned around and returned to the intersection , where she called 911                                                                     | neutral  |
    |  1 | Market data and analytics are derived from primary and secondary research                                                                                              | neutral  |
    |  2 | Exel is headquartered in Mantyharju in Finland                                                                                                                         | neutral  |
    |  3 | Both operating profit and net sales for the three-month period increased , respectively from EUR16 .0 m and EUR139m , as compared to the corresponding quarter in 2006 | positive |
    |  4 | Tampere Science Parks is a Finnish company that owns , leases and builds office properties and it specialises in facilities for technology-oriented businesses         | neutral  |
    ```


    """
    # Ruta raíz del repositorio
    repo_root = Path(".").resolve()

    # 1) Descomprimir el ZIP de entrada
    zip_path = repo_root / "files" / "input.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {zip_path}")

    # Extrae en la raíz del repo; el zip debe crear 'input/'
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(repo_root)

    # 2) Localizar carpeta 'input' (según enunciado debe existir tras extraer)
    input_dir = repo_root / "input"
    if not input_dir.exists():
        # En caso raro de zips que traen la carpeta bajo 'files/input',
        # intenta esa ruta como fallback.
        alt_dir = repo_root / "files" / "input"
        if alt_dir.exists():
            input_dir = alt_dir
        else:
            # Como última opción, intenta detectar un directorio que contenga train/ y test/
            candidatos = [p for p in repo_root.iterdir() if p.is_dir()]
            encontrado = None
            for c in candidatos:
                if (c / "train").exists() and (c / "test").exists():
                    encontrado = c
                    break
            if encontrado is None:
                raise FileNotFoundError(
                    "No se encontró la carpeta 'input' con subcarpetas 'train' y 'test' tras descomprimir."
                )
            input_dir = encontrado

    train_dir = input_dir / "train"
    test_dir = input_dir / "test"

    # 3) Construir DataFrames
    df_train = _construir_df(train_dir)
    df_test = _construir_df(test_dir)

    # 4) Asegurar carpeta de salida y guardar CSVs sin índice
    output_dir = repo_root / "files" / "output"
    os.makedirs(output_dir, exist_ok=True)

    df_train.to_csv(output_dir / "train_dataset.csv", index=False)
    df_test.to_csv(output_dir / "test_dataset.csv", index=False)

    print(f"Archivos guardados en: {output_dir}")
    