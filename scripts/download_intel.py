"""Download the Intel Berkeley Research Lab sensor data.

The data (``data.txt``) is fetched from the MIT mirror
http://db.csail.mit.edu/labdata/labdata.html when reachable, otherwise through
the Kaggle dataset ``divyansh22/intel-berkeley-research-lab-sensor-data``
(requires ``pip install kagglehub``).

    python scripts/download_intel.py            # writes data/data.txt
"""

import gzip
import shutil
import sys
import urllib.request
from pathlib import Path

URL = "http://db.csail.mit.edu/labdata/data.txt.gz"
OUT = Path(__file__).resolve().parents[1] / "data" / "data.txt"


def from_mit():
    gz = OUT.with_suffix(".txt.gz")
    urllib.request.urlretrieve(URL, gz)
    with gzip.open(gz, "rb") as src, open(OUT, "wb") as dst:
        shutil.copyfileobj(src, dst)
    gz.unlink()


def from_kaggle():
    import kagglehub
    folder = Path(kagglehub.dataset_download(
        "divyansh22/intel-berkeley-research-lab-sensor-data"))
    candidates = sorted(folder.rglob("*.txt")) + sorted(folder.rglob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"no data file found in {folder}")
    shutil.copyfile(candidates[0], OUT)


def main():
    OUT.parent.mkdir(exist_ok=True)
    try:
        from_mit()
    except Exception as exc:                                # noqa: BLE001
        print(f"MIT mirror unavailable ({exc}); trying Kaggle", file=sys.stderr)
        from_kaggle()
    print(f"data written to {OUT}")


if __name__ == "__main__":
    main()
