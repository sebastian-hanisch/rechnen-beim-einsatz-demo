"""Zeit und Speicher der Klicks in der Cloud-Näherung (ein Prozess, ein BLAS-Thread, Entwicklungsrechner): Voreinstellung und Reglermaximum je Frage.

Aufruf:  python tools/rbe_kalibrierung.py        # druckt je Messung Sekunden und Spitze der numpy-Zuteilungen (tracemalloc, MB)

Zeit ist kein Befund; die Streamlit Cloud ist **nicht** gemessen. Die Spitze ist die der numpy-Zuteilungen, nicht der Prozessspeicher."""

import os
import sys
import time
import tracemalloc
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import rbe_live as LV  # noqa: E402


def messe(name, f):
    tracemalloc.start()
    t0 = time.perf_counter()
    f()
    dt = time.perf_counter() - t0
    spitze = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    print(f"{name}: {dt:.1f} s, Spitze {spitze:.0f} MB", flush=True)
    return dt, spitze


def main() -> int:
    seeds8 = LV.instanz_seeds(0, 8)
    seeds16 = LV.instanz_seeds(0, 16)
    messe("Voreinstellung (Frage 1 und 2, 8 Instanzen, Zeigernetz, eine Tour)", lambda: (LV.e1(seeds8, 0), LV.e2(seeds8, "zeiger", 0, "mehrfach", 1)))
    messe("Frage 4 (8 Instanzen, sechs Startlösungen mit Verbesserung, Zeigernetz)", lambda: LV.e4(seeds8, "zeiger", 0))
    messe("Frage 2 Reglermaximum Zeigernetz: Sampling 512, 16 Instanzen", lambda: LV.e2(seeds16, "zeiger", 0, "sampling", 512))
    m, _ = LV.plan_runs(16, 97280)
    messe(f"Frage 2 Reglermaximum Merkmalsnetz: Sampling 512, {m} Instanzen (gekürzt)", lambda: LV.e2(seeds16[:m], "merkmal", 0, "sampling", 512))
    m, _ = LV.plan_runs(16, 75807)
    messe(f"Frage 2 Reglermaximum Zeigernetz: Strahlsuche 512, {m} Instanzen (gekürzt)", lambda: LV.e2(seeds16[:m], "zeiger", 0, "strahl", 512))
    messe(f"Frage 2 Reglermaximum Merkmalsnetz: Strahlsuche 512, {m} Instanzen (gekürzt)", lambda: LV.e2(seeds16[:m], "merkmal", 0, "strahl", 512))
    messe("Frage 2 Symmetrien ×8, Zeigernetz, 16 Instanzen", lambda: LV.e2(seeds16, "zeiger", 0, "mehrfach", 20, 8))
    messe("Frage 3 Live-Probe der Neustart-Suche (4 Instanzen, Budget bis 100 000)", lambda: LV.e3_live(seeds8, "zeiger", LV.e2(seeds8, "zeiger", 0, "sampling", 8)["gap_netz"], 1520, 246384 + 8 * 35184, 8.0))
    messe("Frage 5 Voreinstellung (32 Schritte, 16 Touren, 2 Instanzen = 1 024 Touren)", lambda: LV.e5(LV.instanz_seeds(0, 2), 0, 32, 16, 1e-3))
    messe("Frage 5 Maximum (128 Schritte, 32 Touren, 2 Instanzen = 8 192 Touren)", lambda: LV.e5(LV.instanz_seeds(0, 2), 0, 128, 32, 1e-3))
    return 0


if __name__ == "__main__":
    sys.exit(main())
