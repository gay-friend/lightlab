from glob import glob
from pathlib import Path
import shutil

for file in glob("datasets/bushing_sleeve_od_abration/**", recursive=True):
    file = Path(file)
    if file.is_file():
        shutil.move(file, file.with_suffix(".png"))
