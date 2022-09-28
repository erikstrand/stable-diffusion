import glob
import re
import os
import shutil
from pathlib import Path

clip_lengths = [251] * 11 + [751] * 2 + [251] * 5
n_total_files = sum(clip_lengths)
output_dir = Path("../../../outputs/test1")
os.chdir(output_dir)

def collect_files():
    files = glob.glob("*.png")
    files = sorted(files)
    files_array = [None] * (n_total_files + 1)
    number_re = re.compile(r"(\d{6})\.\d+.png")
    for file in files:
        result = number_re.match(file)
        assert(result is not None)
        number = result.group(1)
        number = int(number)
        assert(number < n_total_files + 1)
        files_array[number] = file
    return files_array

def move_clip(clip_idx, files):
    clip_dir = Path(f"clip_{clip_idx:02d}")
    clip_dir.mkdir(exist_ok=True)

    begin_idx = 1 + sum(clip_lengths[:clip_idx])
    end_idx = begin_idx + clip_lengths[clip_idx]

    for idx in range(begin_idx, end_idx):
        assert(files[idx] is not None)
        new_idx = idx - begin_idx + 1
        new_name = f"IM{new_idx:05d}.png"
        shutil.move(files[idx], clip_dir / new_name)


if __name__ == "__main__":
    files = collect_files()

    move_clip(0, files)
    move_clip(1, files)
    move_clip(2, files)
    move_clip(3, files)
    move_clip(4, files)
    move_clip(5, files)
    move_clip(6, files)
    move_clip(7, files)
    move_clip(8, files)
    move_clip(9, files)
    move_clip(10, files)
    move_clip(11, files)
    move_clip(12, files)
    move_clip(13, files)
    move_clip(14, files)
    move_clip(15, files)
    move_clip(16, files)
    move_clip(17, files)