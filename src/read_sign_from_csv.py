import csv
import pathlib
import sys

import numpy as np
#import mayavi.mlab as plt
import matplotlib.pyplot as plt

dataset_dir = "~/Public/excluded/MSLR30/"
dataset_path = pathlib.Path(dataset_dir).expanduser()
output_filename = sys.argv[1] if len(sys.argv) > 1 else '/tmp/test.svg'

assert dataset_path.exists() and dataset_path.is_dir()

testing_path = dataset_path / "Testing"

selection = []
for idx, sample in enumerate(testing_path.glob("*.csv")):
    if sample.stem.startswith("J_"):
        print(f"{idx:3} {sample.stem}")
        selection.append(sample)

current = selection[-1]

frames = []
with current.open(mode="r") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        frame = np.array(row[1:], dtype=float)
        # print(type(row), len(row), row[0], frame.shape[0] / 67)
        frames.append(frame)
        assert frame.shape[0] == 67 * 3


frame = frames[0]
print(current.stem)
print(frame, frame.shape)
indexes = np.array((5, 20, 21, 21))
b_idx, f_idx, lh_idx, rh_idx = (idx * 3 for idx in (indexes))
indexes = indexes.cumsum() * 3
print(indexes, np.roll(indexes, 1))

partitioning = np.block([[0, indexes[:-1]], [indexes]]).T
print(partitioning)
groups =[frame[start:end] for start, end in partitioning]

#groups = [frame[

body, face, left_hand, right_hand = groups

for group in groups:
    #x, y, z = group.reshape(-1, 3).T
    x, y, z = group.reshape(3, -1)
    print(x.shape)
    plt.plot(x, y, '.')
plt.gca().invert_yaxis()
plt.savefig(output_filename)
