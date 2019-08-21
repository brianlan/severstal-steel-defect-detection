import argparse
import multiprocessing
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

from src.lanutils.fs import get_indices
from src.lanutils.dl import RunLengthEncoder
from src.lanutils.plot import get_colors


parser = argparse.ArgumentParser()
parser.add_argument("--dataset-dir", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--label-path", type=Path, required=True)
parser.add_argument("--indices", type=Path)
parser.add_argument("--num-processes", type=int, default=0)
args = parser.parse_args()


def read_label(path):
    _label = defaultdict(dict)
    with open(path, "r") as f:
        is_first_line = True
        for l in f:
            if is_first_line:
                is_first_line = False
                continue
            fn_cls, encoded_seq = l.strip().split(",")
            if encoded_seq:
                fn, cls = fn_cls.split("_")
                _label[fn][int(cls)] = [int(i) for i in encoded_seq.split(" ")]
    return _label


def visualize(file_index):
    fname = f"{file_index}.jpg"
    if fname not in label.keys():
        return
    im = cv2.imread(str(args.dataset_dir / fname))
    for cls, encoded_seq in label[fname].items():
        mask = encoder.decode(encoded_seq, (1600, 256))
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cntr in contours:
            im = cv2.polylines(im, cntr.transpose(1, 0, 2), True, colors[cls - 1])
    cv2.imwrite(str(args.output_dir / fname), im)


if __name__ == "__main__":
    indices = get_indices(args.indices, args.dataset_dir, ".jpg")
    global label
    global encoder
    global colors
    label = read_label(args.label_path)
    encoder = RunLengthEncoder()
    colors = get_colors(4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.num_processes == 0:
        for fidx in tqdm(indices):
            visualize(fidx)
    with multiprocessing.Pool(processes=args.num_processes) as pool:
        _ = list(tqdm(pool.imap_unordered(visualize, indices), total=len(indices)))
