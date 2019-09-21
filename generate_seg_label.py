import argparse
import multiprocessing
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm
import cv2
import numpy as np

from src.lanutils.dl import RunLengthEncoder


parser = argparse.ArgumentParser()
parser.add_argument("--image-dir", type=Path, required=True)
parser.add_argument("--label-path", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
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


def save_npz_mask(mask, path):
    np.savez_compressed(str(path), mask=mask)


def generate_empty_mask():
    print("Generating empty masks.")
    for fname in tqdm(sorted(args.image_dir.rglob("*.jpg"))):
        empty_mask = np.zeros((4, 256, 1600), dtype=np.uint8)
        save_npz_mask((args.output_dir / fname.relative_to(args.image_dir)).with_suffix(""), empty_mask)


def generate_seg_label(fname, im_size=(1600, 256)):
    mask = np.zeros((im_size[1], im_size[0], 4), dtype=np.uint8)
    for cls, encoded_seq in label[fname].items():
        mask[:, :, cls - 1] = encoder.decode(encoded_seq, im_size)
    # cv2.imwrite(str((args.output_dir / fname).with_suffix(".png")), mask)
    save_npz_mask((args.output_dir / fname).with_suffix(""), mask.transpose(2, 0, 1))


if __name__ == "__main__":
    global label
    global encoder
    label = read_label(args.label_path)
    encoder = RunLengthEncoder()
    indices = label.keys()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generate_empty_mask()
    if args.num_processes == 0:
        for fidx in tqdm(indices):
            generate_seg_label(fidx)
    with multiprocessing.Pool(processes=args.num_processes) as pool:
        _ = list(tqdm(pool.imap_unordered(generate_seg_label, indices), total=len(indices)))
