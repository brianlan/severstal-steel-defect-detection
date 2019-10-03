from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.model_selection import StratifiedKFold


def update_class_for_each_file(all_fnames, defect_stats, cls_priority):
    for fn, cls in defect_stats:
        if cls_priority[cls] > cls_priority[all_fnames[fn]]:
            # if all_fnames[fn] > 0:
            #     print(f"{fn}: {all_fnames[fn]}({cls_priority[all_fnames[fn]]:.4f}) => {cls}({cls_priority[cls]:.4f})")
            all_fnames[fn] = cls


trainval_dir = Path("/data2/datasets/kaggle/severstal-steel-defect-detection/raw/trainval")
cv_save_dir = Path("/data2/datasets/kaggle/severstal-steel-defect-detection/indices/cv")
all_fnames = {p.name: 0 for p in trainval_dir.glob("*.jpg")}

with open('/data2/datasets/kaggle/severstal-steel-defect-detection/label/defect_stats.txt', 'r') as f:
    defect_stats = []
    counter = [0] * 5
    for l in f:
        cls, fn = l.strip()[2:].split('/')
        cls = int(cls)
        counter[cls] += 1
        defect_stats.append([fn, cls])
    cls_priority = [0] + [1 / c for c in counter[1:]]
    print(f"counter: {counter}")
    print(f"cls_priority: {cls_priority}")

update_class_for_each_file(all_fnames, defect_stats, cls_priority)

counter = [0] * 5
for fn, cls in all_fnames.items():
    counter[cls] += 1
print(counter)

X, y = zip(*[(fn, cls) for fn, cls in all_fnames.items()])
X, y = np.array(X), np.array(y)
# print(X)
# print(y)
skf = StratifiedKFold(n_splits=5)
skf.get_n_splits(X, y)
for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # print("TRAIN:", train_index, "TEST:", test_index)
    # print(f"TRAIN: {np.bincount(y_train)}, TEST: {np.bincount(y_test)}")
    with open(cv_save_dir / str(i) / "train.txt", "w") as f:
        f.write('\n'.join(fn.split('.')[0] for fn in X_train))
    
    with open(cv_save_dir / str(i) / "val.txt", "w") as f:
        f.write('\n'.join(fn.split('.')[0] for fn in X_test))
