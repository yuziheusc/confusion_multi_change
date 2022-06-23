# Confusion based meta multiple change detection

## Overview

This project implement the confusion based mutiple change detection method. To detect a single change point in obserbations (X<sub>i</sub>, t<sub>i</sub>), the method will try candidates of change points. For every dandiate change point t<sub>a</sub>, the model creates labels y<sub>i</sub>. y<sub>i</sub> = 0 if t<sub>i</sub> <= t<sub>a</sub> and y<sub>i</sub> = 1 otherwise. Then a classifier (any classifier can be used, in the code, the default classifier is random forest) is trained to infer the created lables y<sub>i</sub>. The accuracy is then sotred. After trying all the candiates, the model will fit the recorded accuracy values and report the best estimation of change point. 

Then the model will recurssively split the data into finer pieces in a BFS order and finally a change point tree is built. Every leaf node of the change tree is a range in which no further change can be found.

## Install

run git clone of the repo.

```
git clone https://github.com/yuziheusc/confusion_multi_change
```

## Dependents

* numpy
* scipy
* matplotlib
* sklearn
* pickle
* graphviz
* os
* tensorflow

## Use the code

<!-- dd -->

### Whats inside
- `./example.ipynb` is the example notebook which shows basic use of the code
- `./metachange/` contains all the source code which can be imported
- `./data/` contains data and code to generate synthetic data

### Run the code
First the package should be imported 
```phthon
import import metachange
```
Also import some necessary packages
```
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
```

#### For small dataset
In case the data can be fit into memory, put data into two numpy array `X` and `t`.<br>
As a simple example, first generate data<br>
```python
X = np.array([[0,1]]*1000 + [[1,0]]*1000)
t = np.arange(2000)*1./2000
```
Then, run meta change detection with random forest classifier
```python
clf_rf = RandomForestClassifier(max_depth=32, criterion="entropy", random_state=0)
res_rf = metachange.meta_change_detect_np(X, t, clf_rf)
```
The accuracy deviation curve is plotted using
```python
metachange.plot_curves(res_rf)
```

#### For large dataset
For large image dataset which can only be used in mini-batches, the process is more involved.<br>
The code only support image datasets. The dataset should be stored in the following way:
```
data_root
|
|----data_root/dataset_train/
|        meta_data.bin
|        0000001234.png
|----data_root/dataset_test/
         meta_data.bin
         0000004321.png
```
The image files should be stored using digit file names.<br>
The meta_data contains t for each image.

The code only support CNN written by tensorflow. The model is hard coded. To customize model, please edit `./metachagne/model_tf.py`.

To run the code, first, do confusion based training and save the accuracy.
```python
path = "./data_root"
metachange.train_random_split_tf(path, n_batch=16, n_epoch=20)
```
This will create a folder `data_root/res_folder` 
Then, get change from saved accuracy
```pyhton
res_image = metachange.meta_change_from_file(path)
```

### For multiple changes
Multiple changes can be detected using recursive binary split. Currently, the multiple change feature is only available for small datasets.<br>
Generate a simple dataset
```python
X = np.array([[0,1]]*500 + [[1,0]]*500 + [[2,0]]*500 + [[2,1]]*500)
t = np.arange(2000)*1./2000
```
Detect multiple changes, using random forest classifier
```python
clf_rf = RandomForestClassifier(max_depth=32, criterion="entropy", random_state=0)
res_multi = metachange.change_point_tree(X, t, clf_rf, min_range=0.20)
```
Visualize the change tree
```python
## define a funciton which generates node text
def make_node_text(data):
    t_left = data["t_left"]
    t_right = data["t_right"]
    
    if "t0" in data:
        header = f't_0 = {data["t0"]:.4f}\n alpha = {data["alpha"]:.4f}'
    else:
        header = "Leaf"
    return f"{header}\nRange:{t_left:.4f}-{t_right:.4f}"
    
metachange.show_tree(res_multi, make_node_text)
```

Please see `/example.ipynb` for a tutorial.