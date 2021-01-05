# Confusion based meta multiple change detection framework

## Overview
This project implement the confusion based mutiple change detection method. To detect a single change point in obserbations $(X_i, t_i)$, the method will  try candidates of change points. For every dandiate change point $t_a$, the model creates labels $y_i$. $y_i = 0$ if $t_i <= t_a$ and $y_i = 1$ otherwise. Then a classifier (any classifier can be used, in the code, the default classifier is random forest) is trained to infer the created lables $y_i$. The accuracy is then sotred. After trying all the candiates, the model will fit the recorded accuracy values and report the best estimation of change point. 

Then the model will recurssively split the data into finer pieces in a BFS order and finally a change point tree is built. Every leaf node of the change tree is a range in which no further change can be found.

## Install

run git pull of the repo.

```
git clone https://github.com/yuziheusc/confusion_multi_change
```

Dependent

* numpy
* scipy
* matplotlib
* sklearn
* pickle
* graphviz
* os

## Use the code

### Generate change tree

see jupyter notebook [confusion_multi_change.ipynb](./confusion_multi_change.ipynb)

The tree generated included all the data for further analysis and visualization. Please Save the tree using json or pickle.

### Visualize the change tree

see jupyter notebook [tree_vis.ipynb](./tree_vis.ipynb)

The notebook shows an example for detect text changes in tweets in peroid Jan. 2020 - Mar. 2020. The change points are shown using calender dates. To visualize other types of change, please define your own function
```python
def _make_node_text(node_i):
    return "your own description text"
```