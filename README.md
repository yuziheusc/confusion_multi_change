# Confusion based meta multiple change detection framework

## Overview

Then the model will recurssively split the data into finer pieces in a BFS order and finally a change point tree is built. Every leaf node of the change tree is a range in which no further change can be found.

## Install

run git clone of the repo.

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

The tree generated included all the data for further analysis and visualization. Please save the tree using json or pickle.

### Visualize the change tree

see jupyter notebook [tree_vis.ipynb](./tree_vis.ipynb)

The notebook shows an example for detect text changes in tweets in peroid Jan. 2020 - Mar. 2020. The change points are shown using calender dates. To visualize other types of change, please define your own function
```python
def _make_node_text(node_i):
    return "your own description text"
```