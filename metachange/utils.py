import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from graphviz import Digraph

def plot_curves(res, fnou=None, ylim=None):

    n_res = len(res)
    lbd_array = np.array([res[i]["lbd"] for i in range(n_res)])
    accdev_array = np.array([res[i]["accdev"] for i in range(n_res)])
    curvex_array = np.array([res[i]["model_curve"][0] for i in range(n_res)])
    curvey_array = np.array([res[i]["model_curve"][1] for i in range(n_res)])
    mean_lbd = np.mean(lbd_array, axis=0)
    mean_accdev = np.mean(accdev_array, axis=0)
    err_accdev = np.std(accdev_array, axis=0)
    mean_curvex = np.mean(curvex_array, axis=0)
    mean_curvey = np.mean(curvey_array, axis=0)
    err_curvey = np.std(curvey_array, axis=0)

    plt.scatter(mean_lbd, mean_accdev, label="Observed")
    plt.errorbar(mean_lbd, mean_accdev, yerr=err_accdev, ls='none')
    plt.plot(mean_curvex, mean_curvey, '--', c="k", label="Fitting")
    plt.fill_between(mean_curvex, mean_curvey - err_curvey, mean_curvey + err_curvey, alpha=0.4, color="gray")

    if ylim is not None:
        assert isinstance(ylim, list) or isinstance(ylim, tuple)
        plt.ylim(ylim)

    plt.xlabel("$t_a$", fontsize=18)
    plt.ylabel("Accuracy Deviation", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=14)

    if fnou is not None:
        plt.savefig(fnou, dpi=400, bbox_inches='tight')

    plt.show()


def _rec_show_tree(node_i, node_idx, dot, dot_info, make_node_text):
    ## recurssive function used to show the tree

    node_label = "%d"%(node_idx)
    node_text = node_label

    #if(node_i.left != None): feature = node_i.split_feature
    #else: feature = -1
    
    # if(node_i["left"] is None):
    #     print(dot_info[0],node_i["data"]["t_left"], node_i["data"]["t_right"])
    
    dot.node(node_label, make_node_text(node_i["data"]), style="filled", fillcolor="#B8F0B2")
    
    if(node_i["left"] != None):
        dot_info[0]+=1
        left_idx = dot_info[0]
        left_label = "%d"%(left_idx)
        left_text = left_label
        dot.node(left_label, left_text)
        dot.edge(node_label, left_label)
        _rec_show_tree(node_i["left"], left_idx, dot, dot_info, make_node_text)
    if(node_i["right"] != None):
        dot_info[0] += 1
        right_idx = dot_info[0]
        right_label = "%d"%(right_idx)
        right_text = right_label
        dot.node(right_label, right_text)
        dot.edge(node_label, right_label)
        _rec_show_tree(node_i["right"], right_idx, dot, dot_info, make_node_text)


def show_tree(root, make_node_text, fname="./tree.pdf"):
    '''
        root: root of tree
        fname: output file
        make_node_text: callable. Takes the node data dictionary as input.
        Contains the following fields: t_left, t_right (start and end of t)
        For non-leaf node, also contains t0, alpha, ratio(portion in whole data),
        res (change point infomation, returned by meta_change_detect_np)
    '''
    fname_split = os.path.splitext(fname)
    fname_pref, fname_suf = fname_split[0], fname_split[1][1:]
    dot = Digraph(node_attr={'shape': 'box'}, format=fname_suf)
    dot.node("0", "root")
    _rec_show_tree(root, 0, dot, [0], make_node_text)
    dot.render(fname_pref, view=False)
    return dot