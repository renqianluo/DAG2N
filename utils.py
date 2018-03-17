from __future__ import print_function

from collections import defaultdict
from datetime import datetime
import os
import json
import logging

import numpy as np
import pygraphviz as pgv

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 


try:
    import scipy.misc
    imread = scipy.misc.imread
    imresize = scipy.misc.imresize
    imsave = imwrite = scipy.misc.imsave
except:
    import cv2
    imread = cv2.imread
    imresize = cv2.imresize
    imsave = imwrite = cv2.imwrite


##########################
# Network visualization
##########################

def add_node(graph, node_id, label, shape='box', style='filled'):
    if label.startswith('h'):
        color = 'white'
    elif label.startswith('sep'):
        color = 'yellow'
    elif label.startswith('avg'):
        color = 'yellow'
    elif label.startswith('identity'):
        color = 'yellow'
    elif label.startswith('max'):
        color = 'yello'
    elif label.startswith('add'):
        color = 'seagreen3'
    elif label.startswith('concat'):
        color = 'pink'
    else:
        color = 'white'

    if not any(label.startswith(word) for word in  ['h']):
        label = f"{label}\n({node_id})"

    graph.add_node(
            node_id, label=label, color='black', fillcolor=color,
            shape=shape, style=style,
    )

def draw_network(dag, path):
    makedirs(os.path.dirname(path))
    graph = pgv.AGraph(directed=True, strict=True,
                       fontname='Helvetica', arrowtype='open') # not work?

    num_nodes = len(dag)
    leaf_nodes = ['node_%d' % i for i in range(1, num_nodes+1)]
    for i in range(1, num_nodes+1):
        name = 'node_%d' % i
        node = dag[name]
        assert name == node[0], 'name incompatible with node.name'
        if i == 1:
            add_node(graph, node[0], 'h[i-1]')
        elif i == 2:
            add_node(graph, node[0], 'h[i]')
        else:
            add_node(graph, node[0]+'-1', node[3])
            add_node(graph, node[0]+'-2', node[4])
            add_node(graph, node[0], 'add')
            graph.add_edge(node[1], node[0]+'-1')
            graph.add_edge(node[2], node[0]+'-2')
            graph.add_edge(node[0]+'-1', node[0])
            graph.add_edge(node[0]+'-2', node[0])
            if node[1] in leaf_nodes:
                leaf_nodes.remove(node[1])
            if node[2] in leaf_nodes:
                leaf_nodes.remove(node[2])

    add_node(graph, 'cell_out', 'concat')
    for node_id in leaf_nodes:
        graph.add_edge(node_id, 'cell_out')

    graph.layout(prog='dot')
    graph.draw(path)

def make_gif(paths, gif_path, max_frame=50, prefix=""):
    import imageio

    paths.sort()

    skip_frame = len(paths) // max_frame
    paths = paths[::skip_frame]

    images = [imageio.imread(path) for path in paths]
    max_h, max_w, max_c = np.max(
            np.array([image.shape for image in images]), 0)

    for idx, image in enumerate(images):
        h, w, c = image.shape
        blank = np.ones([max_h, max_w, max_c], dtype=np.uint8) * 255

        pivot_h, pivot_w = (max_h-h)//2, (max_w-w)//2
        blank[pivot_h:pivot_h+h,pivot_w:pivot_w+w,:c] = image

        images[idx] = blank

    try:
        images = [Image.fromarray(image) for image in images]
        draws = [ImageDraw.Draw(image) for image in images]
        font = ImageFont.truetype("assets/arial.ttf", 30)

        steps = [int(os.path.basename(path).rsplit('.', 1)[0].split('-')[1]) for path in paths]
        for step, draw in zip(steps, draws):
            draw.text((max_h//20, max_h//20),
                      f"{prefix}step: {format(step, ',d')}", (0, 0, 0), font=font)
    except IndexError:
        pass

    imageio.mimsave(gif_path, [np.array(img) for img in images], duration=0.5)

def get_time():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def makedirs(path):
    if not os.path.exists(path):
        print("[*] Make directories : {}".format(path))
        os.makedirs(path)
