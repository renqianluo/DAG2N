from __future__ import print_function

import os
import sys
import json

from utils import *

def main():
    dag_file = sys.argv[1]
    fig_file = sys.argv[2]
    with open(dag_file, 'r') as f:
        content = json.load(f)
        conv_dag = content['conv_dag']
        reduc_dag = content['reduc_dag']
        draw_network(conv_dag, fig_file)
        draw_network(reduc_dag, fig_file)


if __name__ == '__main__':
    main()