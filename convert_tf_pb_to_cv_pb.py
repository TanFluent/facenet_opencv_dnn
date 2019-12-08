# reference1: https://github.com/opencv/opencv/issues/14073
# reference2: https://github.com/opencv/opencv/issues/14224
# reference3: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/optimize_for_inference_lib.py

import tensorflow as tf
from tensorflow.tools.graph_transforms import TransformGraph

import pdb


INPUT_GRAPH_DEF_FILE = './models/graph-opt.pb'
OUTPUT_GRAPH_DEF_DIR = './models/'

# -------------------------------------------------------------------------------------------

# [1] get graph from pre-train faceNet .pb models
graph_def = tf.GraphDef()
with tf.gfile.FastGFile(INPUT_GRAPH_DEF_FILE, 'rb') as f:
    graph_def.ParseFromString(f.read())

with tf.Session() as sess:
    sess.graph.as_graph_def()
    tf.import_graph_def(graph_def, name='')

    # [2] remove 'Switch' and 'Merge' node in graph
    for i in reversed(range(len(graph_def.node))):
        op = graph_def.node[i].op
        name = graph_def.node[i].name

        if op == 'Switch' or op == 'Merge':
            # get input of 'Switch'/'Merge' node
            inp = graph_def.node[i].input[0]

            # find nodes connected to 'Switch'/'Merge' node in graph,
            # cut their connection, and redirect to the input of 'Switch'/'Merge';
            for node in graph_def.node:
                for j in range(len(node.input)):
                    if name == node.input[j]:
                        node.input[j] = inp
            del graph_def.node[i]

    node_names = []
    node_op = []
    for i in reversed(range(len(graph_def.node))):
        op = graph_def.node[i].op
        name = graph_def.node[i].name

        node_names.append(name)
        node_op.append(op)

    # [3] transform graph for inference
    graph_def = TransformGraph(graph_def,
                               ['input'],
                               ['embeddings'],
                               [
                                   'strip_unused_nodes(type=float, shape="1,160,160,3")',
                                   'remove_nodes(op=Identity, op=CheckNumerics, op=PlaceholderWithDefault)',
                                   'fold_constants(ignore_errors=true)',
                                   'sort_by_execution_order',
                                   'fold_batch_norms',
                                   'fold_old_batch_norms',
                                   'remove_device'
                               ]
                               )

    tf.train.write_graph(graph_def, "", OUTPUT_GRAPH_DEF_DIR + 'graph_final.pb', as_text=False)

    # [4] (option) get .pbtxt
    for i in reversed(range(len(graph_def.node))):
        if graph_def.node[i].op == 'Const':
            del graph_def.node[i]
        for attr in ['T', 'data_format', 'Tshape', 'N', 'Tidx', 'Tdim',
                     'use_cudnn_on_gpu', 'Index', 'Tperm', 'is_training',
                     'Tpaddings']:
            if attr in graph_def.node[i].attr:
                del graph_def.node[i].attr[attr]

    tf.train.write_graph(graph_def, "", OUTPUT_GRAPH_DEF_DIR + 'graph_final.pbtxt', as_text=True)

