# The following code is taken from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/label_image/label_image.py 
# and is modified to the needs of the project

import numpy as np
import tensorflow as tf

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def read_tensor_from_image_file(image, input_height=299, input_width=299,
				input_mean=0, input_std=255):
  float_caster = tf.cast(image, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0);
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

def prediction(image, graph, labels, input_operation, output_operation):
  input_height = 224
  input_width = 224
  input_mean = 128
  input_std = 128
  
  t = read_tensor_from_image_file(image,
                                  input_height=input_height,
                                  input_width=input_width,
                                  input_mean=input_mean,
                                  input_std=input_std)

  with tf.Session(graph=graph) as sess:
    results = sess.run(output_operation.outputs[0],
                      {input_operation.outputs[0]: t})
  results = np.squeeze(results)

  top_k = results.argsort()[-5:][::-1]
  
  preds = []
  for i in top_k:
    preds.append((labels[i], results[i]))

  return preds
