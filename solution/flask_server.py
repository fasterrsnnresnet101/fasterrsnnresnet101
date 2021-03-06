import numpy as np
import six.moves.urllib as urllib
import tarfile
import tensorflow as tf
import zipfile
import sys
import os
import socket

from PIL import Image
from pascal_voc_writer import Writer
from flask import Flask, request, send_file, make_response

# to access the object-detection library
sys.path.insert(0, '/data/team01/solution/models/research')
sys.path.append("/data/team01//solution/models/research/slim")

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4" # specify which GPU(s) to be used

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# path to frosen graph (model)
PATH_TO_FROZEN_GRAPH = '/data/team01/solution/exported_graphsALL/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('/data/team01/solution/data', 'object-detection.pbtxt')
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# Actual detection.
score_threshold = 0.5

# Load a (frozen) Tensorflow model into memory
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
    
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict


def createXML(image_path):
  image = Image.open(image_path)
  fileName = os.path.basename(image_path)
  picWidth, picHeight = image.size
  writer = Writer(os.getcwd() + '/saved/' + fileName, picWidth, picHeight)

  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = load_image_into_numpy_array(image)
  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
  image_np_expanded = np.expand_dims(image_np, axis=0)
  # Actual detection.
  output_dict = run_inference_for_single_image(image_np, detection_graph)
  
  for index, score in enumerate(output_dict['detection_scores']):
    if (score >= score_threshold):
      writer.addObject(category_index[output_dict['detection_classes'][index]]['name'], 
                       int(output_dict['detection_boxes'][index][1]*picWidth), 
                       int(output_dict['detection_boxes'][index][0]*picHeight),  
                       int(output_dict['detection_boxes'][index][3]*picWidth), 
                       int(output_dict['detection_boxes'][index][2]*picHeight))
    else:
      break
  newFileName = fileName.split('.')[0] + '.xml'
  writer.save(os.getcwd() + '/xml/' + newFileName)
  return newFileName

app = Flask(__name__)

@app.route('/')
def main():
    return '/'

@app.route('/image', methods=['POST', 'GET'])
def image():
    if request.method == 'POST':
        try:
            imageFile = (request.files['file'])
            image_path = os.getcwd() + '/saved/' + imageFile.filename
            imageFile.save(image_path)
            newFileName = createXML(image_path)
            response = make_response(send_file(os.getcwd() + '/xml/' + newFileName, mimetype='text/xml'))
            response.headers['fileName'] = newFileName
            print("respons get off")
            return response
        except Exception as e:
            print(e)
            return 'Error'
    else:
        return '/image'
myhost = socket.gethostbyname(socket.gethostname())
app.run(host=myhost,port=1999)