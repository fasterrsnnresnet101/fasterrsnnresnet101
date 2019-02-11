# fasterrsnnresnet101

## [labelImg-master](../../tree/master/labelImg-master)

[Original labelImg repository](https://github.com/tzutalin/labelImg)

Additions:
+ **"Image analysis" button** - marking of the selected image;
+ **button "Analysis of all images"** - marking of all images in the selected folder (using multithreading for requests).

Installation instructions you can found in the original repository.


## [flask_server.py](../../tree/master/solution/flask_server.py)
Server that processes requests received from labelImg.
To start server type in terminal:
```
python flask_server.py
```

## [parser.py](../../tree/master/solution/parser.py)
Sample script for marking up images and saving the results in .xml files.
Before running the script, you need to select the **`direct`** folder, where the images will be taken from.
```python
direct = "/data/team01/images/train"
```
You can also change the variable **`score_threshold`**, but it is recommended to take it around 0.5.
```python
score_threshold = 0.5
```
All .xml files will be saved in the xml folder.

## [savelabeledimg.ipynb](../../tree/master/solution/savelabeledimg.ipynb)
Sample script for marking up images and saving the results in .jpg files
Before running the script, you must select the **`direct`** folder, from where the images will be taken, and the **`savepath`** folder, where the images will be saved.
```python
direct = "/data/team01/images/train"
savepath = "/data/team01/solution/reusltsImg/"
```


## Constants
In all files:
Needed to access the object-detection library:
```python
sys.path.insert(0, '/data/team01/solution/models/research')
sys.path.append("/data/team01//solution/models/research/slim")
```
Specify which GPU(s) to be used:
```python
os.environ["CUDA_VISIBLE_DEVICES"]="3"
```
Path to model (frosen_graph.pb):
```python
PATH_TO_FROZEN_GRAPH = '/data/team01/solution/exported_graphsALL/frozen_inference_graph.pb'
```
List of the strings that is used to add correct label for each box:
```python
PATH_TO_LABELS = os.path.join('/data/team01/solution/data', 'object-detection.pbtxt')
```

## [Model](../..//tree/master/solution)
Github allows users to save files up to 100mb, so after downloading the repository, you need to unpack the following archives.
```
solution/data/train.rar
solution/training/model.ckpt-27788.rar
solution/exported_graphsALL/frozen_inference_graph.rar
solution/exported_graphsALL/model.ckpt.rar
solution/exported_graphsALL/saved_model/saved_model.rar
solution/faster_rcnn_resnet101_kitti_2018_01_28/frozen_inference_graph.rar
solution/faster_rcnn_resnet101_kitti_2018_01_28/model.ckpt.rar
solution/faster_rcnn_resnet101_kitti_2018_01_28/saved_model/saved_model.rar
```
