# Cornhole Referee

Cornhole Referee is an AI and computer vision project that is meant to calculate the scores of the red team and the blue team in a cornhole game, similar to how a human referee might do so.

## Tools and Technologies Used
* Python Programming
* YOLOv8 (from ultralytics)
* Supervision library (from Roboflow)
* OpenCV
* Visual Studio Code
* pip3
* Sphinx documentation generator (used with napolean package to generate docs from Google formatted docstrings)
* Roboflow Image Annotator

## How the Dataset Was Created
The cornhole images dataset was created by taking many images of the cornhole board and bean bags in different positions. The camera was placed above the cornhole board and in front of it. The following list includes different kinds of images that were taken in each camera position:

* Null image (only the background, no subjects of interest present)
* Bean bags only (only blue, only red, and mixed)
* Board only
* Bean bags in the hole
* Bean bags on the board
* Bean bags that are in frame but have missed the board
* Bean bags that are traveling in the air before hitting the board or ground

A total of 394 images were collected for the cornhole images dataset, and all images were annotated using Roboflow. The *image_capture.py* script was used to collect images to ensure that the collected images were readable by Roboflow. 

After annotating all the images, image augmentations were applied to generate new training examples based on the original images that were collected. The following image augmentations were applied:

| Image Augmentation | Benefit |
| :----------------: | :-----: |
| Flip (horizontal and vertical) | Useful for identifying objects in different orientations |
| 90º rotate (clockwise, counter-clockwise, upside down) | Useful for identifying objects at an angle |
| Crop (0% minimum zoom, 25% maximum zoom) | Useful for identifying objects that may not be fully in view |
| Rotation (between -15º and 15º) | Useful for identifying objects at an angle |
| Shear (±10º horizontal, ±10º vertical) | Useful for identifying objects at an angle |
| Saturation (between -25% and 25%) | Useful for identifying objects in different lighting conditions |
| Brightness (between -15% and 15%) | Useful for identifying objects in different lighting conditions |
| Exposure (between -10% and 10%) | Useful for identifying objects in different lighting conditions |

## Installation Guide
Afer this repository is cloned to your local machine, you will need to install a few libraries and software packages to successfully run this cornhole referee program.

The source code requires two dependencies: *ultralytics*, and *supervision*.

To install ultralytics, you can run run this command in your terminal:
```pip3 install ultralytics```. The ultralytics library contains YOLOv8 which will be used training the model and running an inference on it.

To install supervision, you can run this command in your terminal:
```pip3 install supervision```. The supervision library contains useful methods for displaying annotations to the screen, and is used for score calculations.

To generate your own documentation based on the specified docstrings in your source files, you can use the Sphinx library. The sphinx-rtd-theme was also used to generate the current version of the cornhole referee documentation. To install Sphinx and the sphinx-rtd-theme, you can run the following commands in your terminal:
```
pip install Sphinx
pip install sphinx-rtd-theme
```

## Usage Guide
To use cornhole referee, start the cornhole-referee.py script by running the following command in your terminal: ```python3 cornhole-referee.py```. Running this command with no specified options will start running an inference on a live camera stream with live score updates.

The cornhole-referee.py script takes a variety of options which have been described in the table below:

| Option | Required Arguments | Description | Relevant Links |
| :----: | :----------------: | :---------: | :------------: |
| -c | None | Invokes the *capture_images* function in the *image_capture.py* script to collect images for dataset creation | [image_capture.py](image_capture.py) <br> [image_capture.py documentation](docs/_build/html/image_capture.html) |
| -t | None | Invokes the train_cornhole_model* function in the *cornhole_referee_model_train.py* script to train a new version of the cornhole referee model, even if one already exists | [cornhole_referee_model_train.py](cornhole_referee_model_train.py) <br> [cornhole_referee_model_train.py documentation](docs/_build/html/cornhole_referee_model_train.html) |
| -i | 0 (indicates that an inference should be run on a live camera stream) <br> <br> 1 (indicates that an inference should be run on an image or video file) | Runs an inference without live score updates on either a live camera stream or an image/video file as specified, only bounding boxes and labels will be displayed | [cornhole_referee_inference.py](cornhole_referee_inference.py) <br> [cornhole_referee_inference.py documentation](docs/_build/html/cornhole_referee_inference.html) |
