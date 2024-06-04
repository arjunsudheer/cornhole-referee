# Cornhole Referee

Cornhole Referee is an AI and computer vision project that is meant to calculate the scores of the red team and the blue team in a cornhole game, similar to how a human referee might do so.

## Video Demonstrations of Cornhole Referee
In the video below, a blue bean bag and a red bean bag are thrown onto the board which should award each team +1 point. The score starts off with team blue having 3 points, and team red having 5 points. After the bean bags are thrown onto the board, cornhole referee correctly updates team blue's score to 4 and team red's score to 6.

https://github.com/arjunsudheer/cornhole-referee/assets/20782162/803ad65e-af7b-49ec-acb4-591746019019

In the video below, two blue bean bags and two red bean bags are each thrown. One bean bag for each team lands on the board which awards +1 point, and the other bean bag for each team lands in the hole which awards +3 points. Each team starts off with 0 points. After the bean bags are thrown, cornhole referee correctly updates team blue's score to 4 and team red's score to 4.

https://github.com/arjunsudheer/cornhole-referee/assets/20782162/9a249e81-4846-460d-b734-1f0c31d6482d

In the video below, a blue bean bag and a red bean bag are thrown. Both bean bags end up missing both the board and the hole and land on the ground which awards no points. Each team starts off with 2 points. After the bean bags are thrown, cornhole referee correctly identifies that both bean bags have missed the board and the hole and does not award any extra points. The score for team blue is left at 2 and the score for team red is also left at 2.

https://github.com/arjunsudheer/cornhole-referee/assets/20782162/782dd42b-cc61-45fb-8a05-512c78201071

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

## Usage Guide
To use cornhole referee, start the cornhole-referee.py script by running the following command in your terminal: ```python3 cornhole-referee.py```. Running this command with no specified options will start running an inference on a live camera stream with live score updates. You can access the [cornhole-referee.py documentation](docs/_build/html/cornhole_referee.html) by clicking the link.

The cornhole-referee.py script takes a variety of options which have been described in the table below:

| Option | Required Arguments | Description | Relevant Links |
| :----: | :----------------: | :---------: | :------------: |
| -c | None | Invokes the *capture_images* function in the *image_capture.py* script to collect images for dataset creation | [image_capture.py](image_capture.py) <br> [image_capture.py documentation](docs/_build/html/image_capture.html) |
| -t | None | Invokes the train_cornhole_model* function in the *cornhole_referee_model_train.py* script to train a new version of the cornhole referee model, even if one already exists | [cornhole_referee_model_train.py](cornhole_referee_model_train.py) <br> [cornhole_referee_model_train.py documentation](docs/_build/html/cornhole_referee_model_train.html) |
| -i | 0 (indicates that an inference should be run on a live camera stream) <br> <br> 1 (indicates that an inference should be run on an image or video file) | Runs an inference without live score updates on either a live camera stream or an image/video file as specified, only bounding boxes and labels will be displayed | [cornhole_referee_inference.py](cornhole_referee_inference.py) <br> [cornhole_referee_inference.py documentation](docs/_build/html/cornhole_referee_inference.html) |
