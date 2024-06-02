import sys
import argparse

from ultralytics import YOLO

import image_capture
import cornhole_referee_inference as cornhole_inference
import cornhole_referee_model_train as cornhole_train


def main():
    """
    main Invokes the appropriate cornhole referee function or method based on the provided command line arguments

    If the -c option is used, the image_capture process will start to collect images for dataset creation.
    If the -i option is used with an argument of 0, then an inference on a live camera stream will be started. The score will not be calculated.
    If the -i option is used with an argument of 1, then the user will be prompted for an image or video file path. Once it has been provided, an inference on that file will be started. The score will not be calculated.
    If the -t option is used, a new version of the cornhole referee model will be forcefully trained.
    If no options are provided, then an inference on a live camera stream will be started. The score will be calculated.
    """
    inference = cornhole_inference.CornholeInference()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--capture",
        action="store_true",
        help="Open a camera stream to start saving images for dataset creation.",
    )
    parser.add_argument(
        "-i",
        "--inference",
        type=int,
        choices=[0, 1],
        help="Run an inference using a trained cornhole detection model. Specify 0 to run an inference on a live camera stream, and 1 to run an inference on an image of video file.",
    )
    parser.add_argument(
        "-t",
        "--train",
        action="store_true",
        help="Train a new cornhole detection model, even if one already exists.",
    )
    args = parser.parse_args()

    if args.capture:
        image_capture.capture_images()
    elif args.inference == 0:
        inference.run_inference()
    elif args.inference == 1:
        try:
            inference_file_path = str(
                input(
                    "Please enter the absolute path (as a string) to the image or video file you want to run an inference on: "
                )
            )
        except TypeError:
            sys.exit("Path needs to be in string format. Aborting.")
        inference.run_inference(inference_file_path)
    elif args.train:
        cornhole_train.train_cornhole_model()
    else:
        # Run an inference using the cornhole model and calculate the score
        inference.run_inference(calculate_score=True)


if __name__ == "__main__":
    main()
