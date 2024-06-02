import sys
import cv2
import numpy as np
import torch

from ultralytics import YOLO
import supervision as sv

import cornhole_referee_model_train as cornhole_train


class CornholeInference:

    def __init__(self):
        """
        __init__ Initializes class id values and the board_coordinates Tensor.

        Uses class id constants from the Roboflow annotations.
        """
        # Keep track of each subject's class id
        self.BLUE_BEAN_BAG_CLASS_ID = 0
        self.BLUE_BEAN_BAG_IN_HOLE_CLASS_ID = 1
        self.BOARD_CLASS_ID = 2
        self.HOLE_CLASS_ID = 3
        self.RED_BEAN_BAG_CLASS_ID = 4
        self.RED_BEAN_BAG_IN_HOLE_CLASS_ID = 5

        # Initialize the board coordinates in case the board is not detected in the first frame
        self.board_coordinates = torch.Tensor([0, 0, 0, 0])

    def calculate_score(self):
        """
        calculate_score Calculates the points scored for each team.

        Adds one point for each bean bag on the board, and three points for each bean bag in the hole. Uses a PoylgonZone to identify the number of blue and red bean bags on the board.
        """
        # Get the coordinates of the board to use for points scoring
        i = 0
        while (
            i < len(self.results.boxes)
            and self.results.boxes[i].cls.item() != self.BOARD_CLASS_ID
        ):
            i += 1
        # Get the coordinates of the board from the inference detection results
        # If a board detection was not recorded in the current frame, then use the existing board coordinates
        self.board_coordinates = (
            self.board_coordinates
            if i >= len(self.results.boxes)
            else self.results.boxes[i].xyxy[0]
        )
        # Define a new polygon zone based on the board's coordinates
        # The bean bag's center position must be within the board polygon zone for a point to be awarded
        board_zone = sv.PolygonZone(
            np.array(
                [
                    [int(self.board_coordinates[0]), int(self.board_coordinates[1])],
                    [int(self.board_coordinates[0]), int(self.board_coordinates[3])],
                    [int(self.board_coordinates[2]), int(self.board_coordinates[3])],
                    [int(self.board_coordinates[2]), int(self.board_coordinates[1])],
                ],
            ),
            triggering_anchors=[sv.Position.CENTER],
        )

        # Recalculate the points and count totals for the current frame
        board_zone.trigger(self.blue_bean_bag_detections)
        self.blue_bean_bags_on_board = self.blue_points = board_zone.current_count

        self.blue_points += len(self.blue_bean_bags_in_hole_detections) * 3

        board_zone.trigger(self.red_bean_bag_detections)
        self.red_bean_bags_on_board = self.red_points = board_zone.current_count

        self.red_points += len(self.red_bean_bags_in_hole_detections) * 3

    def display_information(self):
        """
        display_information Displays the score information on the screen.

        Shows the point total, the number of bean bags on the board, and the number of bean bags in the hole for each color. Adds a white rectangle at the top of the screen to make the score text more visible.
        """
        score_information = {
            # Display the number of points that the blue team has in the top left corner of the image
            f"Blue Points: {self.blue_points} ({len(self.blue_bean_bags_in_hole_detections)} in hole) ({self.blue_bean_bags_on_board} on board)": [
                (90, 50),
                (255, 0, 0),
            ],
            # Display the number of points that the red team has in the top right corner of the image
            f"Red Points: {self.red_points} ({len(self.red_bean_bags_in_hole_detections)} in hole) ({self.red_bean_bags_on_board} on board)": [
                (1185, 50),
                (0, 0, 255),
            ],
        }
        # Add a white rectangle at the top of the frame to make the text more visible
        self.annotated_frame = cv2.rectangle(
            self.annotated_frame, (0, 0), (1920, 75), (255, 255, 255), -1
        )
        # Loop through the score_information dictionary and display the information
        for key in score_information:
            self.annotated_frame = cv2.putText(
                self.annotated_frame,
                key,
                score_information[key][0],
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                score_information[key][1],
                2,
            )

    def run_inference(
        self, file_path: str = None, camera_port: int = 0, calculate_score: bool = False
    ):
        """
        run_inference Finds all the detections of the board, hole, blue-bean-bag, red-bean-bag, blue-bean-bag-in-hole, and red-bean-bag-in-hole classes.

        Loops through each frame in the video file, image file, or camera stream and runs an inference on it. Draws bounding boxes and labels around each identified class (detection).

        Args:
            file_path (str, optional): The path to the video or image file to run an inference on. Defaults to None.
            camera_port (int, optional): The camera port to use for live camera stream inference. Defaults to 0.
            calculate_score (bool, optional): Used to decide whether or not to also calculate the score in addition to running an inference. Defaults to False.
        """
        # Get a pretrained cornhole detection model
        cornhole_model = YOLO(cornhole_train.find_trained_model())

        # Annotator to draw boundary boxes around the subjects of interest
        bounding_box_annotator = sv.BoundingBoxAnnotator()
        # Annotator to display labels for subjects of interest
        label_annotator = sv.LabelAnnotator()

        # If a file path is not specified, run an inference on a live camera stream
        if file_path is None:
            cap = cv2.VideoCapture(camera_port)
            # Use a wait_key_duration of 1 so the frames of the camera stream are continuously displayed for the user to see
            wait_key_duration = 1
        # If a file path is specified, run an inference on that file
        else:
            cap = cv2.VideoCapture(file_path)
            # Use a wait_key_duration of 0 so the user can hold down a key to analyze the frames of the image or video one-by-one
            # Using a wait_key_duration of 0 prevents image files from closing immediately after an inference has been run on it
            wait_key_duration = 0

        # Loop through the video frames
        while cap.isOpened():
            # Read a frame from the video capture device
            success, frame = cap.read()

            if success:
                # Run a YOLOv8 inference on the frame and avoid double counts (if an object may be predicted as multiple classes)
                self.results = cornhole_model(frame, agnostic_nms=True)[0]

                # Update the detections variable to contain all detections in the current frame
                # Helps visualize detections and keep score/counts for the bean bags
                detections = sv.Detections.from_ultralytics(self.results)
                # Store the blue bean bag detections
                self.blue_bean_bag_detections = detections[
                    detections.class_id == self.BLUE_BEAN_BAG_CLASS_ID
                ]
                # Store the red bean bag detections
                self.red_bean_bag_detections = detections[
                    detections.class_id == self.RED_BEAN_BAG_CLASS_ID
                ]
                # Store the blue bean bags in hole detections
                self.blue_bean_bags_in_hole_detections = detections[
                    detections.class_id == self.BLUE_BEAN_BAG_IN_HOLE_CLASS_ID
                ]
                # Store the red bean bags in hole detections
                self.red_bean_bags_in_hole_detections = detections[
                    detections.class_id == self.RED_BEAN_BAG_IN_HOLE_CLASS_ID
                ]

                # Display the boundary box with the class name and probability
                self.annotated_frame = bounding_box_annotator.annotate(
                    scene=frame, detections=detections
                )
                # Display the labels for each subject of interest
                self.annotated_frame = label_annotator.annotate(
                    scene=frame, detections=detections
                )
                # Display the score and count totals if necessary
                if calculate_score:
                    self.calculate_score()
                    self.display_information()

                # Display the annotated frame
                cv2.imshow("Inference", self.annotated_frame)
            else:
                break

            # Break the loop if 'q' is pressed
            if cv2.waitKey(wait_key_duration) == ord("q"):
                break

        # Close the camera
        cap.release()
        # Close all the windows displaying the frames of the image, video, or camera stream
        cv2.destroyAllWindows()
