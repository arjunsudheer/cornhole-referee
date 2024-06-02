import os

from ultralytics import YOLO


def train_cornhole_model():
    """
    Trains a cornhole detection model using YOLOv8.

    YOLOv8 will automatically generate a best.pt file for the trained model which can then be accessed in other parts of the cornhole detection program.
    """

    # Build a YOLOv8n model from pretrained weights
    model = YOLO("yolov8n.pt")

    # Display model information
    model.info()

    # Train the model on the custom cornhole images dataset
    # epochs is set to 100 by default, and imgsz is set to 640 x 640 pixels by default
    results = model.train(
        # Path to yaml file for custom cornhole images dataset
        data="/opt/homebrew/datasets/Cornhole Referee.v11i.yolov8/data.yaml",
        # Automatically use the maximum number of batches available based on the available RAM
        # batch=-1,
        # Optimize for apple silicon cores when training
        device="mps",
    )


# Returns the first occurrence of filename
# Returns an error message if more than one filename file was found
# Trains the model if no filename file was found
def find_trained_model(
    filename: str = "best.pt", search_path: str = "/opt/homebrew/runs/detect"
) -> str:
    """
    Finds a pretrained cornhole detection model.

    Uses the os.walk() method to search for the filename parameter in the user's filesystem. If filename is not found, then the train_cornhole_model() function
    will be called and this method will run again recursively. If more than one occurrence of filename is found, then an error message is printed.

    Args:
        filename (str): The filename that this function should search for. Defaults to "best.pt"
        search_path(str): The path that this method should start searching in. Defaults to the /opt/homebrew/runs/detect directory.

    Returns:
        str: The path to filename in the user's computer if only one occurrence (or no occurrence) of filename is found
        None: If more than one occurrence of filename is found
    """

    # Stores all the paths found to filename
    trained_models = []
    for root, dir, files in os.walk(search_path):
        if filename in files:
            trained_models.append(os.path.join(root, filename))

    # Present an error message to the user if multiple versions of the trained model is found
    if len(trained_models) > 1:
        print(
            f"Multiple versions of a trained model have been found at the paths listed below. Please only keep the latest version of the trained model.\n{trained_models}"
        )
        return None
    # Train the model if no trained model was found
    elif len(trained_models) == 0:
        print("No trained model was found. Starting model training now.")
        # Train the model
        train_cornhole_model()
        # Make a recursive call to this function to return the path to the newly trained model
        return find_trained_model()

    # return the path to the filename file
    return trained_models[0]
