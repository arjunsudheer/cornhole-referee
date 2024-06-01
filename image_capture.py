import cv2
import os


# Create the "cornhole-dataset" directory if it doesn't already exist
if not os.path.isdir("./cornhole-dataset-original-images"):
    os.mkdir("./cornhole-dataset-original-images")
# Change the working directory to the cornhole-dataset directory so all images are saved there
os.chdir("./cornhole-dataset-original-images")

# Start the camera stream
camera = cv2.VideoCapture(0)
# Keeps track of the current image count, used for the image filename
image_counter = 1

while True:
    # Read the next frame from the camera stream
    success, image = camera.read()
    # Abort if camera cannot read anymore frames
    if not success:
        print("\nno frames have been grabbed, aborting")
        break
    # Display the camera feed frames
    cv2.imshow("Image", image)
    # Check if the user pressed any key
    # Wait for 1 ms so the frames continuously display
    key_pressed = cv2.waitKey(1)
    # Quit the capture process if the user presses 'q'
    if key_pressed == ord("q"):
        break
    # Save the current frame if the user presses 's'
    elif key_pressed == ord("s"):
        filename = f"image_{image_counter}.jpg"
        cv2.imwrite(filename, image)
        image_counter += 1

# close the camera
camera.release()