from imutils.video import VideoStream
from imutils.video import FPS
import params
import imutils
import time
import cv2

from facetection.recognize import add_info_on_image, identify


# initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(1.0)

# start the FPS throughput estimator
fps = FPS().start()

while True:
    # grab the frame from the threaded video stream
    frame = vs.read()
    frame = imutils.resize(frame, width=params.video_resolution[0])

    id_arr = identify(frame)
    add_info_on_image(id_arr, frame)

    # update the FPS counter
    fps.update()

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
