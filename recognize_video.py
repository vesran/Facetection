import cv2
import params
import imutils

from facetection.recognize import identify, add_info_on_image


def recognize_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    video_resolution = params.video_resolution

    # frame_width = int(cap.get(3))
    # frame_height = int(cap.get(4))
    # out = cv2.VideoWriter('./images/friend_scene_output.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 20, (frame_width, frame_height))

    while cap.isOpened():
        _, frame = cap.read()
        if frame is None or cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame = imutils.resize(frame, width=video_resolution[1])

        # Get id of each faces
        ids_info = identify(frame)

        # Put id into the image
        add_info_on_image(ids_info, frame)

        # Display frame
        cv2.imshow('frame', frame)
        # out.write(frame)

    # out.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    recognize_from_video('./images/local/chandler_alone.mp4')
