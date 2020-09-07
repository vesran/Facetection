import cv2

from facetection.recognize import identify, add_info_on_image


def recognize_from_image(src_image):
    image = cv2.imread(src_image)
    ids_info = identify(image)
    add_info_on_image(ids_info, image)

    cv2.imshow('img', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
