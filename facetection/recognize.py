import cv2

from facetection.face_detection import find_faces
from facetection.predict import get_name, get_age, get_gender


def identify(image):
    arr = []
    faces_coords = find_faces(image)
    # print(faces_coords)
    # print()
    for face_coords in faces_coords:
        face = image[face_coords[2]:face_coords[3], face_coords[0]:face_coords[1]]
        # print(face.shape)

        age = get_age(face)
        gender = get_gender(face)
        name = get_name(face, threshold=0.5)

        arr.append((face_coords[0], face_coords[1], face_coords[2], face_coords[3],
                    age, gender, name))
    return arr


def add_info_on_image(ids_info, frame):
    for face_info in ids_info:
        start_x, start_y = face_info[0], face_info[2]
        end_x, end_y = face_info[1], face_info[3]
        age = face_info[4]
        gender = face_info[5]
        name = face_info[6]

        color = (255, 255, 0)
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color, 2)

        # Add texts
        age_gender_text = f'{age}yo - {gender}'
        cv2.putText(frame, age_gender_text, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.putText(frame, name, (start_x, start_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

