import cv2
import numpy as np


if __name__ == "__main__":
    image = cv2.imread(r"data\road2.jpg")


    left_top = (0, 520)
    right_top = (image.shape[1], 520)
    right_bottom = (image.shape[1], image.shape[0]) # 0 - y, 1 - x
    left_bottom = (0, image.shape[0])

    cv2.circle(image, left_top, 5, (0, 255, 0), -1)
    cv2.circle(image, right_top, 5, (0, 0, 255), -1)
    cv2.circle(image, right_bottom, 5, (0, 0, 255), -1)
    cv2.circle(image, left_bottom, 5, (0, 0, 255), -1)


    original_points = np.float32([left_top, right_top, left_bottom, right_bottom])
    offset = 500
    new_points = np.float32([[offset, 0], [image.shape[1] - offset, 0], [offset, image.shape[0]], [image.shape[1] - offset, image.shape[0]]])

    matrix = cv2.getPerspectiveTransform(original_points, new_points)
    result = cv2.warpPerspective(image, matrix, (image.shape[1], image.shape[0]))

    small_image = cv2.resize(image, None, fx=0.5, fy=0.5)
    small_result = cv2.resize(result, None, fx=0.5, fy=0.5)

    cv2.imshow('Original Image', small_image)
    cv2.imshow('Transformed Image', small_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()