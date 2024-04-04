import cv2
import numpy as np


points = np.array([[0, 0.85], [0.1, 0.82], [0.2, 0.77], [0.25, 0.70], [0.22, 0.57], [0.35, 0.55], [0.55, 0.53]])


def draw_curve(image, points, color=(0, 0, 255), thickness=2, radius=6):
    width = image.shape[1]
    height = image.shape[0]
    for idx in range(1, len(points)):
        pt1x = int(points[idx - 1][0] * float(width))
        pt1y = int(points[idx - 1][1] * float(height))
        pt2x = int(points[idx][0] * float(width))
        pt2y = int(points[idx][1] * float(height))
        
        cv2.line(image, (pt1x, pt1y), (pt2x, pt2y), color, thickness)
    
        cv2.circle(image, (pt1x, pt1y), radius, color, -1)
        if idx >= len(points) - 1:
            cv2.circle(image, (pt2x, pt2y), radius, color, -1)


def perspectiveTransform(image, points, matrix):
    points[:, 0] *= image.shape[1]
    points[:, 1] *= image.shape[0]
    transformed_points = cv2.perspectiveTransform(points.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    points[:, 0] /= image.shape[1]
    points[:, 1] /= image.shape[0]
    transformed_points[:, 0] /= image.shape[1]
    transformed_points[:, 1] /= image.shape[0]
    return transformed_points


def getPerspectiveTransform(image, original_points, new_points):
    original_points[:, 0] *= image.shape[1]
    original_points[:, 1] *= image.shape[0]
    new_points[:, 0] *= image.shape[1]
    new_points[:, 1] *= image.shape[0]
    matrix = cv2.getPerspectiveTransform(original_points, new_points)
    original_points[:, 0] /= image.shape[1]
    original_points[:, 1] /= image.shape[0]
    new_points[:, 0] /= image.shape[1]
    new_points[:, 1] /= image.shape[0]
    return matrix


if __name__ == "__main__":
    image = cv2.imread(r"data/road2.jpg")

    original_points = np.float32([[0, 0.55], [1, 0.55], [0, 1], [1, 1]])
    new_points = np.float32([[0.33, 0], [0.66, 0], [0.33, 1], [0.66, 1]])

    matrix = getPerspectiveTransform(image, original_points, new_points)
    transformed_points = perspectiveTransform(image, points, matrix)

    result = cv2.warpPerspective(image, matrix, (image.shape[1], image.shape[0]))
    
    small_image = cv2.resize(image, None, fx=0.5, fy=0.5)
    small_result = cv2.resize(result, None, fx=0.5, fy=0.5)

    draw_curve(small_result, transformed_points)
    draw_curve(small_image, points)

    cv2.imshow('Original Image', small_image)
    cv2.imshow('Transformed Image', small_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()