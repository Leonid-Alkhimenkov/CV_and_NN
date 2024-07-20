import cv2 as cv
import numpy as np

# Параметры для скользящего среднего
left_fit_buffer = []
right_fit_buffer = []
buffer_size = 10

def make_coordinates(image, line_parameters):
    if line_parameters is None:
        return None
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (13/20))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    global left_fit_buffer, right_fit_buffer

    left_fit = []
    right_fit = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope < -0.5:
                left_fit.append((slope, intercept))
            elif slope > 0.5:
                right_fit.append((slope, intercept))

    # Обновление буферов с параметрами линии
    if left_fit:
        left_fit_average = np.average(left_fit, axis=0)
        left_fit_buffer.append(left_fit_average)
        if len(left_fit_buffer) > buffer_size:
            left_fit_buffer.pop(0)
        left_fit_average = np.average(left_fit_buffer, axis=0)
    else:
        left_fit_average = None

    if right_fit:
        right_fit_average = np.average(right_fit, axis=0)
        right_fit_buffer.append(right_fit_average)
        if len(right_fit_buffer) > buffer_size:
            right_fit_buffer.pop(0)
        right_fit_average = np.average(right_fit_buffer, axis=0)
    else:
        right_fit_average = None

    left_line = make_coordinates(image, left_fit_average) if left_fit_average is not None else None
    right_line = make_coordinates(image, right_fit_average) if right_fit_average is not None else None

    return np.array([left_line, right_line]) if left_line is not None and right_line is not None else None

def canny(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    canny = cv.Canny(blur, 50, 150)
    return canny

def display_mid_line(image, left_line, right_line):
    mid_line_image = np.zeros_like(image)
    if left_line is not None and right_line is not None:
        x1_left, y1_left, x2_left, y2_left = left_line
        x1_right, y1_right, x2_right, y2_right = right_line

        mid_x1 = (x1_left + x1_right) // 2
        mid_y1 = (y1_left + y1_right) // 2
        mid_x2 = (x2_left + x2_right) // 2
        mid_y2 = (y2_left + y2_right) // 2

        # Укорачиваем среднюю линию, изменив конечные координаты
        y2 = int(image.shape[0] * (13/20))
        mid_x2 = int((y2 - mid_y1) * (mid_x2 - mid_x1) / (mid_y2 - mid_y1) + mid_x1)
        mid_y2 = y2

        cv.line(mid_line_image, (mid_x1, mid_y1), (mid_x2, mid_y2), (0, 255, 0), 10)
    return mid_line_image

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            if line is not None:
                x1, y1, x2, y2 = line
                cv.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 10)
    return line_image

def mask(image):
    height = image.shape[0]
    polygons = np.array([
        [(600, height // 1.4), (900, height // 1.4), (1320, height), (380, height)]
    ])
    mask = np.zeros_like(image)
    cv.fillPoly(mask, np.array([polygons], dtype=np.int64), 255)
    masked_image = cv.bitwise_and(image, mask)
    return masked_image

