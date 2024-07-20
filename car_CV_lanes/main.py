import cv2 as cv
import numpy as np
import lanes

video = cv.VideoCapture("1.mp4")

if not video.isOpened():
    print("error")

cv.waitKey(1)

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    copy_img = np.copy(frame)
    try:
        frame = lanes.canny(frame)
        frame = lanes.mask(frame)
        lines = cv.HoughLinesP(frame, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=10)

        averaged_lines = lanes.average_slope_intercept(copy_img, lines)
        if averaged_lines is not None:
            line_image = lanes.display_lines(copy_img, averaged_lines)
            mid_line_image = lanes.display_mid_line(copy_img, averaged_lines[0], averaged_lines[1])
            combo = cv.addWeighted(copy_img, 0.8, line_image, 0.5, 1)
            combo = cv.addWeighted(combo, 0.8, mid_line_image, 0.5, 1)
        else:
            combo = copy_img

        cv.imshow("Video", combo)
    except Exception as e:
        print(e)
        pass

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv.destroyAllWindows()
