import cv2

cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

capture = cv2.VideoCapture(0)
while True:
    ret, frame = capture.read()
    # image = cv2.medianBlur(frame, 5)
    #  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #   ret, jpeg = cv2.imencode('.png', frame)
    #    cv2.imshow('video', frame)

    # get frame shape and size in bytes
    h, w = frame.shape[:2]
    frame_size = frame.size

    # blur the image a bit
    imblur = cv2.medianBlur(frame, 5)

    # transform to black and white
    imgray = cv2.cvtColor(imblur, cv2.COLOR_BGR2GRAY)

    # filter bits with to high or too low value. The THRESH_TRIANGLE type will calculate the optimal threshold
    # and ignore the value I specify. That's why threshold is set to 0
    ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_TRIANGLE)

    thresh_size = thresh.size

    # cannot encode or the contours calculation won't work
    #imgray = cv2.imencode('.jpg', imgray)

    image = imgray
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # approximate the contour shape
    contours = [cv2.approxPolyDP(cnt, 3, True) for cnt in contours]
    # draw contours on top of current image
    cv2.drawContours(image, contours, -1, (0, 0, 0), 2)
    cv2.imshow('image', image)

    # eperimenting with Canny edging function. this finds also internal edges
    # changing the min and max value influences the precision
    edged = cv2.Canny(image, 100, 250)
    edged_size = edged.size
    cv2.imshow("Edges", edged)

    # continue until ESC key is pressed
    if cv2.waitKey(1) == 27:
        break

capture.release()
cv2.destroyAllWindows()
