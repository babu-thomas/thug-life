import cv2

font = cv2.FONT_HERSHEY_SIMPLEX
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
eyeglass_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
thug_life = False
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 4)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = eyeglass_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 4)

    cv2.putText(frame, 'Press T for Thug Life', (5, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow('Normal Life', frame)

    if thug_life:
        cv2.imshow('Thug Life Motherfuckers!!!', gray)
        thug_life = False
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('t'):
        thug_life = True

cap.release()
cv2.destroyAllWindows()
