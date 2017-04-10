import cv2

# For resizing while maintaining aspect ratio
def resize(image, width=None, height=None, type=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None:
        if height is None:
            return image
        else:
            ratio = height / float(h)
            dim = (int(w * ratio), height)
    else:
        if height is None:
            ratio = width / float(w)
            dim = (width, int(h * ratio))
        else:
            dim = (width, height)

    return cv2.resize(image, dim, interpolation=type)

font = cv2.FONT_HERSHEY_SIMPLEX
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
thug_life = False
thug_cap = cv2.imread('thug_life_cap_edit.png')
cap_mask = cv2.imread('cap_mask.png', cv2.IMREAD_GRAYSCALE)
cap_inv_mask = cv2.imread('cap_inv_mask.png', cv2.IMREAD_GRAYSCALE)
# Uncomment below line to live the thug life
# video = cv2.VideoCapture(0)

while True:
    # Uncomment below line to live the thug life
    # ret, frame = video.read()
    frame_orig = cv2.imread('snoop_dogg.jpg')
    frame_copy = frame_orig.copy()
    gray = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)

    face_roi = None
    roi_dims = []

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi_dims.append((x, y, w, h))
        cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)
        face_roi = frame_copy[y:y + h, x:x + w]

    cv2.putText(frame_copy, 'Press T to live the Thug Life', (5, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow('Thug Life', frame_copy)

    if thug_life:
        # Tune this according to the pictures
        # TODO: Figure out a better way
        cap_width_multiplier = 1.5

        cap_width = int(roi_dims[0][2] * cap_width_multiplier)
        # Maintain aspect ratio
        thug_cap_resized = resize(thug_cap, width=cap_width)
        thug_cap_resized_gray = cv2.cvtColor(thug_cap_resized, cv2.COLOR_BGR2GRAY)

        cap_height = thug_cap_resized.shape[0]

        cap_x = roi_dims[0][0]
        cap_y = roi_dims[0][1]

        cap_roi = frame_copy[(cap_y - cap_height):cap_y, cap_x:(cap_x + cap_width)]

        cap_mask_resized = resize(cap_mask, width=cap_width)
        cap_inv_mask_resized = cv2.bitwise_not(cap_mask_resized)


        roi_bg = cv2.bitwise_and(cap_roi, cap_roi, mask=cap_mask_resized)
        cap_fg = cv2.bitwise_and(thug_cap_resized, thug_cap_resized, mask=cap_inv_mask_resized)
        cap_roi = cv2.add(roi_bg, cap_fg)

        with_cap = cv2.add(cap_roi, thug_cap_resized)

        frame_orig[(cap_y - cap_height):cap_y, cap_x:(cap_x + cap_width)] = cap_roi
        cv2.putText(frame_orig, 'Thug life Motherf*ckers!!!', (5, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('Thug Life', frame_orig)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('t'):
        thug_life = True
    elif key == ord('n'):
        thug_life = False
# Uncomment below line to live the thug life
# video.release()
cv2.destroyAllWindows()
