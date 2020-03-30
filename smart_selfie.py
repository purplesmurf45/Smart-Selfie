import cv2 as cv
import numpy as np
import dlib

#Initialisation of dlib's face detector (HOG-based) and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#Indices for the coordinates corresponding to the mouth region are 49 to 68
(mStart,mEnd) = (48, 67)

smile_const = 5
counter = 0
selfie_no = 0

#Converts rectangle predicted by dlib with bounding box in OpenCV
#with the format (x,y,w,h)
def rect_to_box(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)
#function to convert the facial coordinates recognised by the predictor
#to bounding box with the format (x,y,w,h)

def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68,2), dtype=dtype)

    #Loop over the 68 facial landmarks and convert them
    #into a tuple(x,y)
    for i in range(68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords

def smile(shape):
    left = shape[48]
    right = shape[54]
    
    #average of the points in the center of the mouth
    mid = (shape[51] + shape[62] + shape[66] + shape[57])/4

    #Perpendicular distance between the mid and the line joining left and right
    dist = np.abs(np.cross(right-left, left-mid)) / np.linalg.norm(right - left)
    return dist

cam = cv.VideoCapture(0)

while(cam.isOpened()):
    
    ret, image = cam.read()
    image = cv.flip(image, 1)
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)

    #Detect Faces in the Grayscale image
    rects = detector(gray,2)

    #Loop over each face to draw a rectangle around it
    for i in range(len(rects)):
        (x, y, w , h) = rect_to_box(rects[i])

        #Determine the facial landmarks for each face detected
        shape = predictor(gray, rects[i])

        shape = shape_to_np(shape)

        mouth = shape[mStart:]

        smile_param = smile(shape)
        cv.putText(image,"SP: {:.2f}".format(smile_param),(300,30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0,255), 2)

        if smile_param > smile_const:
            cv.putText(image, "Smile Detected", (300,60),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7,(0, 255, 0), 2)
            counter +=1
            if counter >=3:
                selfie_no += 1
                ret, frame = cam.read()
                img_name = "Pic_{}.png".format(selfie_no)
                cv.imwrite(img_name, frame)
                print("{} taken!".format(img_name))
                counter = 0
        else:
            counter = 0
            
    cv.imshow('live_face',image)
    key = cv.waitKey(1)
    if key == 27:
        break
            
cam.release()
cv.destroyAllWindows()

