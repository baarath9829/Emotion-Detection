import dlib
import cv2
import numpy
def plot(image,landmark):
    image_c = image.copy
    for point in landmark:
        pos = (point[0,0],point[0,1])
        cv2.circle(image_c, pos, 3, color=(0, 255, 0))
    return image_c
        
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
win = dlib.image_window()
img = cv2.imread("test1.jpg",0);
win.set_image(img)
dets = detector(img,1)
for k, d in enumerate(dets):
    shape = predictor(img,d)
    #print("Part 0: {}, Part 1: {} part2 : {}".format(shape.part(0),shape.part(1),shape.part(2)))
    mat = numpy.matrix([[p.x, p.y] for p in shape.parts()])
    win.add_overlay(shape)
    #plottedImg = plot(img,mat)
    
win.add_overlay(dets)
#cv2.imshow("landmark", plottedImg)
dlib.hit_enter_to_continue()
