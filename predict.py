import dlib
import cv2
import numpy as np
import math
from sklearn import tree
from sklearn.externals import joblib
#import sys

imgname = raw_input("input")

distance=[]

def load_dataset(li,s):
    with open("train.txt", "a") as f:
        for i in li:
            f.write(str(i))
            f.write(",")
        if s=="happy":
            f.write("1")
            f.write(",T\n")
        elif s=="neutral":
            f.write("10")
            f.write(",T\n")
        elif s=="sad":
            f.write("0")
            f.write(",T\n")


def classify (x):
    if x == "happy":
        y = 1.
    elif x=="sad":
        y= 0.
    else:
        y= 10.
    return y

def distance_cal(array_points):
    for i in range(68):
        x1,y1=array_points[i]
        for j in range(i+1,68):
            x2,y2=array_points[j]
            #cv2.line(img,(x1,y1),(x2,y2),(0,0,255),1)
            distance.append(round(euclidean_distance(x1,y1,x2,y2),2))

def euclidean_distance(x1,y1,x2,y2):
    return math.sqrt( (x1-x2)**2 + (y1-y2)**2 )

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
win = dlib.image_window()
img =cv2.imread(imgname,-1)
image = img.copy()
win.set_image(img)
dets = detector(img,1)
for k, d in enumerate(dets):
    shape = predictor(img,d)
    #print("Part 0: {}, Part 1: {} part2 : {}".format(shape.part(0),shape.part(1),shape.part(2)))
    mat = np.matrix([[p.x, p.y] for p in shape.parts()])
    
    win.add_overlay(shape)

pos_list=[]    
for point in mat:
        pos = (point[0,0],point[0,1])
        #print (pos)
        pos_list.append(pos)
        cv2.circle(image, pos, 3, (0, 255, 0),-1)    
win.add_overlay(dets)
cv2.imshow("landmark", image)
distance_cal(pos_list)

#fetch classifier
model = joblib.load('finalized_model.sav')

output = model.predict([distance])
if (output[0] == 1.):
    ans="happy"
elif(output[0] == 10.):
    ans="neutral"
elif(output[0] == 0.):
    ans="sad"
print (ans)

cv2.waitKey(0)


result = input("enter 1 or 0 to represent ans correct  or wrong respectively : ")
if (result==0):
    feedback = raw_input("enter the expression you think it is : ")
    print (feedback)
    print (type(feedback))
    load_dataset(distance,feedback)
    print (classify(feedback))
    #model = model.fit([distance],[classify(feedback)])
else:
    load_dataset(distance,ans)
    #model = model.fit([distance],[classify(ans)])

#joblib.dump(model, open('finalized_model.sav', 'wb'))

dlib.hit_enter_to_continue()
cv2.destroyAllWindows()
