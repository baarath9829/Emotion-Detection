import dlib
import cv2
import numpy as np
import math
#import cvtest as f1



distance={}

def train(li):
    with open("train.txt", "a") as f:
        for i in l:
            st=str(i)
            f.write(st)
            f.write(",")
        s=input("Enter the expression:")
        if s=="happy":
            f.write("1")
            f.write(",T\n")
        elif s=="neutral":
            f.write("10")
            f.write(",T\n")
        elif s=="sad":
            f.write("0")
            f.write(",T\n")
        

def retrieve_data():
    data_array = np.array([[]])
    li=[]
    with open("train.txt", "r") as file:
        data = file.readlines()
        for line in data:
            s=""
            for i in line:
                if i>='0'or i<='9' or i=='.':
                    if i!=',':
                        s=s+i;
                if i==',' and i!='T':
                    li.append(float(s))
                    s=""
            print(li[2278])
            data_array=np.append(data_array, li)
            #classifier(list li)
            print(len(li))
            li=[]
def distanceline(img,array_points):
    for i in range(68):
        x1,y1=array_points[i]
        for j in range(i+1,68):
            x2,y2=array_points[j]
            cv2.line(img,(x1,y1),(x2,y2),(0,0,255),1)
            distance[str(i)+"->"+str(j)]=euclidean_distance(x1,y1,x2,y2)
def euclidean_distance(x1,y1,x2,y2):
    return math.sqrt( (x1-x2)**2 + (y1-y2)**2 )

def plot(image,landmark): #passing img AS argument causing problem
    image_c = image.copy
    for point in landmark:
        pos = (point[0,0],point[0,1])
        print (pos)
        cv2.circle(image_c, pos, 3, color=(0, 255, 0))
    return image_c
        
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
win = dlib.image_window()
img =cv2.imread("testface.jpg",-1)
image = img.copy()
win.set_image(img)
dets = detector(img,1)
for k, d in enumerate(dets):
    shape = predictor(img,d)
    print("Part 0: {}, Part 1: {} part2 : {}".format(shape.part(0),shape.part(1),shape.part(2)))
    mat = np.matrix([[p.x, p.y] for p in shape.parts()])
    
    win.add_overlay(shape)

list=[]    
for point in mat:
        pos = (point[0,0],point[0,1])
        #print (pos)
        list.append(pos)
        cv2.circle(image, pos, 3, (0, 255, 0),-1)    
win.add_overlay(dets)
cv2.imshow("landmark", image)
distanceline(image,list)


l=[]
for i in range(68):
    for j in range(i+1,68):
        l.append(round(distance[str(i)+"->"+str(j)],2))

train(l)
retrieve_data()



dlib.hit_enter_to_continue()
cv2.waitKey(0)
cv2.destroyAllWindows()

