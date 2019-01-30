import cv2
import math
distance = {}
def plot(img, array_point):
    for point in array_point:
        x,y = point
        cv2.circle(img,(x,y),3,(0,255,0),-1)

def distanceline(img,array_points):
    for i in range(68):
        x1,y1=array_points[i]
        for j in range(i+1,68):
            x2,y2=array_points[j]
            cv2.line(img,(x1,y1),(x2,y2),(0,0,255),1)
            distance[str(i)+"->"+str(j)]=euclidean_distance(x1,y1,x2,y2)
def euclidean_distance(x1,y1,x2,y2):
    return math.sqrt( (x1-x2)**2 + (y1-y2)**2 )
                
shape = [[ 61, 188],
        [ 63, 214],
        [ 68, 241],
        [ 73, 268],
        [ 82, 292],
        [ 98, 313],
        [118, 330],
        [140, 344],
        [165, 348],
        [189, 342],
        [208, 327],
        [226, 309],
        [239, 288],
        [247, 263],
        [251, 237],
        [255, 211],
        [255, 185],
        [ 74, 174],
        [ 86, 160],
        [105, 156],
        [124, 158],
        [143, 164],
        [175, 162],
        [194, 156],
        [213, 153],
        [231, 157],
        [242, 170],
        [160, 181],
        [160, 199],
        [161, 217],
        [162, 236],
        [141, 249],
        [151, 252],
        [161, 254],
        [171, 251],
        [181, 248],
        [ 96, 187],
        [107, 179],
        [122, 179],
        [134, 188],
        [121, 191],
        [107, 192],
        [185, 187],
        [196, 177],
        [210, 177],
        [221, 184],
        [212, 190],
        [198, 190],
        [125, 283],
        [139, 278],
        [151, 275],
        [162, 278],
        [172, 274],
        [185, 277],
        [199, 281],
        [186, 290],
        [174, 296],
        [163, 298],
        [152, 297],
        [140, 292],
        [131, 284],
        [152, 283],
        [162, 284],
        [173, 282],
        [193, 282],
        [173, 283],
        [162, 285],
        [152, 283]]
img=cv2.imread("testface.jpg",1)
cv2.circle(img, (61, 188),3,(0, 255, 0),-1)

distanceline(img,shape)
plot(img,shape)
cv2.imshow("name",img)
cv2.waitKey(0)
cv2.destroyAllWindows()