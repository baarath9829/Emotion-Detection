from sklearn import tree
from sklearn.externals import joblib
features = []
labels = []
def retrieve_data():
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
            labels.append(li.pop())
            features.append(li)
            li=[]

retrieve_data()
clf = tree.DecisionTreeClassifier()
test = features.pop()
target = labels.pop()
clf = clf.fit(features, labels)
# save the model to disk
filename = 'finalized_model.sav'
joblib.dump(clf, open(filename, 'wb'))
model = joblib.load(filename)

print (model.predict([test]))
print (target)
