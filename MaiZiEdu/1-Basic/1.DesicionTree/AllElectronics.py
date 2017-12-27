#encoding=utf-8
import csv

from sklearn import preprocessing
from sklearn import tree
from sklearn.feature_extraction import DictVectorizer

# Read in the csv file and put features into list of dict and list of class label
allElectronicsData = open('AllElectronics.csv', 'rb')
reader = csv.reader(allElectronicsData)
headers = reader.next()

featureList = []
labelList = []

for row in reader:
    labelList.append(row[len(row) - 1])
    rowDict = {}
    for i in range(1, len(row) - 1):
        rowDict[headers[i]] = row[i]
    featureList.append(rowDict)

# Vetorize features
vec = DictVectorizer()
dummyX = vec.fit_transform(featureList).toarray()

# vectorize class labels
lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)

#print(headers)
#print(featureList)
#print("dummyX: " + str(dummyX))
#print(vec.get_feature_names())
#print("labelList: " + str(labelList))
#print("dummyY: " + str(dummyY))

# Using decision tree for classification
# clf = tree.DecisionTreeClassifier()
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(dummyX, dummyY)
print("clf: " + str(clf))

# Visualize model
with open("allElectronicInformationGainOri.dot", 'w') as f:
    tree.export_graphviz(clf, feature_names=vec.get_feature_names(), out_file=f)

oneRowX = dummyX[0, :]
print("oneRowX: " + str(oneRowX))

newRowX = oneRowX
newRowX[0] = 1
newRowX[2] = 0
print("newRowX: " + str(newRowX))

predictedY = clf.predict(newRowX.reshape(1,-1))
print("predictedY: " + str(predictedY))
