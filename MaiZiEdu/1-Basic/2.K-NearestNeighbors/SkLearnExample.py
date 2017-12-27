#encoding=utf-8
from sklearn import datasets
from sklearn import neighbors

knn = neighbors.KNeighborsClassifier()

iris = datasets.load_iris()
# save data
# f = open("iris.data.csv", 'wb')
# f.write(str(iris))
# f.close()

print (iris)

knn.fit(iris.data, iris.target)

predictedLabel = knn.predict([[2.1, 6.2, 7.3, 3.4]])
print ("next")
# print ("predictedLabel is :" + predictedLabel)
print (predictedLabel)
