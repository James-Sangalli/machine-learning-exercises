from sklearn.datasets import load_iris 
import numpy as np
from sklearn import tree


iris = load_iris()

print iris.feature_names
print iris.target_names
print iris.data[0]
print iris.target[0]

#for i in range(len(iris.target)):
#	print iris.target[i]
#	print iris.data[i]
	
test_idx = [0, 50, 100]

#training dataset
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

#testing dataset
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print test_target
print clf.predict(test_data)	

#viz code (not related to ML just helps display as a graph)

from sklearn.externals.six import StringIO
import pydot

dot_data = StringIO()

tree.export_graphviz(clf, out_file = dot_data,
			feature_names = iris.feature_names,
			class_names = iris.target_names,
			filled= True,
			rounded = True,
			impurity=False)

graph = pydot.graph_from_dot_data(dot_data.getvalue())
#graph.write_pdf("iris.pdf")

print test_data[2], test_target[2]

print iris.feature_names, iris.target

