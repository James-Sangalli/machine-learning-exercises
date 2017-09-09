from sklearn import tree

# 0 for bumpy, 1 for smooth

features = [[140, 1], [130, 1], [150, 0], [170, 0]]

#0 for apple, 1 for orange

labels = [0, 0, 1, 1]

clf = tree.DecisionTreeClassifier()

clf = clf.fit(features, labels)

print clf.predict([[150, 0]])
