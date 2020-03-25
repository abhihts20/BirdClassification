import cv2
import numpy as np
import os



path,dirs,files=next(os.walk("renamed"))
file_count=len(files)
xtest = np.empty((file_count, 784), dtype=np.int32)
print(type(xtest))
for j in range(0, file_count):
    data = []
    img = cv2.imread("renamed\\" + str(j + 1) + ".jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (28, 28))
    for i1 in range(0, 28):
        for j1 in range(0, 28):
            data.append(img[i1][j1])

    for i in range(0, len(data)):
        xtest[j][i] = data[i]

ytest = ['blackkite', 'blackstilt', 'chestnutbelliedrockthrush', 'eagle', 'elfowl', 'greattit','himalayanbulbul','himalayangriffon','magpierobin','mynah','pigeon','raven','sparrow','vulture','whitecappedredstart']
# output array
# from sklearn.neighbors import KNeighborsClassifier
# classifier=KNeighborsClassifier(n_neighbors=1)
# classifier.fit(xtest,ytest)
# print(classifier.score(xtest,ytest))


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(xtest,ytest)
print(classifier.score(xtest,ytest))
# model.fit(xtest, ytest)
# print(model.score(xtest, ytest))
# avg_prec=average_precision_score(X_train,X_test)
# X_train,X_test,Y_Train,Y_Test=train_test_split(xtest,ytest,test_size=0.20)
# scaler=StandardScaler()
# scaler.fit(X_train)
# X_train=scaler.transform(X_train)
# X_test=scaler.transform(X_test)
# classifier=KNeighborsClassifier(n_neighbors=2)
# classifier.fit(X_test,Y_Train)
# Y_pred=classifier.predict(X_test)
# print(avg_prec)