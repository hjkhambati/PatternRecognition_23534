import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


class Classification:
    def __init__(self, filename, dep, indep):
        self.data_set = pd.read_csv(filename)
        self.x = self.data_set.iloc[:, indep].values
        self.y = self.data_set.iloc[:, dep].values
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.25,
                                                                                random_state=0)

    def naive_bayes(self):
        classifier = GaussianNB()
        classifier.fit(self.x_train, self.y_train)
        y_pred = classifier.predict(self.x_test)
        cm = confusion_matrix(self.y_test, y_pred)
        print(cm)
        print("Gaussian Naive Bayes model accuracy(in %):", accuracy_score(self.y_test, y_pred) * 100)

    def svm(self):
        classifier = SVC(kernel='linear', random_state=0)  # support vector classifier
        classifier.fit(self.x_train, self.y_train)
        y_pred = classifier.predict(self.x_test)
        cm = confusion_matrix(self.y_test, y_pred)
        print(cm)
        print("Support Vector Machine model accuracy(in %):", accuracy_score(self.y_test, y_pred) * 100)

    def knn(self):
        classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
        classifier.fit(self.x_train, self.y_train)
        y_pred = classifier.predict(self.x_test)
        cm = confusion_matrix(self.y_test, y_pred)
        print(cm)
        print("KNN model accuracy(in %):", accuracy_score(self.y_test, y_pred) * 100)


c1 = Classification("Placement_Data_Full_Class.csv", 13, [1, 2, 3, 4, 5, 6, 7, 8, 10, 12])
c1.naive_bayes()
c1.svm()
c1.knn()
