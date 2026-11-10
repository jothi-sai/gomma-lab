
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix


def svm_classification(test_size=0.3, kernel="linear"):
    iris = load_iris()

    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    model = SVC(kernel=kernel)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(
        cm, annot=True, fmt="d",
        xticklabels=iris.target_names,
        yticklabels=iris.target_names
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("SVM Classification")
    plt.show()

    return accuracy
