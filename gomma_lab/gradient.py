
import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(x=8, y=8, lr=0.1, iterations=20):
    history = []

    for i in range(iterations):
        x = x - lr * 2 * x
        y = y - lr * 2 * y
        history.append((x, y))

    print("Minimum x:", x)
    print("Minimum y:", y)
    print("Minimum value:", x**2 + y**2)

    return history


def plot_gradient(history):
    points = np.array(history)
    plt.plot(points[:, 0], label="X")
    plt.plot(points[:, 1], label="Y")
    plt.xlabel("Iterations")
    plt.ylabel("Values")
    plt.title("Gradient Descent")
    plt.legend()
    plt.show()

