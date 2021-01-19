from dataset import gaussians_dataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from decisionTree import Forest
from sklearn import datasets


def main():

    X, Y = gaussians_dataset(4, [300, 300, 200, 100], [[1, 1], [-4, 6], [6, 6], [1, 10]],
                                                       [[2.5, 2.5], [4.5, 4.5], [1.5, 1.5], [2, 2]])

    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    # visualize the dataset
    fig, ax = plt.subplots(1, 2)

    ax[0].axis('off')
    ax[1].axis('off')

    ax[0].scatter(X[:, 0], X[:, 1], c=Y, s=40)
    plt.waitforbuttonpress()

    # iris = datasets.load_wine()
    # X = iris.data
    # Y = iris.target
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    model = Forest()
    model.fit(X_train, Y_train)
    p = model.predict(X_test)
    score = model.score(p[0], Y_test)
    print('Test set - Classification Accuracy: {}'.format(score))

    mean_score = model.score_mean(p[1], Y_test)
    print('\nMean individual accuracy: ', mean_score)

    # visualize results
    ax[1].scatter(X_test[:, 0], X_test[:, 1], c=p[0], s=40)
    plt.waitforbuttonpress()


# entry point
if __name__ == '__main__':
    main()
