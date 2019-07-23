#!/usr/bin/env python3
from kmeans import KMeans

PATH = "images/apple.jpg"
K = 5
ITERATIONS = 10

if __name__ == "__main__":
    '''Output can be found at examples/apple '''
    model = KMeans(PATH, k=K)
    model.train(iterations=ITERATIONS, verbose=True)
    clusters, labels = model.clusters, model.labels
    # save processed image and plot
    model.save_image()
    model.save_plot()
