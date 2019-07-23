#!/usr/bin/env python3
from kmeans import KMeans

PATH = "images/pein.jpg"
K = 8
ITERATIONS = 100

if __name__ == "__main__":
    '''Output can be found at examples/pein'''
    model = KMeans(PATH, k=K)
    print(
        "Image has shape {0}, with size {1}, and {2} features.".format(
            model.shape, model.size, model.features))
    model.train(iterations=ITERATIONS, verbose=True)
    clusters, labels = model.clusters, model.labels
    # save processed image and plot
    model.save_image()
    model.save_plot()
