#!/usr/bin/env python3
import sys
import uuid
from cv2 import cv2 as cv
import numpy as np
import matplotlib.pyplot as plot
from matplotlib.colors import rgb2hex
from collections import Counter as count


class KMeans:
    """Utilizes k-means clustering algorithm to do image segmentation.
    K-means algorithm is an unsupervised learning technique where
    given data is sorted into k clusters.
    
    Compresses image and constructs a colour palette (model.pie) from 
    the given image. Colour is stored as RGB values (model.clusters) 
    and pie chart is constructed based on on the instances of the 
    colour.

    Attributes:
        k: (int)
        path: (str)
        verbose: (bool)
        image: (numpy.ndarray)
        size: (int)
        shape: tuple
        featues: (int)
        training_set: (numpy.ndarray)
        labels: (numpy.ndarray)
        clusters: (numpy.ndarray)
        distorted_distance: (numpy.float64)
        compressed_image: (numpy.ndarray)
        compressed_image_id = (str)
        pie: (matplotlib.figure.Figure)
        pie_id: (str)

    To use:
    >>> model = KMeans('apple.jpg', k=5)
    >>> model.k
    5
    >>> model.path
    'apple.jpg'
    >>> model.shape
    (460, 460, 3)
    >>> model.size 
    634800
    >>> model.features
    3
    >>> model.train(iterations=10, verbose=False)
    >>> model.verbose
    False
    >>> model.training_set.shape
    (211600, 3)
    >>> model.clusters.shape
    (5, 3)
    >>> model.labels.size
    211600
    """
    def __init__(self, path, k=5):
        """Initialized KMeans class with useful variables """
        self._k = k
        self._path = path
        self._verbose = False
        self._image = cv.cvtColor(self._open_image(path), 4)
        self._image_size = self.image.size
        self._image_shape = self.image.shape
        self._training_set = self.image.reshape(self.shape[0]*self.shape[1], self.features)
        self._labels = np.zeros(tuple([self.shape[0]*self.shape[1]])).astype(int)
        self._clusters = np.random.rand(self.k, self.features)
        self._distance = self._distorted_distance()
        self._compressed_image = None
        self._pie = None
        self._pie_id = None
        self._compressed_image_id = None

    def _open_image(self, path):
        """Given path of image file, returns a numpy.ndarray object"""
        return cv.imread(path, 1)
        #.astype(float)


    def view(self, image, cvtColor=False):
        """Displays image figure"""
        temp = image
        if cvtColor:
            temp = cv.cvtColor(temp, 4)
        plot.imshow(temp)
        plot.show()

    
    @property
    def k(self):
        """Gets or sets the k value of K-means """
        return self._k

    @k.setter
    def k(self, k): 
        self._k = k

    @property
    def path(self):
        """Gets or sets the image path"""
        return self._path
    
    @path.setter
    def path(self, path):
        self._path = path
    
    @property
    def verbose(self):
        """Gets or sets verbose value"""
        return self._verbose
    
    @verbose.setter
    def verbose(self, arg):
        self._verbose = arg

    @property
    def image(self):
        """Gets image"""
        return self._image

    @property
    def size(self):
        """Gets size of image"""
        return self._image_size
    
    @property
    def shape(self):
        """Gets shape of image"""
        return self._image_shape

    @property
    def features(self):
        """Gets the number of image features"""
        return self.shape[2]
    
    @property
    def training_set(self):
        """Gets the set used for training the model """
        return self._training_set

    @property
    def labels(self):
        """Gets the model labels"""
        return self._labels
    
    @property
    def clusters(self):
        """Gets the model clusters"""
        return self._clusters

    @property
    def distorted_distance(self):
        """Gets or sets the model's distored distance"""
        return self._distance

    @distorted_distance.setter
    def distorted_distance(self, distance):
        self._distance = distance

    @property
    def compressed_image(self):
        """Gets or sets model's processed image"""
        return self._compressed_image
    
    @compressed_image.setter
    def compressed_image(self, compressed_image):
        self._compressed_image = compressed_image
    
    @property
    def pie(self):
        """Gets or sets model's pie chart"""
        return self._pie
    
    @pie.setter
    def pie(self, pie):
        self._pie = pie

    @property
    def pie_id(self):
        """Gets or sets id associated with the model's saved pie chart"""
        return self._pie_id
    
    @pie_id.setter
    def pie_id(self, id):
        self._pie_id = id
    
    @property
    def compressed_image_id(self):
        """Gets or sets id associated with the model's saved processed image"""
        return self._compressed_image_id

    @compressed_image_id.setter
    def compressed_image_id(self, id):
        self._compressed_image_id = id


    def train(self, iterations=20, verbose=True):
        """Begins the training process.

        This method will stop the training process if and only if distortion distance has converged
        or the number of iterations given in the parameters have completed.

        Args:
            iterations: number of times learning algorithm will traverse through the dataset (int > 0)
            verbose: whether to print standard output (bool)
        """

        self.verbose = verbose

        i = 0
        optima_reached = False
    
        while (i < iterations) and not(optima_reached):
            
            if self.verbose:
                print("Iteration:   {0}".format(i))

            # locate the nearest cluster
            cluster_labels = self._calculate_nearest_cluster(enumerate(self.training_set), list(None for cluster in range(self.k)))    
            self._relocate_clusters(cluster_labels) # optimize cluster location
            
            new_distance = self._distorted_distance() # distance between training data and clusters (cost)

            if new_distance >= self.distorted_distance:
                if self.verbose:
                    print("Distorted distance has converged at {0}".format(self.distorted_distance))
                    print("Exiting...")
                optima_reached = True
            else:
                self.distorted_distance = new_distance
                if self.verbose:
                    print("distorted_distance:  ", "{0:.2f}".format(self.distorted_distance))
            
            i+=1
        
        # reshape image vectors
        self._reshape_image(np.zeros((self.shape[0]*self.shape[1], self.features)))       
        self._construct_pie() # construct colour palette  


    def _calculate_nearest_cluster(self, pixels, cluster_labels):

        """Assigns pixel value to the nearest cluster (using euclidean distance)

        Args:
            pixels: image vectors (numpy.ndarray) 
            cluster_labels: indices of k clusters (list) 
        Returns:
            cluster labels with assigned RGB values (numpy.ndarray)
        """

        # assign pixel (RGB) to nearest cluster label (index)
        for index, rgb in pixels:
            rgb_vector = np.tile(rgb, (self.k,1))
            self._labels[index] = np.argmin(self._euclid_distance(rgb_vector, self._clusters), axis=0)
     
            if cluster_labels[self._labels[index]] is None:
                cluster_labels[self._labels[index]] = list()

            cluster_labels[self._labels[index]].append(rgb)
        
        return cluster_labels


    def _relocate_clusters(self, cluster_labels):
        """Moves each cluster towards the mean of RGB values assigned to it

        Args:
            cluster_labels: indices of k clusters, with assigned RGB values (list)
        """
        for cluster_label in range(self.k):
            if cluster_labels[cluster_label] is not None:
                # mean of the pixels assigned to cluster
                p_sum, p_count = np.asarray(cluster_labels[cluster_label]).sum(axis=0), len(cluster_labels[cluster_label])
                self._clusters[cluster_label] = p_sum / p_count

    def _euclid_distance(self, A, B, axis=1):
        """Calculates euclidean distance between point A & B on the given axis
        
        Args:
            A: (numpy.ndarray) 
            B: (numpy.ndarray)
            axis: (int)

        Returns:
            distance between points A and B 
        """
        return np.linalg.norm(A - B, axis=axis)
    
    
    def _distorted_distance(self):
        """Sum of squared distances (cost function)

        Returns:
            the new distance value (float)
        """
        distance = 0
        for i, pixel in enumerate(self.training_set):
            distance += self._euclid_distance(pixel, self.clusters[self.labels[i]], axis=0)
        return distance
        
    
    def _reshape_image(self, image_vectors):
        """Reshape image vectors back original shape

        Args:
            image_vectors: (numpy.ndarray) 
        """

        for pixel in range(image_vectors.shape[0]):
            cluster_label = self.labels[pixel]
            RGB = self.clusters[cluster_label]
            image_vectors[pixel] = (1/255)*RGB
        
        # processed image (original size)
        self.compressed_image = image_vectors.reshape(self.shape)

    def _construct_pie(self):
        """ Constructs a pie chart based on identified colours and instances"""
        
        # count labels and instances
        label_count = count(self.labels)
        label_instances = [instance for instance in label_count.values()]
        colours = [rgb2hex(self.clusters[label]/255) for label in label_count.keys()]
        self.pie = plot.figure()
        plot.pie(x=label_instances, labels=label_count.keys(), colors=colours, autopct="%.2f%%")
        plot.tight_layout()
        plot.axis('equal')

    def save_image(self):
        """Saves processed image in project directory"""
        self.compressed_image_id = str(uuid.uuid4().hex)
        plot.imsave(str(self.compressed_image_id + "{}").format(".png"), self.compressed_image)
        
        if self.verbose:
            print("Compressed image saved at " + (str(self.compressed_image_id + "{}").format(".png")))

    def save_plot(self):
        """Saves pie chart in project directory"""
        self.pie_id = str(uuid.uuid4().hex)
        self.pie.savefig(str(self.pie_id + "{}").format(".png"))
        
        if self.verbose:
            print("Pie chart saved at " + (str(self.pie_id + "{}").format(".png")))
