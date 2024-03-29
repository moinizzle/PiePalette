# About
  Utilizes k-means clustering algorithm to do image segmentation.
  K-means is an unsupervised learning technique where
  given data is sorted into *k* clusters.
  Compresses image and constructs a colour palette from
  the given image. Colour is stored as RGB values and pie 
  chart is constructed based on the instances of the
  colour.
  
  I wrote the algorithm from scratch (with help from numpy) because I wanted to learn about the inner details and also because
  I find the subject particularly interesting. My implementation is specifically based on Andrew Ng's lecture notes on
  the subject, which can be found [here](notes/notes.pdf).
  
 # Instructions
  import module
 ```python3
 from kmeans import KMeans
 ```
 build model using image path
 ```python3
 model = KMeans('apple.jpg')
 ```
 set k value (number of clusters)
 ```python3
 model.k = 5
 ```
 train model and specify number of iterations
 ```python3
 model.train(iterations=10)
 ```
 save processed image + palette in your local directory
 ```python3
 model.save_image()
 model.save_plot()
 ```
  
# Examples
| Original Image        | Processed Image           | RGB Instances  |
| --------------------  | ------------------------- | ---------------|
|![apple](images/apple.jpg) | ![](examples/apple/60860f6b29b04b9abdc80943987291c7.png) | ![](examples/apple/ac869790dc514e7bb6830fba743e4162.png) |
|![kawhi](images/kawhi.jpg) | ![](examples/kawhi/4ee1a29c74474d04a555b28dffa01ab7.png)  | ![](examples/kawhi/4aaf86e3177c4cfe9e962812a8e0983f.png) |
|![pein](images/pein.jpg)   | ![](examples/pein/45524159151b4d60a238039ec9736260.png) | ![](examples/pein/6a2ae6cb0e424120bafd3792fab78f6c.png) |


# Credits:
Algorithm implemented according to [Andrew Ng](notes/notes.pdf)'s notes

# License:
[MIT](License)
