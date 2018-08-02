# clusterone-asl
ASL 30 classes

The objective of this project is to run a distributed tensorflow program on the Clusterone platform.

The aim of the classification task is to classify the 30 American Sign Language symbols to alphabet using the images provided. There are 87,000 images which I have equally divided into a test and a training set.

### Details
The given images are in the shape 200x200x3. I have converted these images into 50x50x3 for this problem(for faster computation). As the size of the image is now small, I have used a fairly shallow CNN to perform the classification. My CNN has 2 convolutional layers and one fully connected layer with a dropout layer.

I have then initialized a tensorflow graph as described in the blog https://clusterone.com/blog/2017/09/13/distributed-tensorflow-clusterone/
I then used the TensorflowMonitoredSession to run a distributed session in the graph.

### Running
The model runs for 2000 steps with a batchsize of 128. The test accuracy is around 80%.

