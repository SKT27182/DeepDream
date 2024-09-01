# DeepDream

- DeepDream is an experiment that visualizes the patterns learned by a neural network. It does so by forwarding an image through the network, then calculating the gradient of the image with respect to the activations of a particular layer. <b>The image is then modified to increase these activations</b>, enhancing the patterns seen by the network, and resulting in a dream-like image. This process was dubbed "Inceptionism" (a reference to [InceptionNet](https://arxiv.org/pdf/1409.4842.pdf), and the [movie](https://en.wikipedia.org/wiki/Inception) Inception).

- The code is based on the [TensorFlow tutorial](https://www.tensorflow.org/tutorials/generative/deepdream) and the [DeepDream]( https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html) blog post by Google.