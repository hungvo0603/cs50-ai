# CS50 AI - Introduction to Artificial Intelligence
## Lecture 5 - Neural Networks
### Project 5 - Traffic

When doing experiments on the model, there are some remarks about how the model performs with different training parameters:
- Increasing the number of convolutional layers and pooling layers (from 1 to 2 layers) helps increases the performance of the model in terms of loss and accuracy. However, the increase in the performance is not significant (0.9215 to 0.9328).
- Increasing the number and sizes of filters for convolutional layers significantly increases the performance of the model in terms of accuracy (~0.92 to 0.969). However, the computational time also increases about 1ms/step.
- Increasing the pool size results in a decrease in the performance of the model in terms of accuracy, but the computational time also decreases. This is because more features of the images are lost when going through the pooling layers.
- Adding more hidden layers and increasing the the size of those layers improve the performace of the model, but results in longer computational time.
- The dropout rate has to be large enough, about 0.5, to avoid overfitting.

Note that in the code, these lines of code are used to improve the performance of Tensorflow in M1 Macbook with ARM architecture.
```
from tensorflow.python.compiler.mlcompute import mlcompute

tf.config.run_functions_eagerly(False)
mlcompute.set_mlc_device(device_name='gpu')
```
