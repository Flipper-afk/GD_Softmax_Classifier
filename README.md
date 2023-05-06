# GD_Softmax_Classifier
Create a little n-layer Neural Network (fully connected multilayer perceptron)

Example:
```
# Number of input features (feature space)
input_dim = 2

# Number of Classes (labels)
output_dim = 4

# Setup Network Layers and regulizer
network = [input_dim, 100, 100, output_dim]
regulizer = 1e-6

# Build Network Model
NN = Model(network=network, regulizer=regulizer)

```

Train the Model

```
# Setup learning rate
lr = [[0.1]]

# Gradient descent loop steps (single epoch)
gd_loop = 1_000

loss = NN.train(X_train,
         y_train,
         learning_rate=lr,
         gd_loop=gd_loop,
         output=100)
         
 ```
 
 Evalute and Predict:
 
 ```
 accurracy = NN.evaluate(X_test, y_test))
 
 class_id = NN.predict(X_test[:, 0])
 
 ```
