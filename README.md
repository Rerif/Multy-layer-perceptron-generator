# Multy-layer perceptron
Simple perceptron generator.

Use this if you need a simple perceptron without use special ML libraries.
This genarator based on numpy.

# Using

### Step I
Create the perceptron:
```python
from perceptron import Multy-layer perceptron as plol

per = plol([3, 5, 1], 0.1)
```
In this example we use 3 input neurons and 1 output neurons. And we need at least one hidden leyer with any positive integer number of neurons.


If you need more layers:


[2, 3, 4, 3, 2, 1] - 6 layers, 2 input neurons, 1 output neurons. In hidden layers we have respectively 3, 4, 3, 2, neurons.

### Step II
Prepare a data:

You need to bring you data to the numbers 0...1. Use the input data according to the number of input neurons and the output data according to the number of output neurons.

```python
data_in = [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.4, 0.5, 0.6]]
data_out = [0.4, 0.5, 0.7]
```

### Step III
Training perceptron:

```python
for epoch in range(10000):
    for i in range(len(data_in)):
        per.perceptron_training(data_in[i], data_out[i])
```

### Step IV(optional)
Save trained perceptron in pickle.

### Step V
Use trained perceptron:
```python
print(per.use_perceptron([0.3, 0.4, 0.5]))
