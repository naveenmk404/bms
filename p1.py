import numpy as np

class Neuron:
    def __init__(self,input_size,activation_function="sigmoid"):
        self.wieghts = np.random.rand(input_size)
        self.bias = np.random.rand()

        if (activation_function == "sigmoid"):
            self.activation_function = self.sigmoid
            
        elif(activation_function == "step"):
            self.activation_function = self.step
            
        else:
            raise ValueError("invlaid activation function")
        
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
    def step(self,x):
        return 1 if x>= 0 else 0
    
    def forward(self,inputs):
        weighted_sum = np.dot(inputs,self.wieghts)+self.bias
        return self.activation_function(weighted_sum)

input_size = 3
inputs = np.random.rand(input_size)

neuron = Neuron(input_size,activation_function="sigmoid")

output = neuron.forward(inputs)

print(f"input : {inputs}")
print(f"weights : {neuron.wieghts}")
print(f"bias : {neuron.bias}")
print(f"output : {output}")
