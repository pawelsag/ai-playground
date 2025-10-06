from value import Value
import random

class Module:
    
    def zerograd(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron:
    def __init__(self, numberOfInputs):
        # number of weigths must be eugal to number of inputs
        self.w = [Value(random.uniform(-1,1)) for _ in range(numberOfInputs)]
        self.b = Value(0.0)
    def __call__(self, x):
        out = sum([xi*wi for xi,wi in zip(x, self.w)] + [self.b])
        return out.tanh()

    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, numberOfInputs, numberOfOutputs):
        self.neurons = [Neuron(numberOfInputs) for _ in range(numberOfOutputs)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs if len(outs) !=1 else outs[0]

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP(Module):

    def __init__(self, numberOfInputs, numberOfOutputs):
        # numberOfInputs is the number of neurons in first layer
        # numberofOuputs in the list of neurons in each later layer
        neuronsInEachLayer = [numberOfInputs] + numberOfOutputs
        self.layers = [Layer(neuronsInEachLayer[i],neuronsInEachLayer[i+1]) for i in range(len(neuronsInEachLayer)-1)]

    def __call__(self, x):
        # each iteration produces new outputs which are later treated as new input set of neurons
        # called layer
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]