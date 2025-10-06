import math

class Value:
    def __init__(self, data, children=(), op='') -> None:
        self.data = data
        self.grad = 0.0
        self._prev = children
        self._backward = lambda: None
        self._op = op
        self.label = ''

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def backward():
           self.grad += out.grad
           other.grad += out.grad
        out._backward = backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def backward():
           self.grad += out.grad * other.data
           other.grad += out.grad * self.data

        out._backward = backward

        return out

    def __pow__(self, n):
        assert isinstance(n, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**n , (self, ), '**')

        def backward():
           self.grad += n * self.data**(n-1) * out.grad

        out._backward = backward

        return out

    def tanh(self):
        x = self.data
        out = Value((math.exp(2*x)-1)/(math.exp(2*x)+1), (self,), "tanh")

        def backward():
           self.grad += (1 - out.data**2) * out.grad

        out._backward = backward
        return out

    def backward(self):
        visited = set()
        topo = []
        def build_nodes(child):
            if child in visited:
                return 
            visited.add(child)
            topo.append(child)
            for ch in child._prev:
                build_nodes(ch)
        build_nodes(self)

        self.grad = 1.0
        for v in topo:
            v._backward()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"
