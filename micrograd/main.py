#!/usr/bin/python
from nn import MLP

if __name__ == "__main__":
    # inputs
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0,  1.0],
        [1.0, 1.0, -1.0]
    ]
    # expected output
    ys = [1.0, -1.0, -1.0 , 1.0]
    # mpl with 4 layers 
    # first layer with 3 inputs
    # 2,3 layer are hidden with 4 neourons each
    # 4'th layer is output with single value
    mlp = MLP(3, [4, 4, 1])

    
    for epoch in range(2000):
        # forward pass
        outs = [mlp(x) for x in xs]
        loss = sum([(out - y)**2 for y, out in zip(ys, outs)])

        # backward pass / evaluating grads``
        mlp.zerograd()
        loss.backward()

        # calculating new weigts
        for p in mlp.parameters():
            p.data += -0.01 * p.grad

        if epoch % 200 == 0:
            print(f"Epoch {epoch:4d} | loss = {loss.data:.6f}")

    print("\nFinal predictions:")
    print(outs)
