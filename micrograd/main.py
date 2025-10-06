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

    
    while True:
        # forward pass
        outs = [mlp(x) for x in xs]
        loss = sum([(out - y)**2 for y, out in zip(ys, outs)])
        mlp.zerograd()
        loss.backward()

        print(loss.data)
        
        for p in mlp.parameters():
            p.data += -0.01 * p.grad

        if loss.data < 0.01:
            break

    print(outs)
