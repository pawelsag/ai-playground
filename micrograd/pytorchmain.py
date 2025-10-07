#!/usr/bin/python
import torch
import torch.nn as nn
import torch.optim as optim

class TinyModel(nn.Module):
    def __init__(self):
        super(TinyModel, self).__init__()
        self.linear1 = nn.Linear(3, 4)
        self.linear2 = nn.Linear(4, 4)
        self.linear3 = nn.Linear(4, 1)

    def forward(self, x):
        x = torch.tanh(self.linear1(x))
        x = torch.tanh(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        return x

if __name__ == "__main__":
    xs = torch.tensor([
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0]
    ])
    ys = torch.tensor([[1.0], [-1.0], [-1.0], [1.0]])

    mlp = TinyModel()
    lossFn = nn.MSELoss()

    for epoch in range(2000):
        out = mlp(xs)

        loss = lossFn(out, ys)

        mlp.zero_grad()
        loss.backward()

        # can be replaced with some optimizer
        for p in mlp.parameters():
            p.data += -0.01 * p.grad

        if epoch % 200 == 0:
            print(f"Epoch {epoch:4d} | loss = {loss.item():.6f}")

    print("\nFinal predictions:")
    print(mlp(xs).detach())

