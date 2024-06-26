import torch
from torch import FloatTensor, optim
from torch.autograd import Variable


MODE = 'no_grad'  # none, no_grad, or inference_mode

weights = [Variable(FloatTensor([i]), requires_grad=True) for i in (2, 5, 9, 7)]

# unpack the weights for nicer assignment
w1, w2, w3, w4 = weights

w2.requires_grad = False

optimizer = optim.SGD(weights, lr=0.01, momentum=0.9)
optimizer.zero_grad()

for a_values in (4, 8):
    # Define the leaf nodes
    a = Variable(FloatTensor([a_values]))

    b = w1 * a  # = 8

    print(MODE)
    if MODE == 'inference_mode':
        with torch.inference_mode():
            c = w2 * a  # = 20
        d = w3 * b + w4 * c  # = 72 + 140 = 212
        # d = w3 * b + w4 * torch.clone(c)  # = 72 + 140 = 212
    else:
        if MODE == 'no_grad':
            with torch.no_grad():
                c = w2 * a  # = 20
        else:
            c = w2 * a  # = 20
        d = w3 * b + w4 * c  # = 72 + 140 = 212

    L = (10 - d)  # = -202

    L.register_hook(lambda grad: print(grad))
    d.register_hook(lambda grad: print(grad))
    b.register_hook(lambda grad: print(grad))
    # c.register_hook(lambda grad: print(grad))
    b.register_hook(lambda grad: print(grad))

    L.backward()
    optimizer.step()

    for index, weight in enumerate(weights, start=1):
        print(f"Weight value of w{index}: {weight}")
        if index == 2:
            continue
        gradient, *_ = weight.grad.data
        print(f"Gradient of w{index} w.r.t to L: {gradient}")
