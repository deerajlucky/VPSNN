import nn_layers.vp_layers as vp
import matplotlib.pyplot as plt
import torch

def realline(m, params):
    dilation, translation = params
    t = torch.arange(-(m // 2), m // 2 + 1) if m % 2 else torch.arange(-(m / 2), m / 2)
    x = dilation * (t - translation * m / 2)
    return x

def test_vp_fun():
    import torch
    m = 101
    n = 5
    params = torch.tensor([0.1, 0.0])
    coeffs = torch.tensor([[3., 3., 2., 1., -1.]], dtype=torch.double)
    x = realline(m, params)
    Phi, dPhi, ind = vp.ada_hermite(m, n, params)
    Phi = Phi.float()
    coeffs = coeffs.float()
    signal = Phi @ coeffs.T  # Use .T for transpose if coeffs is 2D
    c = torch.zeros_like(coeffs)
    for i in range(n):
        c[:, :i+1] = coeffs[:, :i+1]
        aprx = Phi @ c.T
        ax = plt.subplot(n, 1, int(i+1))
        ax.plot(x, torch.squeeze(signal), 'b', label='signal')
        ax.plot(x, torch.squeeze(aprx), 'r--', label='approx')
    plt.show()

'''Checking the projection.'''
params = torch.tensor([0.1, 0.0])
coeffs = torch.tensor([3., 3., 2., 1., -1.], dtype=torch.double).unsqueeze(0)
m = 101
n = coeffs.size()[1]
#signal = test_vp_fun(m, n, params, coeffs)
