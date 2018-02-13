import torch as t
from torch.nn import SELU, ReLU
from utils import variable
import numpy as np


def get_q_target(icnn, next_state, rewards, gamma=1., max_steps_a=30):
    next_state_ = []
    good = dict()
    for i, s in enumerate(next_state):
        if not np.isnan(s.data.numpy()[0]):
            next_state_.append(s.resize(1, 50))
            good[i] = len(good)
    if len(good) == 0:
        return rewards.squeeze()
    else:
        next_state_ = t.cat(next_state_)
        max_action = variable(np.zeros((len(next_state_), 2)), requires_grad=True).float()
        prev_action = variable(np.zeros((len(next_state_), 2)), requires_grad=True).float()
        input_param = t.nn.Parameter(max_action.data)
        optimizer_for_a = t.optim.Adam([input_param], lr=1.)
        for k in range(max_steps_a):
            max_action, input_param, optimizer_for_a = gradient_step_action(icnn, next_state_, max_action, input_param=input_param, optimizer=optimizer_for_a)
            if np.max(np.abs(prev_action.data.numpy() - max_action.data.numpy())) < 1e-5:
                break
            prev_action = max_action * 1

        pred = icnn.forward(next_state_, max_action)
        Qvalues = [pred[good[i]].float().resize(1, 1) if i in good else variable(np.zeros((1, 1))).float() for i in range(len(next_state))]
        max_prev_Q_value = t.cat(Qvalues, dim=0)
        Q_target = rewards.squeeze() + gamma * max_prev_Q_value.squeeze()
        return Q_target


class ICNN(t.nn.Module):
    """
    CONCAVE Q network
    """
    def __init__(self, n_layers, hidden_dim, activation=SELU()):
        super(ICNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.activation = activation
        for l in range(n_layers - 1):
            setattr(self, 'u' + str(l), t.nn.Linear(hidden_dim, hidden_dim))

        for l in range(n_layers):
            output_dim = (l < n_layers - 1) * hidden_dim + (l == n_layers - 1) * 1
            setattr(self, 'z_u' + str(l), t.nn.Linear(hidden_dim, output_dim))
            setattr(self, 'z_au' + str(l), t.nn.Linear(hidden_dim, 2))
            setattr(self, 'z_au_' + str(l), t.nn.Linear(2, output_dim, bias=False))
            if l > 0:
                setattr(self, 'z_zu' + str(l), t.nn.Linear(hidden_dim, hidden_dim))
                setattr(self, 'z_zu_' + str(l), t.nn.Linear(hidden_dim, output_dim, bias=False))

    def forward(self, s, a):
        u = s
        for l in range(self.n_layers):
            if l < self.n_layers - 1:
                fc = getattr(self, 'u' + str(l))
                u = self.activation(fc(u))
            if l == 0:
                fc_u = getattr(self, 'z_u' + str(l))
                fc_au_ = getattr(self, 'z_au_' + str(l))
                fc_au = getattr(self, 'z_au' + str(l))
                z = self.activation(fc_u(u) + fc_au_(fc_au(u) * a))
            else:
                fc_u = getattr(self, 'z_u' + str(l))
                fc_au_ = getattr(self, 'z_au_' + str(l))
                fc_au = getattr(self, 'z_au' + str(l))
                fc_zu_ = getattr(self, 'z_zu_' + str(l))
                fc_zu = getattr(self, 'z_zu' + str(l))
                z = fc_u(u) + fc_au_(fc_au(u) * a) + fc_zu_(ReLU()(fc_zu(u)) * z)
                if l < self.n_layers - 1:
                    z = self.activation(z)
        return -z

    def make_cvx(self):
        """Make the neural network convex by absoluvaluing its W_zu weights"""
        for l in range(1, self.n_layers):
            self._modules['z_zu_' + str(l)].weight = t.nn.Parameter(data=t.abs(self._modules['z_zu_' + str(l)].weight.data))

    def proj(self):
        """If some weights became positive, set them to 0"""
        for l in range(1, self.n_layers):
            self._modules['z_zu_' + str(l)].weight = t.nn.Parameter(data=.5 * (t.abs(self._modules['z_zu_' + str(l)].weight.data) + self._modules['z_zu_' + str(l)].weight.data))


def freeze_weights(icnn, frozen):
    """
    Freeze (or unfreeze) the weights of the ICNN
    :param icnn:
    :param frozen: whether to freeze the weights or not
    :return:
    """
    for l in range(icnn.n_layers):
        if l < icnn.n_layers - 1:
            fc = getattr(icnn, 'u' + str(l))
            fc.weight.requires_grad = not frozen
        if l == 0:
            fc_u = getattr(icnn, 'z_u' + str(l))
            fc_au_ = getattr(icnn, 'z_au_' + str(l))
            fc_au = getattr(icnn, 'z_au' + str(l))
            fc_u.weight.requires_grad = not frozen
            fc_au_.weight.requires_grad = not frozen
            fc_au.weight.requires_grad = not frozen
        else:
            fc_u = getattr(icnn, 'z_u' + str(l))
            fc_au_ = getattr(icnn, 'z_au_' + str(l))
            fc_au = getattr(icnn, 'z_au' + str(l))
            fc_zu_ = getattr(icnn, 'z_zu_' + str(l))
            fc_zu = getattr(icnn, 'z_zu' + str(l))
            fc_u.weight.requires_grad = not frozen
            fc_au_.weight.requires_grad = not frozen
            fc_au.weight.requires_grad = not frozen
            fc_zu_.weight.requires_grad = not frozen
            fc_zu.weight.requires_grad = not frozen


def gradient_step_action(Q, s, a, input_param=None, optimizer=None):
    """
    Compute the gradients with respect to the action and update the action
    The first pass of this function defines the optimizer and the input param if it has not already been done

    :param Q: The Q network. WATCH OUT! IT SHOULD BE `-icnn` AND NOT `icnn`
    :param s: a torch Variable representing the state
    :param a: a torch Variable representing the actions. IT SHOULD HAVE `requires_grad=True`
    :param input_param: It is just the same thing as `a`, but wrapped into a pytorch Parameter (t.nn.Parameter(a))
    :param optimizer: the optimizer (by default use Adam)
    :returns the updated value of `a`, the updated value of `input_param` (a Parameter wrapping `a`), the optimizer

    Example use case:
    ```
        a = variable(np.zeros((1,2)), requires_grad=True).float()
        s = variable(np.zeros((1,50))).float()

        input_param = None
        optimizer = None
        for k in range(200):
            a, input_param, optimizer = gradient_step_action(icnn, s, a, input_param=input_param, optimizer=optimizer)
            print(a)
    ```
    """
    if input_param is None or optimizer is None:
        input_param = t.nn.Parameter(a.data)
        optimizer = t.optim.Adam([input_param])

    assert len(s) == len(a), 'There should be as many states as there are actions'
    batch_size = len(s)

    # erase previous gradients
    optimizer.zero_grad()

    # trick to get the gradients wrt `a`
    grad = {}
    def f(x):
        grad['a'] = x
    a.register_hook(f)

    # get output (we want to maximize Q, so minimize -Q (the optimizer minimizes by default))
    output = -Q(s, a)

    # compute gradients
    output.backward(t.FloatTensor(batch_size*[[1.]]))

    # use the gradients that was deceitfully obtained using the hook
    input_param.grad = grad['a']

    # update the action
    optimizer.step()

    # returns the new value of `a` (a pytorch variable), the same thing but wrapped in t.nn.Parameter, and the optimizer
    return variable(input_param.data, requires_grad=True), input_param, optimizer


def diff_params(current_params, params0):
    """Check if the parameters of the network have been updated"""
    return t.sum(t.cat([t.sum((x1-x2)**2).resize(1,1) for x1,x2 in zip(current_params, params0)], 0)).data.numpy()[0]
