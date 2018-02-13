from queue import PriorityQueue
from utils import variable
import torch as t
import numpy as np
from icnn import get_q_target


def get_experience_buffer(icnn, transitions_train, max_steps_a=30, gamma=1.):
    q = PriorityQueue(maxsize=len(transitions_train))

    for i in range(0, len(transitions_train), 500):
        batch = transitions_train[i:i + 500]
        states = variable(np.concatenate([s.reshape((1, -1)) for s, a, r, s_ in batch])).float()
        actions = variable(np.concatenate([a.reshape((1, -1)) for s, a, r, s_ in batch])).float()
        rewards = variable(np.concatenate([r.reshape((1, -1)) for s, a, r, s_ in batch])).float().squeeze()
        next_state = variable(np.concatenate([s_.reshape((1, -1)) if s_ is not None else np.array([50 * [np.nan]]) for s, a, r, s_ in batch])).float().squeeze()

        Q_target = get_q_target(icnn, next_state, rewards, gamma=gamma, max_steps_a=max_steps_a)

        pred = icnn.forward(states, actions).squeeze()
        losses = ((pred - Q_target) ** 2).data.numpy()

        for j, loss in enumerate(losses):
            q.put((-loss, j + i))

    return q


def update_experience(q, loss, states_idx):
    if isinstance(loss, t.autograd.Variable):
        loss_ = loss.data.numpy()
    elif isinstance(loss, np.ndarray):
        loss_ = loss
    elif isinstance(loss, list):
        loss_ = np.array(loss)
    else:
        raise ValueError("loss type was : %s and should be either list, ndarray, or Variable" % str(type(loss)))
    if isinstance(states_idx, t.autograd.Variable):
        states_idx_ = states_idx.data.numpy()
    elif isinstance(states_idx, np.ndarray):
        states_idx_ = states_idx
    elif isinstance(states_idx, list):
        states_idx_ = np.array(states_idx)
    else:
        raise ValueError("states_idx type was : %s and should be either list, ndarray, or Variable" % str(type(states_idx)))

    assert loss_.shape[0] == states_idx_.shape[0]
    for l, s in zip(loss_, states_idx_):
        q.put((-l, s))


def sample(q, batch_size):
    return [q.get()[1] for _ in range(batch_size)]
