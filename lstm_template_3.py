"""
Minimal character-level LSTM model. Written by Ngoc Quan Pham
Code structure borrowed from the Vanilla RNN model from Andreij Karparthy @karparthy.
BSD License
"""
import numpy as np
from random import uniform
import sys



# Since numpy doesn't have a function for sigmoid
# We implement it manually here
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# The derivative of the sigmoid function
def dsigmoid(y):
    return y * (1 - y)

def tanh(x):
    return np.tanh(x)

# The derivative of the tanh function
def dtanh(x):
    return 1 - x * x

# The numerically stable softmax implementation
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# data I/O
# data = open('data/input.txt', 'r').read()  # should be simple plain text file
data = open('data/linux-kernel.txt', 'r').read()  # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = { ch: i for i, ch in enumerate(chars) }
ix_to_char = { i: ch for i, ch in enumerate(chars) }
std = 0.1

option = sys.argv[1]

# hyperparameters
emb_size = 4
hidden_size = 32  # size of hidden layer of neurons
seq_length = 64  # number of steps to unroll the RNN for
learning_rate = 5e-2
max_updates = 500000

concat_size = emb_size + hidden_size

# model parameters
# char embedding parameters
Wex = np.random.randn(emb_size, vocab_size) * std  # embedding layer

# LSTM parameters
Wf = np.random.randn(hidden_size, concat_size) * std  # forget gate
Wi = np.random.randn(hidden_size, concat_size) * std  # input gate
Wo = np.random.randn(hidden_size, concat_size) * std  # output gate
Wc = np.random.randn(hidden_size, concat_size) * std  # c term

bf = np.zeros((hidden_size, 1))  # forget bias
bi = np.zeros((hidden_size, 1))  # input bias
bo = np.zeros((hidden_size, 1))  # output bias
bc = np.zeros((hidden_size, 1))  # memory bias

# Output layer parameters
Why = np.random.randn(vocab_size, hidden_size) * 0.01  # hidden to output
by = np.zeros((vocab_size, 1))  # output bias

dWex, dWhy = np.zeros_like(Wex), np.zeros_like(Why)
dWf, dWi, dWc, dWo = np.zeros_like(Wf), np.zeros_like(Wi), np.zeros_like(Wc), np.zeros_like(Wo)
dbf, dbi, dbc, dbo = np.zeros_like(bf), np.zeros_like(bi), np.zeros_like(bc), np.zeros_like(bo)

def forward_step(x, h_prev, C_prev):
    assert x.shape == (hidden_size + emb_size, 1)
    assert h_prev.shape == (hidden_size, 1)
    assert C_prev.shape == (hidden_size, 1)

    # z = np.row_stack((h_prev, x))

    z = x
    f = sigmoid(np.dot(Wf, z) + bf)
    i = sigmoid(np.dot(Wi, z) + bi)
    C_bar = tanh(np.dot(Wc, z) + bc)

    C = f * C_prev + i * C_bar
    o = sigmoid(np.dot(Wo, z) + bo)
    h = o * tanh(C)

    y = np.dot(Why, h) + by
    p = np.exp(y) / np.sum(np.exp(y))

    return z, f, i, C_bar, C, o, h, y, p

def forward(inputs, targets, memory):
    """
    inputs,targets are both list of integers.
    hprev is Hx1 array of initial hidden state
    returns the loss, gradients on model parameters, and last hidden state
    """

    # The LSTM is different than the simple RNN that it has two memory cells
    # so here you need two different hidden layers
    hprev, cprev = memory

    # Here you should allocate some variables to store the activations during forward
    # One of them here is to store the hiddens and the cells
    hs, cs = { }, { }
    xs, wes, zs, f_s, i_s, C_bar_s, C_s, o_s, y_s, p_s = { }, { }, { }, { }, { }, { }, { }, { }, { }, { }

    hs[-1] = np.copy(hprev)
    cs[-1] = np.copy(cprev)

    loss = 0
    # forward pass
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size, 1))  # encode in 1-of-k representation
        xs[t][inputs[t]] = 1

        # convert word indices to word embeddings
        wes[t] = np.dot(Wex, xs[t])

        # LSTM cell operation
        # first concatenate the input and h
        # This step is irregular (to save the amount of matrix multiplication we have to do)
        # I will refer to this vector as [h X]
        zs[t] = np.row_stack((hs[t - 1], wes[t]))

        z, f, i, C_bar, C, o, h, y, p = forward_step(zs[t], hs[t - 1], cs[t - 1])
        f_s[t] = f
        i_s[t] = i
        C_bar_s[t] = C_bar
        cs[t] = C
        o_s[t] = o
        hs[t] = h
        y_s[t] = y
        p_s[t] = p
        loss_t = -np.log(p_s[t][targets[t], 0])
        loss += loss_t

        # YOUR IMPLEMENTATION should begin from here

        # compute the forget gate
        # f_gate = sigmoid (W_f \cdot [h X] + b_f)

        # compute the input gate
        # i_gate = sigmoid (W_i \cdot [h X] + b_i)

        # compute the candidate memory
        # \hat{c} = tanh (W_c \cdot [h X] + b_c])

        # new memory: applying forget gate on the previous memory
        # and then adding the input gate on the candidate memory
        # c_new = f_gate * prev_c + i_gate * \hat{c}

        # output gate
        # o_gate = sigmoid (Wo \cdot [h X] + b_o)

        # new hidden state for the LSTM
        # h = o_gate * tanh(c_new)

        # DONE LSTM
        # output layer - softmax and cross-entropy loss
        # unnormalized log probabilities for next chars

        # o = Why \cdot h + by

        # softmax for probabilities for next chars
        # p = softmax(o)

        # cross-entropy loss
        # cross entropy loss at time t:
        # create an one hot vector for the label y

        # and then cross-entropy (see the elman-rnn file for the hint)

    # define your activations
    memory = (hs[len(inputs) - 1], cs[len(inputs) - 1])
    activations = (xs, wes, zs, f_s, i_s, C_bar_s, C_s, o_s, y_s, p_s, hs, cs)

    return loss, activations, memory

def backward_step(target, dh_next, dC_next, C_prev, z, f, i, C_bar, C, o, h, y, p, x):
    global dWf, dWi, dWc, dWo, dWhy, dWex
    global dbf, dbi, dbc, dbo, dby

    assert z.shape == (emb_size + hidden_size, 1)
    assert y.shape == (vocab_size, 1)
    assert p.shape == (vocab_size, 1)

    for param in [dh_next, dC_next, C_prev, f, i, C_bar, C, o, h]:
        assert param.shape == (hidden_size, 1)

    dy = np.copy(p)
    dy[target] -= 1

    dWhy += np.dot(dy, h.T)
    dby += dy

    dh = np.dot(Why.T, dy)
    dh += dh_next
    do = dh * tanh(C)
    do = dsigmoid(o) * do
    dWo += np.dot(do, z.T)
    dbo += do

    dC = np.copy(dC_next)
    dC += dh * o * dtanh(tanh(C))
    dC_bar = dC * i
    dC_bar = dC_bar * dtanh(C_bar)
    dWc += np.dot(dC_bar, z.T)
    dbc += dC_bar

    di = dC * C_bar
    di = dsigmoid(i) * di
    dWi += np.dot(di, z.T)
    dbi += di

    df = dC * C_prev
    df = dsigmoid(f) * df
    dWf += np.dot(df, z.T)
    dbf += df

    dz = np.dot(Wf.T, df) \
         + np.dot(Wi.T, di) \
         + np.dot(Wc.T, dC_bar) \
         + np.dot(Wo.T, do)
    dh_prev = dz[:hidden_size, :]
    dC_prev = f * dC

    de = dz[hidden_size:hidden_size + emb_size:, :]
    # embedding backprop
    dWex += np.dot(de, x.T)

    return dh_prev, dC_prev

def backward(activations, clipping = True):
    # backward pass: compute gradients going backwards
    # Here we allocate memory for the gradients

    global dWf, dWi, dWc, dWo, dWhy, dWex
    global dbf, dbi, dbc, dbo, dby
    dWex, dWhy = np.zeros_like(Wex), np.zeros_like(Why)
    dby = np.zeros_like(by)
    dWf, dWi, dWc, dWo = np.zeros_like(Wf), np.zeros_like(Wi), np.zeros_like(Wc), np.zeros_like(Wo)
    dbf, dbi, dbc, dbo = np.zeros_like(bf), np.zeros_like(bi), np.zeros_like(bc), np.zeros_like(bo)

    (xs, wes, zs, f_s, i_s, C_bar_s, C_s, o_s, y_s, p_s, hs, cs) = activations

    # similar to the hidden states in the vanilla RNN
    # We need to initialize the gradients for these variables
    dh_next = np.zeros_like(hs[0])
    dC_next = np.zeros_like(cs[0])

    # back propagation through time starts here
    for t in reversed(range(len(inputs))):
        # IMPLEMENT YOUR BACKPROP HERE
        # refer to the file elman_rnn.py for more details
        dh_next, dC_next = backward_step(target = targets[t], dh_next = dh_next, dC_next = dC_next, C_prev = cs[t - 1],
                                         z = zs[t], f = f_s[t], i = i_s[t], C_bar = C_bar_s[t], C = cs[t], o = o_s[t],
                                         h = hs[t], y = y_s[t], p = p_s[t], x = xs[t])

    if clipping:
        # clip to mitigate exploding gradients
        for dparam in [dWex, dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWhy, dby]:
            np.clip(dparam, -5, 5, out = dparam)

    gradients = (dWex, dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWhy, dby)

    return gradients

def sample(memory, seed_ix, n):
    """
    sample a sequence of integers from the model
    h is memory state, seed_ix is seed letter for first time step
    """
    h, c = memory
    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1

    indexes = []

    for t in range(n):
        # IMPLEMENT THE FORWARD FUNCTION ONE MORE TIME HERE
        # BUT YOU DON"T NEED TO STORE THE ACTIVATIONS
        wes = np.dot(Wex, x)
        z = np.row_stack((h, wes))
        _, _, _, _, c, _, h, _, p = forward_step(z, h, c)
        idx = np.random.choice(range(vocab_size), p = p.ravel())
        x = np.zeros((vocab_size, 1))
        x[idx] = 1
        indexes.append(idx)

    return indexes

if option == 'train':

    n, p = 0, 0
    n_updates = 0

    # momentum variables for Adagrad
    mWex, mWhy = np.zeros_like(Wex), np.zeros_like(Why)
    mby = np.zeros_like(by)

    mWf, mWi, mWo, mWc = np.zeros_like(Wf), np.zeros_like(Wi), np.zeros_like(Wo), np.zeros_like(Wc)
    mbf, mbi, mbo, mbc = np.zeros_like(bf), np.zeros_like(bi), np.zeros_like(bo), np.zeros_like(bc)

    smooth_loss = -np.log(1.0 / vocab_size) * seq_length  # loss at iteration 0

    while True:
        # prepare inputs (we're sweeping from left to right in steps seq_length long)
        if p + seq_length + 1 >= len(data) or n == 0:
            hprev = np.zeros((hidden_size, 1))  # reset RNN memory
            cprev = np.zeros((hidden_size, 1))
            p = 0  # go from start of data
        inputs = [char_to_ix[ch] for ch in data[p:p + seq_length]]
        targets = [char_to_ix[ch] for ch in data[p + 1:p + seq_length + 1]]

        # sample from the model now and then
        if n % 100 == 0:
            sample_ix = sample((hprev, cprev), inputs[0], 200)
            txt = ''.join(ix_to_char[ix] for ix in sample_ix)
            print('----\n %s \n----' % (txt,))

        # forward seq_length characters through the net and fetch gradient
        loss, activations, memory = forward(inputs, targets, (hprev, cprev))
        gradients = backward(activations)

        hprev, cprev = memory
        dWex, dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWhy, dby = gradients
        smooth_loss = smooth_loss * 0.999 + loss * 0.001
        if n % 100 == 0:
            print('iter %d, loss: %f' % (n, smooth_loss))  # print progress

        # perform parameter update with Adagrad
        for param, dparam, mem in zip([Wf, Wi, Wo, Wc, bf, bi, bo, bc, Wex, Why, by],
                                      [dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWex, dWhy, dby],
                                      [mWf, mWi, mWo, mWc, mbf, mbi, mbo, mbc, mWex, mWhy, mby]):
            mem += dparam * dparam
            param += -learning_rate * dparam / np.sqrt(mem + 1e-8)  # adagrad update

        p += seq_length  # move data pointer
        n += 1  # iteration counter
        n_updates += 1
        if n_updates >= max_updates:
            break

elif option == 'gradcheck':

    p = 0
    inputs = [char_to_ix[ch] for ch in data[p:p + seq_length]]
    targets = [char_to_ix[ch] for ch in data[p + 1:p + seq_length + 1]]

    delta = 0.001

    hprev = np.zeros((hidden_size, 1))
    cprev = np.zeros((hidden_size, 1))

    memory = (hprev, cprev)

    loss, activations, _ = forward(inputs, targets, memory)
    gradients = backward(activations, clipping = False)
    dWex, dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWhy, dby = gradients

    for weight, grad, name in zip([Wf, Wi, Wo, Wc, bf, bi, bo, bc, Wex, Why, by],
                                  [dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWex, dWhy, dby],
                                  ['Wf', 'Wi', 'Wo', 'Wc', 'bf', 'bi', 'bo', 'bc', 'Wex', 'Why', 'by']):

        str_ = ("Dimensions dont match between weight and gradient %s and %s." % (weight.shape, grad.shape))
        assert (weight.shape == grad.shape), str_

        print(name)
        for i in range(weight.size):

            # evaluate cost at [x + delta] and [x - delta]
            w = weight.flat[i]
            weight.flat[i] = w + delta
            loss_positive, _, _ = forward(inputs, targets, memory)
            weight.flat[i] = w - delta
            loss_negative, _, _ = forward(inputs, targets, memory)
            weight.flat[i] = w  # reset old value for this parameter

            grad_analytic = grad.flat[i]
            grad_numerical = (loss_positive - loss_negative) / (2 * delta)

            # compare the relative error between analytical and numerical gradients
            # rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic)
            rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic + 1e-9)

            if rel_error > 0.01:
                print('WARNING %f, %f => %e ' % (grad_numerical, grad_analytic, rel_error))