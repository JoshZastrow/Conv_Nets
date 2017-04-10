import numpy as np
from cs231n.layers import *
from cs231n.layer_utils import *


'''
TO DO: Fix addition of b in layers.py line 260. The b has a shape of 15 (dimensions and does not add to
the dot product of x and w)
'''

class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                             weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg


        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################

        scale1 = .010
        scale2 = .010

        self.params['W1'] = scale1 * np.random.randn(input_dim, hidden_dim)
        self.params['W2'] = scale2 * np.random.randn(hidden_dim, num_classes)
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['b2'] = np.zeros(num_classes)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
            scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
            names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################

        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']

        hidden, cache_hidden = affine_relu_forward(X, W1, b1)
        scores, cache_scores = affine_forward(hidden, W2, b2)

        # If y is None then we are in test mode so just return scores
        if y is None:
                return scores

        loss, grads = 0, {}

        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################

        # Loss Function
        data_loss, dscores = softmax_loss(scores, y)
        regu_loss = 0.5 * self.reg * (np.sum(W1 ** 2) + np.sum(W2 ** 2))
        loss = data_loss + regu_loss

        da1, dW2, db2 = affine_backward(dscores, cache_scores)
        dX, dW1, db1 = affine_relu_backward(da1, cache_hidden)

        grads['W2'] = dW2
        grads['b2'] = db2
        grads['W1'] = dW1
        grads['b1'] = db1


        # Add Regularizatoin
        grads['W2'] += self.reg * W2
        grads['W1'] += self.reg * W1

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                             dropout=0, use_batchnorm=False, reg=0.0,
                             weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
            the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
            will make the dropout layers deteriminstic so we can gradient check the
            model.
        """

        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################

        layer_dimensions = hidden_dims + [num_classes]
        d_in = input_dim

        for i, d_out in enumerate(layer_dimensions):

            self.params['W{}'.format(i)] = weight_scale * np.random.randn(d_in, d_out)
            self.params['b{}'.format(i)] = np.zeros(d_out)

            if self.use_batchnorm and i < (self.num_layers - 1):
                self.params['gamma{}'.format(i)] = np.ones(d_out)
                self.params['beta{}'.format(i)] = np.zeros(d_out)
            d_in = d_out


        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.

        self.dropout_param = {}

        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.

        self.bn_params = []

        if self.use_batchnorm:
            self .bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.
        Input / output: Same as TwoLayerNet above.
        """

        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.dropout_param is not None:
            self.dropout_param['mode'] = mode

        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None

############################################################################
# TODO: Implement the forward pass for the fully-connected net, computing  #
# the class scores for X and storing them in the scores variable.          #
#                                                                          #
# When using dropout, you'll need to pass self.dropout_param to each       #
# dropout forward pass.                                                    #
#                                                                          #
# When using batch normalization, you'll need to pass self.bn_params[0] to #
# the forward pass for the first batch normalization layer, pass           #
# self.bn_params[1] to the forward pass for the second batch normalization #
# layer, etc.                                                              #
############################################################################

        cache_layer = {}
        inp = X.copy()

        for i in range(self.num_layers):

            if i < (self.num_layers - 1):  # leave out final layer
                # load parameters
                W = self.params['W{}'.format(i)]
                b = self.params['b{}'.format(i)]

                # Fully Connected layer
                inp, fc_cache = affine_forward(inp, W, b)
                cache_layer['fc{}'.format(i)] = fc_cache

                # Batch Norm layer
                if self.use_batchnorm:
                    g = self.params['gamma{}'.format(i)]
                    bt = self.params['beta{}'.format(i)]
                    inp, bn_cache = batchnorm_forward(inp, g, bt,
                                                      self.bn_params[i])
                    cache_layer['bn{}'.format(i)] = bn_cache

                # Drop Out layer
                if self.use_dropout:
                    inp, drop_cache = dropout_forward(inp, self.dropout_param)
                    cache_layer['drop{}'.format(i)] = drop_cache

                # Non Linearity Activation Function
                inp, relu_cache = relu_forward(inp)
                cache_layer['relu{}'.format(i)] = relu_cache

            else:  # Final Layer

                # load parameters
                W = self.params['W{}'.format(i)]
                b = self.params['b{}'.format(i)]
                scores, fc_cache = affine_forward(inp, W, b)
                cache_layer['fc{}'.format(i)] = fc_cache

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}

############################################################################
# TODO: Implement the backward pass for the fully-connected net. Store the #
# loss in the loss variable and gradients in the grads dictionary. Compute #
# data loss using softmax, and make sure that grads[k] holds the gradients #
# for self.params[k]. Don't forget to add L2 regularization!               #
#                                                                          #
# When using batch normalization, you don't need to regularize the scale   #
# and shift parameters.                                                    #
#                                                                          #
# NOTE: To ensure that your implementation matches ours and you pass the   #
# automated tests, make sure that your L2 regularization includes a factor #
# of 0.5 to simplify the expression for the gradient.                      #
############################################################################
        loss, dout = softmax_loss(scores, y)

        for i in reversed(range(self.num_layers)):

            # Retrieve the weights for reg loss
            W = self.params['W{}'.format(i)]
            loss += 0.5 * self.reg * np.sum(W ** 2)

            if i == self.num_layers - 1:  # Last Layer
                dx, dw, db = affine_backward(
                    dout, cache_layer['fc{}'.format(i)])

            else:  # Relu, dropout, batchNorm, fully connected backprop
                dx = relu_backward(
                    dout, cache_layer['relu{}'.format(i)])

                if self.use_dropout:
                    dx = dropout_backward(
                        dx, cache_layer['drop{}'.format(i)])

                if self.use_batchnorm:
                    dx, dgamma, dbeta = batchnorm_backward(
                        dx, cache_layer['bn{}'.format(i)])

                dx, dw, db = affine_backward(
                    dx, cache_layer['fc{}'.format(i)])

            # Update grads
            grads['W{}'.format(i)] = dw + self.reg * W
            grads['b{}'.format(i)] = db

            # BatchNorm grads
            if self.use_batchnorm and i < self.num_layers - 1:
                grads['gamma{}'.format(i)] = dgamma
                grads['beta{}'.format(i)] = dbeta

            # New upstream gradient
            dout = dx

        return loss, grads
