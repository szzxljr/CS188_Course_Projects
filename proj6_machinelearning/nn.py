import numpy as np


def main():
    """
    This is sample code for linear regression, which demonstrates how to use the
    Graph class.

    Once you have answered Questions 2 and 3, you can run `python nn.py` to
    execute this code.
    """

    # This is our data, where x is a 4x2 matrix and y is a 4x1 matrix
    x = np.array([[0., 0.],
                  [0., 1.],
                  [1., 0.],
                  [1., 1.]])
    y = np.dot(x, np.array([[7.],
                            [8.]])) + 3

    # Let's construct a simple model to approximate a function from 2D
    # points to numbers, f(x) = x_0 * m_0 + x_1 * m_1 + b
    # Here m and b are variables (trainable parameters):
    m = Variable(2, 1)
    b = Variable(1)

    # We train our network using batch gradient descent on our data
    for iteration in range(10000):
        # At each iteration, we first calculate a loss that measures how
        # good our network is. The graph keeps track of all operations used
        graph = Graph([m, b])
        input_x = Input(graph, x)
        input_y = Input(graph, y)
        xm = MatrixMultiply(graph, input_x, m)
        xm_plus_b = MatrixVectorAdd(graph, xm, b)
        loss = SquareLoss(graph, xm_plus_b, input_y)
        # Then we use the graph to perform backprop and update our variables
        graph.backprop()
        graph.step(0.01)

    # After training, we should have recovered m=[[7],[8]] and b=[3]
    print("Final values are: {}".format([m.data[0, 0], m.data[1, 0], b.data[0]]))
    assert np.isclose(m.data[0, 0], 7)
    assert np.isclose(m.data[1, 0], 8)
    assert np.isclose(b.data[0], 3)
    print("Success!")


class Graph(object):
    """
    TODO: Question 3 - [Neural Network] Computation Graph

    A graph that keeps track of the computations performed by a neural network
    in order to implement back-propagation.

    Each evaluation of the neural network (during both training and test-time)
    will create a new Graph. The computation will add nodes to the graph, where
    each node is either a DataNode or a FunctionNode.

    A DataNode represents a trainable parameter or an input to the computation.
    A FunctionNode represents doing a computation based on two previous nodes in
    the graph.

    The Graph is responsible for keeping track of all nodes and the order they
    are added to the graph, for computing gradients using back-propagation, and
    for performing updates to the trainable parameters.

    For an example of how the Graph can be used, see the function `main` above.
    """

    def __init__(self, variables):
        """
        TODO: Question 3 - [Neural Network] Computation Graph

        Initializes a new computation graph.

        variables: a list of Variable objects that store the trainable parameters
            for the neural network.

        Hint: each Variable is also a node that needs to be added to the graph,
        so don't forget to call `self.add` on each of the variables.
        """
        "*** YOUR CODE HERE ***"
        self.variables = variables
        self.graph = []
        for var in variables:
            self.add(var)

    def get_nodes(self):
        """
        TODO: Question 3 - [Neural Network] Computation Graph

        Returns a list of all nodes that have been added to this Graph, in the
        order they were added. This list should include all of the Variable
        nodes that were passed to `Graph.__init__`.

        Returns: a list of nodes
        """
        "*** YOUR CODE HERE ***"
        return self.graph

    def get_inputs(self, node):
        """
        TODO: Question 3 - [Neural Network] Computation Graph

        Retrieves the inputs to a node in the graph. Assume the `node` has
        already been added to the graph.

        Returns: a list of numpy arrays

        Hint: every node has a `.get_parents()` method
        """
        "*** YOUR CODE HERE ***"
        inputs = []
        for parent in node.get_parents():
            inputs.append(self.get_output(parent))
        return inputs

    def get_output(self, node):
        """
        TODO: Question 3 - [Neural Network] Computation Graph

        Retrieves the output to a node in the graph. Assume the `node` has
        already been added to the graph.

        Returns: a numpy array or a scalar
        """
        "*** YOUR CODE HERE ***"
        return node.output

    def get_gradient(self, node):
        """
        TODO: Question 3 - [Neural Network] Computation Graph

        Retrieves the gradient for a node in the graph. Assume the `node` has
        already been added to the graph.

        If `Graph.backprop` has already been called, this should return the
        gradient of the loss with respect to the output of the node. If
        `Graph.backprop` has not been called, it should instead return a numpy
        array with correct shape to hold the gradient, but with all entries set
        to zero.

        Returns: a numpy array
        """
        "*** YOUR CODE HERE ***"
        return node.gradient

    def add(self, node):
        """
        TODO: Question 3 - [Neural Network] Computation Graph

        Adds a node to the graph.

        This method should calculate and remember the output of the node in the
        forwards pass (which can later be retrieved by calling `get_output`)
        We compute the output here because we only want to compute it once,
        whereas we may wish to call `get_output` multiple times.

        Additionally, this method should initialize an all-zero gradient
        accumulator for the node, with correct shape.
        """
        "*** YOUR CODE HERE ***"
        inputs = [self.get_output(parent) for parent in node.get_parents()]
        node.output = node.forward(inputs)
        node.gradient = np.zeros_like(node.output)
        self.graph.append(node)

    def backprop(self):
        """
        TODO: Question 3 - [Neural Network] Computation Graph

        Runs back-propagation. Assume that the very last node added to the graph
        represents the loss.

        After back-propagation completes, `get_gradient(node)` should return the
        gradient of the loss with respect to the `node`.

        Hint: the gradient of the loss with respect to itself is 1.0, and
        back-propagation should process nodes in the exact opposite of the order
        in which they were added to the graph.
        """
        loss_node = self.get_nodes()[-1]
        assert np.asarray(self.get_output(loss_node)).ndim == 0

        "*** YOUR CODE HERE ***"
        loss_node.gradient = 1.0
        for node in list(reversed(self.graph)):
            parent_grads = node.backward(self.get_inputs(node), node.gradient)
            parents = node.get_parents()
            for i in range(len(parents)):
                parents[i].gradient += parent_grads[i]
        # for node in list(reversed(self.graph)):
        #     node.get_parents().gradient += node.backward(self.get_inputs(node), node.gradient)

    def step(self, step_size):
        """
        TODO: Question 3 - [Neural Network] Computation Graph

        Updates the values of all variables based on computed gradients.
        Assume that `backprop()` has already been called, and that gradients
        have already been computed.

        Hint: each Variable has a `.data` attribute
        """
        "*** YOUR CODE HERE ***"
        for var in self.variables:
            var.data -= step_size * var.gradient


class DataNode(object):
    """
    DataNode is the parent class for Variable and Input nodes.

    Each DataNode must define a `.data` attribute, which represents the data
    stored at the node.
    """

    @staticmethod
    def get_parents():
        # A DataNode has no parent nodes, only a `.data` attribute
        return []

    def forward(self, inputs):
        # The forwards pass for a data node simply returns its data
        return self.data

    @staticmethod
    def backward(inputs, gradient):
        # A DataNode has no parents or inputs, so there are no gradients to
        # compute in the backwards pass
        return []


class Variable(DataNode):
    """
    A Variable stores parameters used in a neural network.

    Variables should be created once and then passed to all future Graph
    constructors. Use `.data` to access or modify the numpy array of parameters.
    """

    def __init__(self, *shape):
        """
        Initializes a Variable with a given shape.

        For example, Variable(5) will create 5-dimensional vector variable,
        while Variable(10, 10) will create a 10x10 matrix variable.

        The initial value of the variable before training starts can have a big
        effect on how long the network takes to train. The provided initializer
        works well across a wide range of applications.
        """
        assert shape
        limit = np.sqrt(3.0 / np.mean(shape))
        self.data = np.random.uniform(low=-limit, high=limit, size=shape)


class Input(DataNode):
    """
    An Input node packages a numpy array into a node in a computation graph.
    Use this node for inputs to your neural network.

    For trainable parameters, use Variable instead.
    """

    def __init__(self, graph, data):
        """
        Initializes a new Input and adds it to a graph.
        """
        assert isinstance(data, np.ndarray), "data must be a numpy array"
        assert data.dtype.kind == "f", "data must have floating-point entries"
        self.data = data
        graph.add(self)


class FunctionNode(object):
    """
    A FunctionNode represents a value that is computed based on other nodes in
    the graph. Each function must implement both a forward and backward pass.
    """

    def __init__(self, graph, *parents):
        self.parents = parents
        graph.add(self)

    def get_parents(self):
        return self.parents

    @staticmethod
    def forward(inputs):
        raise NotImplementedError

    @staticmethod
    def backward(inputs, gradient):
        raise NotImplementedError


class Add(FunctionNode):
    """
    TODO: Question 2 - [Neural Network] Nodes

    Adds two vectors or matrices, element-wise

    Inputs: [x, y]
        x may represent either a vector or a matrix
        y must have the same shape as x
    Output: x + y
    """

    @staticmethod
    def forward(inputs):
        "*** YOUR CODE HERE ***"
        return inputs[0] + inputs[1]

    @staticmethod
    def backward(inputs, gradient):
        "*** YOUR CODE HERE ***"
        return [gradient, gradient]


class MatrixMultiply(FunctionNode):
    """
    TODO: Question 2 - [Neural Network] Nodes

    Represents matrix multiplication.

    Inputs: [A, B]
        A represents a matrix of shape (n x m)
        B represents a matrix of shape (m x k)
    Output: a matrix of shape (n x k)
    """

    @staticmethod
    def forward(inputs):
        "*** YOUR CODE HERE ***"
        return inputs[0].dot(inputs[1])

    @staticmethod
    def backward(inputs, gradient):
        "*** YOUR CODE HERE ***"
        return [gradient.dot(inputs[1].T), inputs[0].T.dot(gradient)]


class MatrixVectorAdd(FunctionNode):
    """
    TODO: Question 2 - [Neural Network] Nodes

    Adds a vector to each row of a matrix.

    Inputs: [A, x]
        A represents a matrix of shape (n x m)
        x represents a vector (m)
    Output: a matrix of shape (n x m)
    """

    @staticmethod
    def forward(inputs):
        "*** YOUR CODE HERE ***"
        return inputs[0] + inputs[1]

    @staticmethod
    def backward(inputs, gradient):
        "*** YOUR CODE HERE ***"
        return [gradient, np.sum(gradient, axis=0)]


class ReLU(FunctionNode):
    """
    TODO: Question 2 - [Neural Network] Nodes

    An element-wise Rectified Linear Unit nonlinearity: max(x, 0).
    This nonlinearity replaces all negative entries in its input with zeros.

    Input: [x]
        x represents either a vector or matrix
    Output: same shape as x, with no negative entries
    """

    @staticmethod
    def forward(inputs):
        "*** YOUR CODE HERE ***"
        return np.maximum(0, inputs)[0]

    @staticmethod
    def backward(inputs, gradient):
        "*** YOUR CODE HERE ***"
        return [np.where(inputs[0] > 0, gradient, 0)]
        # return [np.where(np.maximum(0, inputs[0]) > 0, gradient, 0)]

class SquareLoss(FunctionNode):
    """
    TODO: Question 2 - [Neural Network] Nodes

    Inputs: [a, b]
        a represents a matrix of size (batch_size x dim)
        b must have the same shape as a
    Output: a number

    This node first computes 0.5 * (a[i,j] - b[i,j])**2 at all positions (i,j)
    in the inputs, which creates a (batch_size x dim) matrix. It then calculates
    and returns the mean of all elements in this matrix.
    """

    @staticmethod
    def forward(inputs):
        "*** YOUR CODE HERE ***"
        return np.mean(0.5 * np.power(inputs[0] - inputs[1], 2))

    @staticmethod
    def backward(inputs, gradient):
        "*** YOUR CODE HERE ***"
        back = gradient * (inputs[0]-inputs[1]) / inputs[0].size
        return [back, -back]


class SoftmaxLoss(FunctionNode):
    """
    A batched softmax loss, used for classification problems.

    IMPORTANT: do not swap the order of the inputs to this node!

    Inputs: [logits, labels]
        logits: a (batch_size x num_classes) matrix of scores, that is typically
            calculated based on previous layers. Each score can be an arbitrary
            real number.
        labels: a (batch_size x num_classes) matrix that encodes the correct
            labels for the examples. All entries must be non-negative and the
            sum of values along each row should be 1.
    Output: a number

    We have provided the complete implementation for your convenience.
    """

    @staticmethod
    def log_softmax(logits):
        log_probs = logits - np.max(logits, axis=1, keepdims=True)
        log_probs -= np.log(np.sum(np.exp(log_probs), axis=1, keepdims=True))
        return log_probs

    @staticmethod
    def forward(inputs):
        labels = inputs[1]
        assert np.all(labels >= 0), \
            "Labels input to SoftmaxLoss must be non-negative. (Did you pass the inputs in the right order?)"
        assert np.allclose(np.sum(labels, axis=1), np.ones(labels.shape[0])), \
            "Labels input to SoftmaxLoss do not sum to 1 along each row. (Did you pass the inputs in the right order?)"

        log_probs = SoftmaxLoss.log_softmax(inputs[0])
        return np.mean(-np.sum(inputs[1] * log_probs, axis=1))

    @staticmethod
    def backward(inputs, gradient):
        log_probs = SoftmaxLoss.log_softmax(inputs[0])
        return [
            gradient * (np.exp(log_probs) - inputs[1]) / inputs[0].shape[0],
            gradient * -log_probs / inputs[0].shape[0]
        ]


if __name__ == '__main__':
    main()
