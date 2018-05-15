# A custom autograder for this project

################################################################################
# A mini-framework for autograding
################################################################################

import optparse
import pickle
import random
import sys
import traceback


class WritableNull:
    def write(self, string):
        pass

    def flush(self):
        pass


class Tracker(object):
    def __init__(self, questions, maxes, prereqs, mute_output):
        self.questions = questions
        self.maxes = maxes
        self.prereqs = prereqs

        self.points = {q: 0 for q in self.questions}

        self.current_question = None

        self.current_test = None
        self.points_at_test_start = None
        self.possible_points_remaining = None

        self.mute_output = mute_output
        self.original_stdout = None
        self.muted = False

    def mute(self):
        if self.muted:
            return

        self.muted = True
        self.original_stdout = sys.stdout
        sys.stdout = WritableNull()

    def unmute(self):
        if not self.muted:
            return

        self.muted = False
        sys.stdout = self.original_stdout

    def begin_q(self, q):
        assert q in self.questions
        text = 'Question {}'.format(q)
        print('\n' + text)
        print('=' * len(text))

        for prereq in sorted(self.prereqs[q]):
            if self.points[prereq] < self.maxes[prereq]:
                print("""*** NOTE: Make sure to complete Question {} before working on Question {},
                *** because Question {} builds upon your answer for Question {}.
                """.format(prereq, q, q, prereq))
                return False

        self.current_question = q
        self.possible_points_remaining = self.maxes[q]
        return True

    def begin_test(self, test_name):
        self.current_test = test_name
        self.points_at_test_start = self.points[self.current_question]
        print("*** {}) {}".format(self.current_question, self.current_test))
        if self.mute_output:
            self.mute()

    def end_test(self, pts):
        if self.mute_output:
            self.unmute()
        self.possible_points_remaining -= pts
        if self.points[self.current_question] == self.points_at_test_start + pts:
            print("*** PASS: {}".format(self.current_test))
        elif self.points[self.current_question] == self.points_at_test_start:
            print("*** FAIL")

        self.current_test = None
        self.points_at_test_start = None

    def end_q(self):
        assert self.current_question is not None
        assert self.possible_points_remaining == 0
        print('\n### Question {}: {}/{} ###'.format(
            self.current_question,
            self.points[self.current_question],
            self.maxes[self.current_question]))

        self.current_question = None
        self.possible_points_remaining = None

    def finalize(self):
        import time
        print('\nFinished at %d:%02d:%02d' % time.localtime()[3:6])
        print("\nProvisional grades\n==================")

        for q in self.questions:
            print('Question %s: %d/%d' % (q, self.points[q], self.maxes[q]))
        print('------------------')
        print('Total: %d/%d' % (sum(self.points.values()),
                                sum([self.maxes[q] for q in self.questions])))

        print("""Your grades are NOT yet registered.  To register your grades, make sure
        to follow your instructor's guidelines to receive credit on your project.
        """)

    def add_points(self, pts):
        self.points[self.current_question] += pts


TESTS = []
PREREQS = {}


def add_prereq(q, pre):
    if isinstance(pre, str):
        pre = [pre]

    if q not in PREREQS:
        PREREQS[q] = set()
    PREREQS[q] |= set(pre)


def test(q, points):
    def deco(fn):
        TESTS.append((q, points, fn))
        return fn

    return deco


def parse_options(argv):
    parser = optparse.OptionParser(description='Run public tests on student code')
    parser.set_defaults(
        edx_output=False,
        gs_output=False,
        no_graphics=False,
        mute_output=False,
        check_dependencies=False,
    )
    parser.add_option('--edx-output',
                      dest='edx_output',
                      action='store_true',
                      help='Ignored, present for compatibility only')
    parser.add_option('--gradescope-output',
                      dest='gs_output',
                      action='store_true',
                      help='Ignored, present for compatibility only')
    parser.add_option('--question', '-q',
                      dest='grade_question',
                      default=None,
                      help='Grade only one question (e.g. `-q q1`)')
    parser.add_option('--no-graphics',
                      dest='no_graphics',
                      action='store_true',
                      help='Do not display graphics (visualizing your implementation is highly recommended for debugging).')
    parser.add_option('--mute',
                      dest='mute_output',
                      action='store_true',
                      help='Mute output from executing tests')
    parser.add_option('--check-dependencies',
                      dest='check_dependencies',
                      action='store_true',
                      help='check that numpy and matplotlib are installed')
    (options, args) = parser.parse_args(argv)
    return options


def main():
    options = parse_options(sys.argv)
    if options.check_dependencies:
        check_dependencies()
        return

    if options.no_graphics:
        disable_graphics()

    questions = set()
    maxes = {}
    for q, points, fn in TESTS:
        questions.add(q)
        maxes[q] = maxes.get(q, 0) + points
        if q not in PREREQS:
            PREREQS[q] = set()

    questions = list(sorted(questions))
    if options.grade_question:
        if options.grade_question not in questions:
            print("ERROR: question {} does not exist".format(options.grade_question))
            sys.exit(1)
        else:
            questions = [options.grade_question]
            PREREQS[options.grade_question] = set()

    tracker = Tracker(questions, maxes, PREREQS, options.mute_output)
    for q in questions:
        started = tracker.begin_q(q)
        if not started:
            continue

        for testq, points, fn in TESTS:
            if testq != q:
                continue
            tracker.begin_test(fn.__name__)
            try:
                fn(tracker)
            except KeyboardInterrupt:
                tracker.unmute()
                print("\n\nCaught KeyboardInterrupt: aborting autograder")
                tracker.finalize()
                print("\n[autograder was interrupted before finishing]")
                sys.exit(1)
            except:
                tracker.unmute()
                print(traceback.format_exc())
            tracker.end_test(points)
        tracker.end_q()
    tracker.finalize()


################################################################################
# Tests begin here
################################################################################

import numpy as np
import matplotlib
import contextlib
np.set_printoptions(precision=6)


def check_dependencies():
    import matplotlib.pyplot as plt
    import time
    fig, ax = plt.subplots(1, 1)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    line, = ax.plot([], [], color="black")
    plt.show(block=False)

    for t in range(400):
        angle = t * 0.05
        x = np.sin(angle)
        y = np.cos(angle)
        line.set_data([x, -x], [y, -y])
        fig.canvas.draw_idle()
        fig.canvas.start_event_loop(1e-3)


def disable_graphics():
    import backend
    backend.use_graphics = False


@contextlib.contextmanager
def no_graphics():
    import backend
    old_use_graphics = backend.use_graphics
    backend.use_graphics = False
    yield
    backend.use_graphics = old_use_graphics


@test('q1', points=3)
def check_perceptron(tracker):
    import perceptron, backend

    print("Sanity checking perceptron...")
    np_random = np.random.RandomState(0)
    with no_graphics():
        # Check that the perceptron weights are initialized to a zero vector of `dimensions` entries.
        for _ in range(10):
            dimensions = np_random.randint(1, 10)
            p = perceptron.Perceptron(dimensions)
            p_weights = p.get_weights()
            assert p_weights is not None, \
                "Perceptron.get_weights() should return weights, not None"
            p_shape = np.asarray(p_weights).shape
            assert p_shape == (dimensions,), \
                "Perceptron weights had shape {}, expected {}".format(p_shape, (dimensions,))
            assert np.count_nonzero(p.get_weights()) == 0, "Perceptron weights should be initialized to zero."
        # Check that the untrained perceptron predicts 1 on any point
        for _ in range(10):
            dimensions = np_random.randint(1, 10)
            p = perceptron.Perceptron(dimensions)
            point = np_random.uniform(-10, 10, dimensions)
            pred = p.predict(point)
            assert pred == 1, "Untrained perceptron should predict 1 instead of {} for {}.".format(pred, point)
        # Check that a correctly classified point does not update the perceptron.
        for _ in range(10):
            dimensions = np_random.randint(1, 10)
            p = perceptron.Perceptron(dimensions)
            point = np_random.uniform(-10, 10, dimensions)
            old_weights = p.get_weights().copy()
            # All points should be classified as 1 at this point
            updated = p.update(point, 1)
            assert updated is not None, "Perceptron.update should return True or False, not None"
            assert not updated, "Updating with a correctly classified point ({}) should return `False`.".format(point)
            new_weights = p.get_weights()
            assert np.allclose(new_weights, old_weights), \
                "Updating with a correctly classified point ({}) should not change weights from {} to {}".format(point,
                                                                                                                 new_weights,
                                                                                                                 old_weights)
        # Check that the perceptron weight updates are correct
        xs = np.array([[0., -1., 1., 1.8],
                       [-1., 0., -2., 2.6]]).T
        ys = np.array([-1, -1, -1, -1], dtype=int)
        expected_weights = np.array([[0., 1., 1., -0.8],
                                     [1., 1., 1., -1.6]]).T
        expected_returns = [True, True, False, True]
        p = perceptron.Perceptron(2)
        for i in range(xs.shape[0]):
            old_weights = p.get_weights().copy()
            res = p.update(xs[i], ys[i])
            assert res == expected_returns[i], \
                """Perceptron.update returned {}. Expected: {}
    Old weights: {}
    x: {}
    y: {}""".format(res, expected_returns[i], old_weights, xs[i], ys[i])
            assert np.allclose(p.get_weights(), expected_weights[i]), \
                """Perceptron weights are {}. Expected: {}
    Old weights: {}
    x: {}
    y: {}""".format(p.get_weights(), expected_weights[i], old_weights, xs[i], ys[i])

    print("Sanity checking complete. Now training perceptron")
    p = perceptron.Perceptron(3)
    p.train()
    backend.maybe_sleep_and_close(1)

    stats = backend.get_stats(p)
    if stats is None:
        print("Your perceptron never trained for a full epoch!")
        return

    if stats["accuracy"] < 1.0:
        print("The weights learned by your perceptron correctly classified {:.2%} of training examples".format(
            stats['accuracy']))
        print("To receive points for this question, your perceptron must converge to 100% accuracy")
    else:
        tracker.add_points(3)


def sanity_test_node(cls, *inputs):
    inputs_copy = [np.copy(x) for x in inputs]

    output = cls.forward(inputs)
    name = cls.__name__
    assert output is not None, "{}.forward returned None".format(name)
    assert isinstance(output, np.ndarray) or np.isscalar(output), "{}.forward returned wrong type".format(name)
    if np.isscalar(output) or (isinstance(output, np.ndarray) and output.ndim == 0):
        grad = 1.0
    else:
        grad = np.ones_like(output)

    for a, b in zip(inputs, inputs_copy):
        assert np.allclose(a, b), "{}.forward appears to have modified an input".format(name)

    output_grad = cls.backward(inputs, grad)
    assert output_grad is not None, "{}.backward returned None".format(name)
    assert isinstance(output_grad, list), "{}.backward must return a list of gradients".format(name)
    assert len(output_grad) == len(inputs), "{}.backward returned {} elements, not {}".format(
        name, len(output_grad), len(inputs))
    for i, el in enumerate(output_grad):
        assert isinstance(el,
                          np.ndarray), "The gradient from {}.backward with respect to input {} is not a numpy array".format(
            name, i)

    for a, b in zip(inputs, inputs_copy):
        assert np.allclose(a, b), "{}.backward appears to have modified an input".format(name)
    assert np.allclose(grad, np.ones_like(grad)), "{}.backward appears to have modified the gradient array".format(name)


def numerical_test_node(cls, inputs, grad, output, grads_wrt_inputs):
    name = cls.__name__

    student_output = cls.forward(inputs)

    # Note that scalars may not have a .shape
    if np.shape(student_output) != np.shape(output):
        print("{}.forward returned output with incorrect shape".format(name))
        print("    Input shapes: {}".format(
            ";  ".join([str(x.shape) for x in inputs])))
        print("    Your output shape: {} expected: {}".format(
            np.shape(student_output), np.shape(output)))
        return False

    if not np.allclose(student_output, output):
        print("{}.forward did not return expected result".format(name))
        for i, inp in enumerate(inputs):
            print("Input {}:".format(i))
            print(inp)
        print("Output:")
        print(student_output)
        print("Expected:")
        print(output)
        return False

    student_grads_wrt_inputs = cls.backward(inputs, grad)

    for gi, (g, student_g) in enumerate(
            zip(grads_wrt_inputs, student_grads_wrt_inputs)):
        if np.shape(g) != np.shape(student_g) or not np.allclose(g, student_g):
            print("{}.backward did not calculate correct gradient with respect to input {}".format(
                name, gi))
            for i, inp in enumerate(inputs):
                print("Input {}:".format(i))
                print(inp)
            print("Second argument to backward:")
            print(grad)
            print("Your gradient with respect to input {}:".format(gi))
            print(student_g)
            print("Expected gradient with respect to input {}:".format(gi))
            print(g)
            return False

    return True


def create_node_tests():
    np_random = np.random.RandomState(0)

    vals = []
    for _ in range(10):
        n = np_random.randint(1, 10)
        m = np_random.randint(1, 10)
        u = np_random.uniform(-10, 10, m)
        v = np_random.uniform(-10, 10, u.shape)
        U = u[:, np.newaxis]
        V = v[:, np.newaxis]
        A = np_random.uniform(-10, 10, (n, m))
        B = np_random.uniform(-10, 10, A.shape)
        vals.append((u, v, U, V, A, B, A.T))
    u, v, U, V, A, B, A_T = zip(*vals)

    test_dict = {}
    with open('node_checks.pkl', 'rb') as f:
        tests = pickle.load(f)

    for name, inputs, grad, output, grads_wrt_inputs in tests:
        # Convert info from lists to numpy arrays
        inputs = [np.asarray(x, dtype=float) for x in inputs]
        grad = np.asarray(grad, dtype=float)
        output = np.asarray(output, dtype=float)
        grads_wrt_inputs = [np.asarray(x, dtype=float) for x in grads_wrt_inputs]
        info = (inputs, grad, output, grads_wrt_inputs)

        if name not in test_dict:
            test_dict[name] = []

        test_dict[name].append(info)

    def create_test(node_name, values, values2=None, points=1):
        @test('q2', points=points)
        def the_test(tracker):
            import nn
            cls = getattr(nn, node_name)
            for inputs in zip(*values):
                sanity_test_node(cls, *inputs)
            if values2 is not None:
                for inputs in zip(*values2):
                    sanity_test_node(cls, *inputs)
            correct = True
            for test in test_dict.get(node_name, []):
                correct = correct and numerical_test_node(cls, *test)
            if correct:
                tracker.add_points(points)

        the_test.__name__ = "check_{}".format(node_name)

    create_test('Add', (u, v), (A, B), points=1)
    create_test('MatrixMultiply', (A_T, B), points=2)
    create_test('MatrixVectorAdd', (A, v), points=2)
    create_test('ReLU', (u,), (A,), points=1)
    create_test('SquareLoss', (U, V), (A, B), points=1)


create_node_tests()

add_prereq('q3', ['q2'])


@test('q3', points=2)
def check_graph_basic(tracker):
    # First test with a basic graph. These tests are designed to pass even with
    # a broken gradient accumulator, so people can get started somewhere.
    import nn

    v1 = nn.Variable(1, 5)
    v1_data = np.ones_like(v1.data)
    v1.data = v1_data.copy()

    v2 = nn.Variable(1, 5)
    v2_data = np.ones_like(v2.data) / 5.0
    v2.data = v2_data.copy()

    graph = nn.Graph([v1, v2])

    g_nodes = graph.get_nodes()
    assert g_nodes is not None, "Graph.get_nodes returned None"
    assert g_nodes == [v1, v2], "Graph.get_nodes on newly-constructed graph did not return the variables"
    assert graph.get_inputs(v1) is not None, "Graph.get_inputs returned None"
    assert graph.get_inputs(v2) is not None, "Graph.get_inputs returned None"
    assert graph.get_inputs(v1) == [], "Graph.get_inputs should return no inputs for a Variable node"
    assert graph.get_inputs(v2) == [], "Graph.get_inputs should return no inputs for a Variable node"

    assert graph.get_output(v1) is not None, "Graph.get_output returned None"
    assert graph.get_output(v2) is not None, "Graph.get_output returned None"
    assert np.allclose(graph.get_output(v1), v1_data), "Graph.get_output for a Variable should be its data ({})," \
                                                       " returned {}".format(v1_data, graph.get_output(v1))
    assert np.allclose(graph.get_output(v2), v2_data), "Graph.get_output for a Variable should be its data ({})," \
                                                       " returned {}".format(v2_data, graph.get_output(v2))

    loss = nn.SoftmaxLoss(graph, v1, v2)
    assert graph.get_nodes() == [v1, v2, loss], \
        "Not all nodes are present after adding a node"

    loss_inputs = graph.get_inputs(loss)
    loss_inputs_list = []
    try:
        loss_inputs_list = list(loss_inputs)
    except:
        pass
    assert len(loss_inputs_list) == 2, \
        "Graph.get_inputs for SoftmaxLoss node returned {}. Expected: a length-2 list.".format(loss_inputs)
    assert np.allclose(v1.data, v1_data), \
        "Graph appears to have modified a Variable's data, even though step() has never been called"
    assert np.allclose(v2.data, v2_data), \
        "Graph appears to have modified a Variable's data, even though step() has never been called"
    for loss_input, data in zip(loss_inputs, [v1_data, v2_data]):
        assert (isinstance(loss_input, np.ndarray)
                and np.allclose(loss_input, data)), \
            "Graph.get_inputs returned wrong inputs for a SoftmaxLoss node"

    expected_loss = 1.60943791243
    numerical_loss = graph.get_output(loss)
    assert numerical_loss is not None, "Graph.get_output returned None"
    try:
        numerical_loss_float = float(numerical_loss)
    except:
        assert False, \
            "Graph.get_output for SoftmaxLoss returned {}. Expected: a number".format(numerical_loss)
    assert np.isclose(numerical_loss_float, expected_loss), \
        "Graph.get_output for SoftmaxLoss was {}. Expected: {}".format(numerical_loss, expected_loss)

    graph.backprop()

    loss_grad = graph.get_gradient(loss)
    try:
        loss_grad_float = float(loss_grad)
    except:
        assert False, \
            "Graph.get_gradient for the loss node returned {}. Expected: 1.0".format(loss_grad)
    assert np.isclose(loss_grad_float, 1.0), \
        "Graph.get_gradient for the loss node returned {}. Expected: 1.0".format(loss_grad)
    assert np.asarray(loss_grad).dtype.kind == 'f', \
        "Graph.get_gradient for the loss node must return a floating point number. (Did you return an integer?)".format(
            loss_grad, type(loss_grad))

    v1_grad = graph.get_gradient(v1)
    assert v1_grad is not None, "Graph.get_gradient returned None"
    assert v1_grad.shape == v1.data.shape, \
        "Graph.get_gradient returned gradient of wrong shape"

    v2_grad = graph.get_gradient(v2)
    assert v2_grad is not None, "Graph.get_gradient returned None"
    assert v2_grad.shape == v2.data.shape, \
        "Graph.get_gradient returned gradient of wrong shape"

    assert np.allclose(v1_grad, np.zeros_like(v1_grad)), "Incorrect gradient after running" \
                                                         " Graph.backprop()\nStudent returned:\n{}\n" \
                                                         "Expected:\n{}".format(v1_grad, np.zeros_like(v1_grad))
    assert np.allclose(v2_grad, np.ones_like(v2_grad) * expected_loss), "Incorrect gradient after running" \
                                                                        " Graph.backprop()\nStudent returned:\n{}\n" \
                                                                        "Expected:\n{}".format(v2_grad,
                                                                                               np.ones_like(v2_grad)
                                                                                               * expected_loss)
    assert np.allclose(v1.data, v1_data), \
        "Graph appears to have modified a Variable's data, even though step() has never been called"
    assert np.allclose(v2.data, v2_data), \
        "Graph appears to have modified a Variable's data, even though step() has never been called"
    graph.step(1.0)
    assert np.allclose(v1.data - v1_data, np.zeros_like(v1_grad)), \
        "Incorrect parameter update after running Graph.step()"
    assert np.allclose(v2.data - v2_data,
                       np.ones_like(v2_grad) * -expected_loss), \
        "Incorrect parameter update after running Graph.step()"

    tracker.add_points(2)


@test('q3', points=3)
def check_graph_accumulator(tracker):
    # A more thorough test that now requires gradient accumulators to be working
    import nn

    v1 = nn.Variable(1, 5)
    v1_data = np.ones_like(v1.data) / 10
    v1.data = v1_data
    graph = nn.Graph([v1])
    adder = nn.Add(graph, v1, v1)
    assert graph.get_nodes() == [v1, adder], \
        "Not all nodes are present after adding a node."
    assert graph.get_inputs(v1) == [], \
        "Graph.get_inputs should return no inputs for a Variable node"
    assert np.allclose(graph.get_output(v1), v1_data), \
        "Graph.get_output for a Variable should be its data:\n{}\n" \
        "Student returned:\n{}".format(v1_data, graph.get_output(v1))
    expected = [graph.get_output(v1)] * 2
    student = graph.get_inputs(adder)
    for a, b in zip(student, expected):
        assert np.allclose(a, b), "Graph.get_inputs returned incorrect value for an Add node\nStudent returned:\n{}\n" \
                                  "Expected:\n{}".format(a, b)
    assert np.allclose(graph.get_output(adder), 2 * graph.get_output(v1)), \
        "Graph.get_output returned incorrect value for an Add node\nStudent returned:\n{}\nExpected:\n{}"\
        .format(graph.get_output(adder), 2 * graph.get_output(v1))
    loss = nn.SoftmaxLoss(graph, adder, adder)
    for node in [v1, adder]:
        output_shape = graph.get_output(node).shape
        node_grad = graph.get_gradient(node)
        assert node_grad is not None, \
            "Graph.get_gradient returned None, instead of an all-zero value"
        assert np.shape(node_grad) == output_shape, \
            "Graph.get_gradient returned gradient of wrong shape, {0}; expected, {1}".format(np.shape(node_grad),
                                                                                             output_shape)
        assert np.allclose(node_grad, np.zeros_like(node_grad)), "Graph.get_gradient should return all-zero values" \
                                                                 " before backprop is called, instead returned:\n{}"\
            .format(node_grad)

    expected_loss = 1.60943791243
    graph.backprop()
    v1_grad = graph.get_gradient(v1)
    assert np.allclose(v1_grad, np.ones_like(v1_grad) * expected_loss * 2), \
        "Incorrect gradient after running Graph.backprop().\nStudent returned:\n{}\nExpected:\n{}\nMake sure you are" \
        " correctly accumulating your gradients.".format(v1_grad, np.ones_like(v1_grad) * expected_loss * 2)
    tracker.add_points(3)


@test('q3', points=3)
def check_graph_linear_regression(tracker):
    # Runs the Graph sample code, and makes sure that the Graph and the nodes
    # work well together for linear regression.
    import nn

    # This is our data, where x is a 4x2 matrix and y is a 4x1 matrix
    x = np.array([[0., 0., 1., 1.],
                  [0., 1., 0., 1.]]).T
    y = np.dot(x, np.array([[7., 8.]]).T) + 3

    # Let's construct a simple model to approximate a function from 2D
    # points to numbers, f(x) = x_0 * m_0 + x_1 * m_1 + b
    # Here m and b are variables (trainable parameters):
    m = nn.Variable(2, 1)
    b = nn.Variable(1)

    # Instead of fixing a random seed, just set .data directly
    m.data[0, 0] = -1.
    m.data[1, 0] = -1.
    b.data[0] = -1.

    # We train our network using batch gradient descent on our data
    for iteration in range(500):
        # At each iteration, we first calculate a loss that measures how
        # good our network is. The graph keeps track of all operations used
        graph = nn.Graph([m, b])
        input_x = nn.Input(graph, x)
        input_y = nn.Input(graph, y)
        xm = nn.MatrixMultiply(graph, input_x, m)
        xm_plus_b = nn.MatrixVectorAdd(graph, xm, b)
        loss = nn.SquareLoss(graph, xm_plus_b, input_y)
        # Then we use the graph to perform backprop and update our variables
        graph.backprop()
        graph.step(1.0)

    # After training, we should have recovered m=[[7],[8]] and b=[3]
    actual_values = [m.data[0, 0], m.data[1, 0], b.data[0]]
    expected_values = [7, 8, 3]
    assert np.allclose(actual_values, expected_values), \
        "Linear regression sample code did not run correctly. Final parameters {}. Expected: {}".format(
            actual_values, expected_values)
    tracker.add_points(3)


add_prereq('q4', ['q2', 'q3'])


@test('q4', points=2)
def check_regression(tracker):
    import models, backend
    model = models.RegressionModel()
    assert model.get_data_and_monitor == backend.get_data_and_monitor_regression, "RegressionModel.get_data_and_monitor is not set correctly"
    assert model.learning_rate > 0, "RegressionModel.learning_rate is not set correctly"
    model.train()

    stats = backend.get_stats(model)
    loss_threshold = 0.02
    if stats['loss'] <= loss_threshold:
        tracker.add_points(2)
    else:
        print("Your final loss ({:f}) must be no more than {:.4f} to receive points for this question".format(
            stats['loss'], loss_threshold))


add_prereq('q5', ['q2', 'q3', 'q4'])


@test('q5', points=1)
def check_odd_regression(tracker):
    loss_threshold = 0.02
    asymmetry_threshold = 1e-8

    import models, backend
    model = models.OddRegressionModel()
    assert model.get_data_and_monitor == backend.get_data_and_monitor_regression, "OddRegressionModel.get_data_and_monitor is not set correctly"
    assert model.learning_rate > 0, "OddRegressionModel.learning_rate is not set correctly"

    x_vals = np.linspace(-2 * np.pi, 2 * np.pi, num=16)[:, np.newaxis]
    y_vals = model.run(x_vals)
    asymmetry_vals = np.abs(y_vals + y_vals[::-1])
    max_asymmetry = np.max(asymmetry_vals)
    max_asymmetry_x = float(x_vals[np.argmax(asymmetry_vals)])
    if max_asymmetry > asymmetry_threshold:
        print("You do not appear to be modelling an odd function.")
        print("Prior to training, you have abs(f(x) + f(-x)) = {} for x = {:.3f}.".format(
            max_asymmetry, max_asymmetry_x))
        print("An odd function has abs(f(x) + f(-x)) = 0 for all x")
        return

    f_0 = float(model.run(np.array([[0.]])))
    if np.abs(f_0) > asymmetry_threshold:
        print("Your OddRegressionModel does not satisfy f(0) = 0")
        return

    model.train()

    stats = backend.get_stats(model)
    full_points = True

    if stats['loss'] > loss_threshold:
        full_points = False
        print("Your final loss ({:f}) must be no more than {:.4f} to receive points for this question".format(
            stats['loss'], loss_threshold))

    if stats['max_asymmetry'] > asymmetry_threshold:
        full_points = False
        print("You do not appear to be modelling an odd function.")
        print("After training, you have abs(f(x) + f(-x)) = {} for x = {:.3f}.".format(
            stats['max_asymmetry'], stats['max_asymmetry_x']))
        print("An odd function has abs(f(x) + f(-x)) = 0 for all x")

    if full_points:
        tracker.add_points(1)


add_prereq('q6', ['q2', 'q3'])


@test('q6', points=1)
def check_digit_classification(tracker):
    import models, backend
    model = models.DigitClassificationModel()
    assert model.get_data_and_monitor == backend.get_data_and_monitor_digit_classification, "DigitClassificationModel.get_data_and_monitor is not set correctly"
    assert model.learning_rate > 0, "DigitClassificationModel.learning_rate is not set correctly"
    model.train()

    stats = backend.get_stats(model)
    accuracy_threshold = 0.97
    if stats['dev_accuracy'] >= accuracy_threshold:
        tracker.add_points(1)
    else:
        print(
            "Your final validation accuracy ({:%}) must be at least {:.0%} to receive points for this question".format(
                stats['dev_accuracy'], accuracy_threshold))


add_prereq('q7', ['q2', 'q3'])


@test('q7', points=1)
def check_rl(tracker):
    import models, backend

    num_trials = 6
    trials_satisfied = 0
    trials_satisfied_required = 3
    for trial_number in range(num_trials):
        model = models.DeepQModel()
        assert model.get_data_and_monitor == backend.get_data_and_monitor_rl, "DeepQModel.get_data_and_monitor is not set correctly"
        assert model.learning_rate > 0, "DeepQModel.learning_rate is not set correctly"
        model.train()

        stats = backend.get_stats(model)
        if stats['mean_reward'] >= stats['reward_threshold']:
            trials_satisfied += 1

        if trials_satisfied >= trials_satisfied_required:
            tracker.add_points(1)
            return
        else:
            trials_left = num_trials - (trial_number + 1)
            if trials_satisfied + trials_left < trials_satisfied_required:
                break

    print(
        "To receive credit for this question, your agent must receive a mean reward of at least {} on {} out of {} trials".format(
            stats['reward_threshold'], trials_satisfied_required, num_trials))


add_prereq('q8', ['q2', 'q3'])


@test('q8', points=2)
def check_lang_id(tracker):
    import models, backend
    model = models.LanguageIDModel()
    assert model.get_data_and_monitor == backend.get_data_and_monitor_lang_id, "LanguageIDModel.get_data_and_monitor is not set correctly"
    assert model.learning_rate > 0, "LanguageIDModel.learning_rate is not set correctly"
    model.train()

    stats = backend.get_stats(model)
    accuracy_threshold = 0.81
    if stats['dev_accuracy'] >= accuracy_threshold:
        tracker.add_points(2)
    else:
        print(
            "Your final validation accuracy ({:%}) must be at least {:.0%} to receive points for this question".format(
                stats['dev_accuracy'], accuracy_threshold))


if __name__ == '__main__':
    main()
