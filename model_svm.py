import numpy as np
from model_base import BaseModel, inval_check


class SVM_Model(BaseModel):
    """
        A model of a support vector machine for multiclass classification.

        Builds a hyperplane with the maximum distance from points in space,
        dividing the input data into 2 classes.

        To classify into 3 or more classes, you need to build several models
        and compare their results in the way of 'one against one' or
        'one against the rest' (One vs One, One vs Rest).
        At the moment, the One vs Rest method is implemented

        The Soft Margin Loss cost function is applied:
        https://towardsdatascience.com/svm-implementation-from-scratch-python-2db2fc52e5c2#66a2

        For the cost function to work, you need answers on a training sample
        serve in the form of: [-1, -1, 1, -1], where the unit position is the point class number.

        Parameters:
            Arbitrary:
                data_converter - function-converter for input data
                shift_column: bool - use a column of constants to act as additive
                            for folmula: wx_1 + wx_2 .... + wx_n + b = Y
                normalization: bool - use normalization for input values
                num_classes: int - number of classes in dataset
                learning_rate: float - how fast model will fit
                batch_size: int - size of mini-datasets to process while fitting.
                                  helps avoid overfitting
                epochs: int - count of gradient descent steps for whole dataset
                regularization: float - regularization strength for loss function

    """

    def __init__(self, custom_params=None) -> None:
        super().__init__(custom_params)
        must_have_params = ['shift_column', 'normalization',
                            'num_classes', 'learning_rate', 'batch_size', 'epochs',
                            'regularization']
        self.assert_have(must_have_params)

    def fit(self, data):
        y, x = self._splice_data(data)
        y = self._ovr_split(y)
        self.params = np.zeros_like(x.T @ y)
        num_samples = x.shape[0]

        learning_rate = getattr(self, 'learning_rate', 0.1)
        epochs = getattr(self, "epochs", 100)
        k = getattr(self, "regularization", 1)
        debug = getattr(self, "debug", False)

        for epoch in range(epochs):
            indexes = np.random.permutation(num_samples)
            y = y[indexes]
            x = x[indexes]
            for x_batch, y_batch in self._iter_batches(x, y):
                y_pred = self._forward(x_batch)

                self.params -= self._compute_gradient(
                    x_batch, y_pred, y_batch, learning_rate, k=k)

            loss = self.soft_margin_loss(self._forward(x), y)

            if hasattr(self, '_tick'):
                self._tick()

            if debug:
                print(f'-----epoch {epoch+1}/{epochs}-----  ', end='')
                print(f'Current cost: {loss.sum():.2f}')

        if debug:
            print(
                f'''params properties: mean={self.params.mean()},
                std={self.params.std()}, max={self.params.max()},
                min={self.params.min()}
                ''')

        return self

    @inval_check
    def hinge_loss(self, y_pred, y):
        """
            Applying the formula:
            `max(0, 1 - y_i*(w*x_i+b))` - for each value in the class

            inputs:
                y_pred - model's output
                y - training data answer

            output:
                np.array (shape = n_classes)
        """
        return np.vectorize(lambda x: x if x > 0 else 0)(1-y*y_pred).mean(axis=0)

    @inval_check
    def soft_margin_loss(self, y_pred, y, k=1):
        """
            Formula:
            J = k * ||w||^2 + 1/N * SUM_n_i=1(max(0, 1 - y_i*(w*x_i+b)))

            inputs:
                y_pred - model's output
                y - training data answer
                k - regularization strength

            output:
                np.array (shape = n_classes)
        """
        return (self.hinge_loss(y_pred, y) + (k * self.params.T @ self.params)).mean(axis=0)

    def _ovr_choose(self, x):
        """
            Get the position of the highest value

            input:
                x - input array, 2 dims

            output - np.ndarray, 1 dim
        """
        return np.argmax(x, axis=1) + 1

    def _ovr_split(self, x):
        """
            Get one at the position of the answer value, as in
            `.get_probabilities()`, only other values
            are equated to -1.
            input:
                x - input array, 2 dims

            output - np.ndarray, 1 dim
        """
        return self.get_probabilities(x, custom_fun=self._plus_minus_one)

    def _plus_minus_one(self, x):
        num = getattr(self, "num_classes")
        out = np.zeros(num) - 1
        out[int(x) - 1] = 1
        return out

    def _compute_gradient(self, x, y_pred, y, lr=0.01, k=1) -> np.ndarray:
        """
            Compute gradient using soft margin loss function

            inputs:
                x: np.ndarray - model's input
                y_pred: np.ndarray - model's output
                y: np.ndarray - true labels
                lr: float - learning rate
                k: float - regularization coefficient for loss function
        """
        diff = np.zeros_like(self.params)
        diff = self.params - (k * self.soft_margin_loss(y_pred, y) * (x.T @ y))

        return diff * lr

    def _forward(self, x):
        return x @ self.params

    def predict(self, x):
        self.data = x
        return self._ovr_choose(self.data @ self.params)
