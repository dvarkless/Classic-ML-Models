import numpy as np
from model_base import BaseModel, inval_check


class MultilogRegression(BaseModel):
    """
        A model of multinomial logistic regression.

        The principle of operation:
            The training sample is divided into mini samples randomly, the calculation takes place
            answers according to the formula:
                y[N, n_classes] = softmax(
                    X[N, n_input] @ theta[n_input, n_classes])
                where N is the number of values in the sample,
                n_classes - the number of classes in the response
                n_input - dimension of the input vector


            Next, the coefficients of the model (model.params) are minimized
            using gradient descent.

            parameters:
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
                    reg: str, function - regularization function (l1 or l2)
                Optional:
                    reg_w: float - regularization coefficient

    """

    def __init__(self, custom_params=None):
        self.must_have_params = ['data_converter', 'shift_column', 'normalization',
                                 'num_classes', 'learning_rate', 'batch_size', 'epochs',
                                 'reg']
        super().__init__(custom_params=custom_params)

    @inval_check
    def _compute_gradient(self, x, y_pred, y, lr=0.01, additive=None) -> np.ndarray:
        """
            A method for calculating the gradient descent step.

            inputs:
                x      - input vector
                y_pred - model's predicted output
                y      - dataset's real answer for this x
                lr     - learning rate
                additive - add extra weight if necessary
                           (e.g. regularization)

            output:
                numpy.ndarray - gradient value
        """
        if additive is None:
            additive = 0
        data_size = x.shape[0]
        diff = y_pred - y
        # disable refcheck to get a debugger to work
        diff.resize(data_size, y.shape[1], refcheck=False)
        gradient = (lr / data_size) * ((x.T @ diff) + additive)

        return gradient

    def _forward(self, x) -> np.ndarray:
        """
            The model calculates the response for a given input using
            its coefficient matrix at the moment

            input:
                x - input array
            output:
                y - output array, predicted answers
        """
        return self._softmax(x @ self.params)

    def fit(self, data: np.ndarray):
        # Get the hyperparameters if they are provided
        learning_rate = getattr(self, 'learning_rate', 0.01)
        epochs = getattr(self, "epochs", 100)

        # Splice the dataset
        y, x = self._splice_data(data)
        y = self.get_probabilities(y)
        num_samples = x.shape[0]

        if getattr(self, 'weight_init', 'zeros') == 'randn':
            self.params = np.random.randn(
                x.shape[1], getattr(self, 'num_classes')
            )
        else:
            self.params = np.zeros(
                (x.shape[1], getattr(self, 'num_classes'))
            )

        for epoch in range(epochs):
            # get minibatches
            indexes = np.random.permutation(num_samples)
            y = y[indexes]
            x = x[indexes]

            for x_batch, y_batch in self._iter_batches(x, y):
                y_pred = self._forward(x_batch)
                # Regularization
                reg_val = 0
                if getattr(self, 'reg', False):
                    reg_w = getattr(self, 'reg_w')
                    if getattr(self, 'reg', '') == 'l1':
                        reg_val = self.l2_reg(reg_w)
                    if getattr(self, 'reg', '') == 'l2':
                        reg_val = self.l2_reg(reg_w)

                self.params -= self._compute_gradient(
                    x, y_pred, y_batch, lr=learning_rate, additive=reg_val
                )

            if hasattr(self, '_tick'):
                self._tick()

            if getattr(self, "debug", False):
                print(
                    f'-------------------------epoch {epoch}--------------------------')
                y_pred = self._forward(x)
                self.cost = self.BCEloss(
                    y_pred, y, num_samples, regularization=getattr(
                        self, "reg", None)
                ).sum()
                print(f"training cost: {self.cost}")
                print(f"params = {self.params}")
                print(f"_forward {self._forward(x)}")
                pred_labels = self.get_labels(self._softmax(y_pred))
                y_labels = self.get_labels(self._softmax(y))
                for metric in getattr(self, "metrics", []):
                    print(f"{metric.__name__} = {metric(pred_labels, y_labels)}")

        return self

    def predict(self, x):
        """
            A method for predicting responses for an input vector,
            differs from `model._forward` in that the input and
            the output is converted

            input:
                x - raw, non-normalized vector
            output:
                y - predicted labels
        """
        self.data = x
        return self.get_labels(self._forward(self.data))
