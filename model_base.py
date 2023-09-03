import functools
import math
from collections.abc import Generator

import numpy as np


class BaseModel:
    def __init__(self, custom_params=None) -> None:
        """
            use case:
                class MyModel(BaseModel):
                    super.__init__(self, kwargs)

            inputs:
                data_converter: Function - converter for input vectors

                custom_params: dict - a dict object for children model's parameters
                                      use self.must_have_params() method to check parameter's 
                                      availability

        """
        self._data = np.array([])
        self._params = np.array([])

        if custom_params:
            for key, val in custom_params.items():
                setattr(self, key, val)

        self.assert_have(['data_converter'])

    def assert_have(self, must_have_names: list) -> None:
        """
            Checks whether the class has attributes with names as in the list

        """
        for name in must_have_names:
            getattr(self, name)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, input_data):
        data = np.array(list(map(self.data_converter, input_data)))
        # Normalize if needed
        if getattr(self, 'normalization', False):
            std = np.std(data, axis=0)
            std[std == 0] = 1
            self._data = (data - np.mean(data, axis=0))/std
        else:
            self._data = data

        if getattr(self, 'shift_column', False):
            self._data = np.hstack(
                (self._data, np.ones((self._data.shape[0], 1))))

    @property
    def params(self) -> np.ndarray:
        # Do some math checks
        if math.isnan(self._params[0, 0]) or math.isinf(self._params[0, 0]):
            raise ValueError(f"self.params is invalid \n {self._params}")
        return self._params

    @params.setter
    def params(self, input_data):
        self._params = input_data

    @property
    def cost(self):
        return self._cost

    @cost.setter
    def cost(self, input_data):
        # Save cost functions to use them in the plot
        if not hasattr(self, '_cost_list'):
            self._cost_list = []
        self._cost_list.append(input_data)
        self._cost = input_data

    @staticmethod
    def wierd_vals_detector(x: float):
        """
            input:
                x: float - value to check
            output:
                x: float - the same value
        """
        if x in (np.inf, -np.inf, np.nan):
            raise ValueError(f"Invalid value detected x = {x}")
        return x

    def l1_reg(self, weight: float) -> np.ndarray:      # L1 regularization
        """
            L1 regularization

            input - weight constant (0 < c < 1)
        """
        return weight * np.abs(self.params).sum(axis=0)

    def l2_reg(self, weight: float) -> np.ndarray:      # L2 regularization
        """
            L2 regularization

            input - weight constant (0 < c < 1)
        """
        return weight * np.sqrt(np.power(self.params, 2)).sum(axis=0)

    def BCEloss(self, y_pred, y, sample_size, weight=0.01, regularization=None) -> np.ndarray:
        """
            Cost Function:
            Binary cross entropy 
            Used in models where output values range from 0 to 1.
            Differs from a mathematical formula by the presence of a restriction for 
            the minimum value of the logarithm = -100 (as in Pytorch)
            inputs:
                y_pred - model's predicted values in range 0-1 (output)
                y      - true outputs in range 0-1
                sample size - batch size or dataset size
                weight - formula constant
                regularization: Function - pass the regularization function 
                         to add to the resulting formula
                regularization: str - pass 'l1' or 'l2' to use class methods for this job
        """
        log_val = np.log10(y_pred)
        np.putmask(log_val, log_val < -100, -100)
        add1 = y * log_val
        log_val = np.log10(1 - y_pred)
        np.putmask(log_val, log_val < -100, -100)
        add2 = (1 - y) * log_val
        out = -weight * (add1 + add2)
        # отключаем refcheck для работы дебаггера
        out.resize(sample_size, out.shape[1], refcheck=False)

        if regularization == "l1":
            out += self.l1_reg(weight)
        elif regularization == "l2":
            out += self.l2_reg(weight)
        elif type(regularization) == type(self.l1_reg):
            out += regularization(weight)
        return out

    def _softmax(self, x):
        """
            Returns a soft maximum for an array.
            The sum of values for one dimension is equal to 1,
            so you can interpret the resulting array
            as an array of probability vectors

            Calculated by rows.

            input:
                x - input array
            output:
                numpy.ndarray
        """
        mod_array = x + 2   # Таким образом можно избежать деления на крохотные числа
        softvals = (np.exp(mod_array - np.max(x)).T /
                    np.exp(mod_array - np.max(x)).sum(axis=1)).T

        return np.vectorize(self.wierd_vals_detector)(softvals)

    def _iter_batches(self, *args: np.ndarray) -> Generator:
        """
            Returns iterator which divides datasets into batches

            use case:
                self.batch_size = 50
                >>> for x_batch, y_batch in self._iter_batches(x,y):
                >>>     pass
            inputs:
                args: np.ndarray - multiple datasets to divide and iterate to
            output:
                Generator object
        """
        batch_size = getattr(self, "batch_size", args[0].shape[1])

        for i in range(int(args[0].shape[0]/batch_size)):
            curr_stack = i * batch_size

            out_arrs = []
            for arr in args:
                arr_batch = arr[curr_stack: curr_stack + batch_size, :]
                out_arrs.append(arr_batch)

            yield out_arrs

    def get_probabilities(self, ans_arr: np.ndarray, custom_fun=None) -> np.ndarray:
        """
            Convert answers into probabilities
            (counts from 1)
            Example:
                >> self.num_classes = 3
                >> a = [1, 2, 3]
                >> self.get_probabilities(a)
                [[1, 0, 0],
                 [0, 1, 0], 
                 [0, 0, 1]]
        """
        num = getattr(self, "num_classes")
        buff_arr = np.zeros(shape=(ans_arr.shape[0], num))
        for i, val in enumerate(ans_arr):
            buff_arr[i, :] = self._ans_as_probs(
                val) if custom_fun is None else custom_fun(val)

        return buff_arr

    def _ans_as_probs(self, ans):
        """
            Converts each answer into probability
        """
        num = getattr(self, "num_classes")
        out = np.zeros(num)
        out[int(ans) - 1] = 1

        return out

    def get_labels(self, ans_arr):
        """
            Converts probability into answers
            (counts from 1)
            Example:
                >> a = [[0.1, 0.5, 0.4],
                 [0.1, 0.1, 0.8], 
                 [1, 0, 0]]
                >> self.get_probabilities(a)
                 [2, 3, 1]

        """
        return np.array(list(map(self._ans_as_labels, ans_arr)))

    def _ans_as_labels(self, ans):
        return np.argmax(ans) + 1

    def _splice_data(self, train_data) -> tuple:
        """
            Divides the EMNIST training dataset into responses and input values

            also converts and normalizes input values

            input: 
                train_data: np.ndarray - whole EMNIST dataset, first column is answers, 
                                         the rest is input
        """
        ans = train_data[:, 0]
        self.data = train_data[:, 1:]
        return ans, self.data

    def define_tick(self, foo, additive=0):
        """
            Allows you to pass a function to be called when
            the model passes one step of training.

            Used for smoother display of the loading bar
            when launching a model via ModelRunner

            input:
                foo - function to call
                additive: int - value to add to returning value
            output:
                count_ticks: int - how many times function will be called
        """
        self._tick = foo
        type1 = ['batch_size', 'epochs']
        type2 = ['tree_type']
        type3 = ['k']

        if all([hasattr(self, attr) for attr in type1]):
            return getattr(self, 'epochs')
        elif all([hasattr(self, attr) for attr in type2]):
            num_classes = getattr(self, 'num_classes')
            tree_table = {
                'binary': 1,
                'multilabel': 1,
                'multilabel_ovo': num_classes*(num_classes-1)//2,
                'multilabel_ovr': num_classes,
            }
            return tree_table[getattr(self, 'tree_type')]
        elif all([hasattr(self, attr) for attr in type3]):
            return 1+additive
        else:
            return 1

    def get_cost_list(self):
        """
            return costs per epoch if exists
        """
        try:
            return self._cost_list
        except AttributeError as e:
            print(e)
            raise AttributeError(
                'Compute cost function to get a cost plot: \nself.cost = self.cost_func(y_pred, y)  //example')

    def get_params(self) -> np.ndarray:
        return self.params

    def predict(self, x):
        """
            A method for predicting responses for an input vector,
            differs from model._forward in that the input and
            the output is converted

            input:
                x - raw, non-normalized vector
            output:
                y - predicted labels
        """

        raise NotImplementedError

    def fit(self, data):
        """
            The method divides the input sample into subsamples and
            finds optimal parameters for calculation.

            inputs:
                data - input data, answers + input vectors
                       [:, 0] - answers
                       [:, 1:] - input vectors
                learning_rate: float
                batch_size - subsample size
                epochs - number of iterations for whole sample
                         number of gradient steps:(epochs*(N/batch_size))

            output:
                self - model instance
        """

        raise NotImplementedError


def inval_check(func):
    @functools.wraps(func)
    def _wrapper(self, *args, **kwargs):
        out = func(self, *args, **kwargs)
        if isinstance(out, np.ndarray):
            np.vectorize(BaseModel.wierd_vals_detector)(out)
        elif isinstance(out, list) or isinstance(out, tuple):
            if isinstance(out[0], np.ndarray):
                np.vectorize(BaseModel.wierd_vals_detector)(out)
        return out
    return _wrapper
