import concurrent.futures as parallel
from itertools import combinations
from typing import Any

import numpy as np
from image_feature_detector import (PCA_transform, convert_to_emnist,
                                    get_plain_data, threshold_mid)
from metrics import (return_accuracy, return_f1, return_precision,
                     return_recall, return_roc_auc_ovr, show_metrics_matrix)
from model_base import BaseModel, inval_check
from model_runner import ModelRunner


class Question:
    """
        A class containing information about the node of the Decision Tree.

        When called, the nodes below along the tree are recursively called.
        The left branch is called if the node responded negatively to the received data, or
        the right branch is called if it responded positively.

        When converted to a string, it displays all the useful information contained in the node.    """

    def __init__(self, level: int, feature: int, question_val: float) -> None:
        self.level = level
        self.feature = feature
        self.question_val = question_val
        self.is_true_mask = np.array([])
        self.right_len = 0
        self.left_len = 0

    def __call__(self, row: np.ndarray):
        if row[self.feature] > self.question_val:
            return self.right_branch(row)
        else:
            return self.left_branch(row)

    def __name__(self):
        return f'Question "#{self.feature} > {self.question_val:.3f}?" on level {self.level}'

    def __repr__(self):
        return f'{self.level}) Is var #{self.feature} > {self.question_val:.3f}?\n \
                (G={self.gini:.2f})((L({self.left_len})->{self.left_mean:.3f})R({self.right_len})->{self.right_mean:.3f})'

    def split_by_question(self, data: np.ndarray):
        """
            input:
                data: np.ndarray - dataset to split
            output:
                tuple(data_false, data_true):
                    data_false: np.ndarray - part of dataset
                        where class' anwers results in False
                    data_true: np.ndarray - part of dataset
                        where class' answer results in True
        """
        is_true = data[:, self.feature] > self.question_val
        is_false = ~is_true
        self.right_len = data[is_true].shape[0]
        self.right_mean = data[is_true][:, -1].mean()

        self.left_len = data[is_false].shape[0]
        self.left_mean = data[is_false][:, -1].mean()

        return data[is_false], data[is_true]

    def my_gini(self, data: np.ndarray, min_samples: int = 0, balance_coeff: float = 1):
        """
            Compute gini coefficient for the dataset

            inputs:
                data: np.ndarray - dataset where we want to calculate gini
                min_samples: int - maximum size of dataset to ignore and return 0
                balance_coeff: float - balance coefficient for datasets,
                coeff > 1 - multiply gini coeff for true values side by coeff
                coeff < 1 - multiply gini coeff for false values side by 1/coeff

            output:
                self.my_gini: float - gini coefficient
        """
        self.gini = DecisionTreeClassifier.get_gini(
            *self.split_by_question(data), min_samples, balance_coeff)
        return self.gini

    def hold_branches(self, left_branch, right_branch):
        """
            inputs:
                left_branch - Question or Leaf class to refer to in case of
                              false answer
                right_branch- Question or Leaf class to refer to in case of
                              true answer
            output:
                self
        """
        self.left_branch = left_branch
        self.right_branch = right_branch

        return self


class Leaf:
    """
        Contains a mean answer for the passed data

        Can be converted into a string repr
    """

    def __init__(self, level: int, data: np.ndarray) -> None:
        self.level = level
        self.prediction = data[:, -1].mean()

    def __call__(self, row: Any):
        return self.prediction

    def __repr__(self):
        return f'{self.level}) return {self.prediction:.3f}'


class DecisionTreeClassifier(BaseModel):
    """
        The model of the Decision tree.

        A binary classification model learning with a teacher. Checks the input data
        according to several criteria and gives a response based on criteria verification.

        The nodes of the tree are generated based on the Gini coefficient (CART algorithm).
        The node divides the dataset based on this question in such a way that the resulting
        the halves of the dataset are obtained as homogeneous as possible. Next for the two halves
        new nodes are generated recursively and so on until parts of the datasets become
        completely homogeneous. In this case, sheets are generated in these places.

        Parameters:
            Arbitrary:
                data_converter: function - function-converter for input data
                sample_len: int - number of questions to choose from. Bigger - more precise and slower
                num_classes: int - number of classes for multiclass classification
                min_samples: int - returns for small while tree building then dataset is less
                                   or equal to this number (pre-pruning)
                max_depth: int - maximum depth of tree, return leafs then exceeded (pre-prunning)
                tree_type: str - Tree type chosen from ('binary', 'multilabel_ovr', 'multilabel_ovr')

    """

    def __init__(self, custom_params=None) -> None:
        self.must_have_params = [
            'sample_len',
            'num_classes',
            'min_samples',
            'max_depth',
            'tree_type',
        ]
        super().__init__(custom_params)
        self.tree_root = []

        if getattr(self, 'balance_coeff', 1) <= 0:
            raise ValueError(
                f'Balance coefficient could not be negative or zero, got {getattr(self,"balance_coeff")}')

    def fit(self, dataset: np.ndarray):
        self.y, self.x = self._splice_data(dataset)
        # Generate an answer array based on the type of the tree
        if getattr(self, 'tree_type') in ('binary', 'multilabel'):
            self.y = self.y[:, np.newaxis]
        elif getattr(self, 'tree_type') == 'multilabel_ovo':
            self.y = self._ovo_split(self.y)
            if not hasattr(self, 'balance_coeff'):
                self.balance_coeff = getattr(self, 'num_classes')//2
        elif getattr(self, 'tree_type') == 'multilabel_ovr':
            self.y = self.get_probabilities(self.y)
            if not hasattr(self, 'balance_coeff'):
                self.balance_coeff = getattr(self, 'num_classes')
        else:
            raise AssertionError(
                f"Unknown tree type: '{getattr(self, 'tree_type')}'")

        self.tree_root = []
        self.question_list = self._return_ranges(
            self.x.max(), self.x.mean(), getattr(self, 'sample_len'))

        # Multiprocessing module cannot process proxy objects
        if hasattr(self, '_tick'):
            del self._tick
        with parallel.ProcessPoolExecutor() as executor:
            results = executor.map(
                self.grow_branch, range(self.y.shape[1]), timeout=60)

        self.tree_root = list(results)

    def grow_branch(self, num_branch):
        """
            Build a tree branch for a single column.

            inputs:
                num_branch: int - current branch number
            output:
                Question or Leaf class
        """
        data = np.hstack((self.x, self.y[:, num_branch][:, np.newaxis]))

        return self.build_tree(0, data)

    @inval_check
    def _return_ranges(self, max_val: float, mean_val: float, num: int):
        """
            Returns values from `min_val` to `max_val` with smaller steps in the `mean_val` area,
            the step value approximately corresponds to the density of the normal distribution.

            >>> self._return_ranges(5, 2, 10)
            [0.22  0.94  1.34  1.67  1.92  2.17  2.42  2.68  3.04  3.81]

            inputs:
                max_val: float - maximum value of input dataset
                mean_val: float - mean value of input dataset
                num: int - number of values in output range we want
        """
        if num > 500:
            raise ValueError(f'num is too high {num}, should be less than 500')
        sample = ((max_val-mean_val)/3*np.random.randn(1000))+mean_val
        sample.sort()
        out = []
        for i in range(num):
            low = len(sample)//num*i
            high = len(sample)//num*(i+1)
            out.append(sample[low:high].mean())

        return np.array(out)

    def build_tree(self, level: int, data: np.ndarray):
        """
            With a recursive call, builds a Decision Tree.

            A node is created for the input data, then recursively
            we are building nodes for the left and right branches of this node.
            With a small Gini coefficient, we return the Sheet on this
            place.

            inputs:
                level: int - level of tree node, used in printing
                data: np.ndarray - training data on current node to
                                   process
            output:
                Question or Leaf class

        """
        my_question = self.create_node(data, level)

        if my_question.my_gini(data, getattr(self, 'min_samples'), getattr(self, 'balance_coeff', 1)) == 0:
            return Leaf(level, data)
        if level == getattr(self, 'max_depth', 10):
            return Leaf(level, data)

        left, right = my_question.split_by_question(data)

        left_branch = self.build_tree(level+1, left)
        right_branch = self.build_tree(level+1, right)

        return my_question.hold_branches(left_branch, right_branch)

    def predict(self, input_data: np.ndarray):
        self.data = input_data
        out = np.zeros((input_data.shape[0], self.y.shape[1]))
        for i, tree in enumerate(self.tree_root):
            out[:, i] = np.array(list(map(tree, self.data)))

        if getattr(self, 'tree_type') in ('binary', 'multilabel'):
            return out
        elif getattr(self, 'tree_type') == 'multilabel_ovo':
            return self._ovo_choose(out)
        elif getattr(self, 'tree_type') == 'multilabel_ovr':
            return self._ovr_choose(out)

    @inval_check
    def _ovr_choose(self, x: np.ndarray):
        """
        Get a position of the highest value

        input:
            x - input array, 2 dims

        output - np.ndarray, 1 dim
        """
        return np.argmax(x, axis=1) + 1

    def _ovo_split(self, x: np.ndarray):
        """
            We divide the data into pairs for the classifier
            like One Vs One.

            The length of the returned array is equal to the number of combinations
            2 classes each in the input array.

            If the result is positive, 1 is returned in the column,
            if negative, we return -1 if the answer is
            is a different class, we return 0.

                          (1vs2)......
            [[1],           [1].......
             [2],   --->    [-1]......
             [3]]           [0].......

            inputs:
                x: np.ndarray - input answer data with classes as numbers
            output:
                out: np.ndarray - output array
        """
        num_count = getattr(self, 'num_classes')
        pairs = list(combinations(range(1, num_count+1), 2))
        out = np.zeros((x.shape[0], len(pairs)))
        for i, vals in enumerate(pairs):
            val1, val2 = vals
            out[:, i] = -x
            out[:, i][out[:, i] == -val1] = 1
            out[:, i][out[:, i] == -val2] = -1
            out[:, i][out[:, i] < 0] = 0

        return out

    def _ovo_choose(self, x: np.ndarray):
        """
            Convert `One vs One` classificator's answers into 
            a classes.

            inputs:
                x: np.ndarray - input answer data from ovo
            output:
                out: np.ndarray - output array with classes as numbers
        """
        num_count = getattr(self, 'num_classes')
        out = np.zeros((x.shape[0], num_count))
        for i, vals in enumerate(combinations(range(num_count), 2)):
            val1, val2 = vals
            out[:, val1] += x[:, i]
            out[:, val2] -= x[:, i]

        return self._ovr_choose(out)

    def _array_window(self, arr: np.ndarray, x: int, y: int, size: int):
        """
            Creates a window in an array of size size centered in x, y

            !!!NOT USED!!!
            inputs:
                arr: np.ndarray - array to slice from
                x: int - center x
                y: int - center y
                size: int - window size
            output:
                out: np.ndarray - smaller array
        """
        arr = np.roll(np.roll(arr, shift=-x+1, axis=0), shift=-y+1, axis=1)
        return arr[:size, :size]

    def create_node(self, data: np.ndarray, level: int):
        """
            Formulates a question for a given data set.

            Uses the Gini coefficient to separate the data.

            TODO: Add parallel computing

            inputs:
                data: np.ndarray - input data without answers
                level: int - current tree level
            output:
                Question class with optimal question (x > question_list(y)?)
        """

        feature_len = self.x.shape[1]

        possible_questions = np.ones(
            (feature_len, len(self.question_list))) * self.question_list
        feature_array = np.ones(
            (feature_len, len(self.question_list))) * np.arange(feature_len)[:, np.newaxis]
        feature_array = feature_array.astype(np.int64)

        v_gini = np.vectorize(self.calculate_gini)
        v_gini.excluded.add(0)
        local_gini = v_gini(data, feature_array, possible_questions)

        max_ind = np.unravel_index(
            np.argmax(local_gini, axis=np.newaxis), local_gini.shape)
        x, y = [int(val) for val in max_ind]

        return Question(level, x, self.question_list[y])

    @inval_check
    def calculate_gini(self, data: np.ndarray, feature: int, question_val: float):
        """
            Wrapper for a gini calculation method.

            inputs:
                data: np.ndarray - input data without answers
                feature: int - column number in data
                question_val: float - threshold number for data
            output:
                gini_val: float - gini value
        """
        my_question = Question(0, feature, question_val)

        left, right = my_question.split_by_question(data)
        return self.get_gini(left, right, getattr(self, 'min_samples', 0),
                             getattr(self, 'balance_coeff', 1))

    @staticmethod
    def get_gini(left: np.ndarray, right: np.ndarray, min_samples: int = 0, balance_coeff: float = 1):
        """
            A method for calculating the Gini coefficient.

            The formula used:
                1/L * SUM_n_i=1(l_count_i*w_i) + 1/R * SUM_n_i=1(r_count_i*w_i) -> max

                , where L, R is the number of elements in the arrays
                l_count, r_count - the number of unique elements in arrays
                w_i - weight for a specific value

            inputs:
                left: np.ndarray - output data from question, false branch
                right: np.ndarray - output data from question, true branch
                min_samples: int - return 0 if length of one of the datasets
                is smaller than it
                balance_coeff: float - balance coefficient for datasets,
                coeff > 1 - multiply gini coeff for true values side by coeff
                coeff < 1 - multiply gini coeff for false values side by 1/coeff

        """
        if balance_coeff >= 1:
            bal_true = balance_coeff
            bal_false = 1
        else:
            bal_true = 1
            bal_false = 1/balance_coeff

        l_index, l_counts = np.unique(left[:, -1], return_counts=True)
        r_index, r_counts = np.unique(right[:, -1], return_counts=True)

        sum_L = l_counts.sum()
        sum_R = r_counts.sum()

        l_counts[l_index != 0] *= bal_true
        l_counts[l_index == 0] *= bal_false

        r_counts[r_index != 0] *= bal_true
        r_counts[r_index == 0] *= bal_false
        if np.array_equal(l_index, r_index) and len(l_index) == 1:
            return 0
        if sum_L <= min_samples or sum_R <= min_samples:
            return 0
        return np.sum(np.power(l_counts, 2))/sum_L + np.sum(np.power(r_counts, 2))/sum_R

    def print_tree(self):
        print('========DecisionTreeModel=======')
        print(f'Parameters are: ')
        for name in self.must_have_params:
            val = getattr(self, name)
            print(f'{name} = {val}')
        if hasattr(self, 'balance_coeff'):
            print(f'balance_coeff = {getattr(self,"balance_coeff")}')
        print('===Tree structure:===')
        for i, branch in enumerate(self.tree_root):
            print(f'===================={i}=======================')
            self.__print_tree(branch)

    def __print_tree(self, node, spacing=""):

        # Print the question at this node
        print(spacing + str(node))

        if isinstance(node, Leaf):
            return
        # Call this function recursively on the true branch
        print(spacing + '--> True:')
        self.__print_tree(node.right_branch, spacing + "  ")

        # Call this function recursively on the false branch
        print(spacing + '--> False:')
        self.__print_tree(node.left_branch, spacing + "  ")


if __name__ == "__main__":
    np.seterr(invalid="ignore")

    training_data_lite = np.genfromtxt("datasets/light-train.csv",
                                       delimiter=",", filling_values=0)
    evaluation_data_lite = np.genfromtxt(
        "datasets/medium-test.csv", delimiter=",", filling_values=0)
    evaluation_input_lite = evaluation_data_lite[:, 1:]
    evaluation_answers_lite = evaluation_data_lite[:, 0]

    datapack_lite = (training_data_lite, evaluation_input_lite,
                     evaluation_answers_lite)

    my_metrics = [return_accuracy, return_recall, return_precision, return_f1]

    PCA_extreme = PCA_transform(24).fit(training_data_lite)

    hp = {
        'data_converter': PCA_extreme,
        'sample_len': 32,
        'num_classes': 26,
        'window_size': -1,
        'min_samples': 3,
        'max_depth': 7,
        'tree_type': 'multilabel_ovr',
    }

    TreeRunner = ModelRunner(DecisionTreeClassifier,
                             defaults=hp, metrics=my_metrics, responsive_bar=True)
    params_to_change = {
        'sample_len': [32],
    }

    import cProfile
    import pstats

    with cProfile.Profile() as pr:
        TreeRunner.run(*datapack_lite, params_to_change, one_vs_one=True)

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    # stats.print_stats()
    # stats.dump_stats(filename='my_stats.prof')
