import numpy as np
from model_base import BaseModel


class BayesianClassifier(BaseModel):
    """
        A model of a Naive Bayesian classifier.

        Uses Bayes' theorem to determine the class of a point in space:
            P(Y | a0, a1, ... an) = (P(Y) * P(a0, a1, ... an | Y)) / P(a0, a1, ... an)
            P() - probability
            Y - model's answer (0 или 1)
            a0, a1, ... an - model's parameters

        For multiclass classification in this case, the Y value is represented as
        vectors Y = (Y1, Y2... Ym). The highest probability in
        the vector is chosen as the answer.

        To correctly calculate the probabilities of the parameters, the input values must be equal to
        0 or 1

        Parameters:
            Arbitrary:
                num_classes: int - number of classes in dataset
    """

    def __init__(self, custom_params=None):
        super().__init__(custom_params)
        self.assert_have(['num_classes'])

    def fit(self, data):
        y, x = self._splice_data(data)
        y = self.get_probabilities(y)

        max_val = x.max()
        prob_in = x.T.mean(axis=1)/max_val  # P(a0, a1, ... an)
        prob_ans = y.mean(axis=0)  # P(Y)

        prob_conditional = x.T @ y / y.sum(axis=0)  # P(a0, a1, ... an | Y)
        # Get inverse matrix. Lower the maximum value, to make softmax work properly
        prob_inverse = np.power(prob_in, -1)/10000
        # Get rid of zero division
        prob_inverse[prob_inverse == np.inf] = 0
        self.params = np.outer(prob_inverse, prob_ans) * prob_conditional

        return self

    def predict(self, x):
        self.data = x
        return self.get_labels(self._softmax(self.data @ self.params))
