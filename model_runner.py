import functools
import math
import time
from itertools import product
from typing import Callable

import numpy as np
from alive_progress import alive_bar


def timer(attr):
    @functools.wraps(attr)
    def _wrapper(self, *args, **kwargs):
        start = time.perf_counter()
        result = attr(self, *args, **kwargs)
        runtime = time.perf_counter() - start
        print(f'{runtime:.3f}s')
        return result
    return _wrapper


class ModelRunner:
    """
        A class designed to conveniently run machine learning models.

        Its features:
            - Creation of model instances with parameters set via the dictionary
              and their launch via methods `.fit()` and `.predict().`
            - Output of the progress scale and execution time of model methods
            - Output of various metrics
            - Launch of a single model with a combination of different parameters
        use case:
            >>> defaults = {'lr': 0.01, 'epochs': 100}
            >>> runner_inst = ModelRunner(ModelClass, timer=True, defaults=defaults, metrics=[accuracy])
            >>> runner_inst.run(training_data, eval_input, eval_answers, params={'lr': [0.001, 0.005], 'batch_size':[100],})

        inputs:
            model_class - Class of your model (not instance), all parameters should be passed through **kwargs
            defaults: dict - default kwargs for your model
            metrics: list - list of functions, they must take only two positional args: foo(preds, answers)
    """

    def __init__(self, model_class, defaults=None, metrics=None, responsive_bar=False) -> None:
        self.model_class = model_class
        self.metrics = metrics
        self._metric_data = []
        self._parameters_data = []
        if defaults is not None:
            self.defaults = defaults

        self._responsive_bar = responsive_bar

    def run(self, train: np.ndarray, eval_input: np.ndarray, eval_ans: np.ndarray, params: dict, one_vs_one: bool = False):
        """
            Start checking models with the specified data and parameters.

            The iterable parameters are set in the params dictionary as:
                >>> params = {
                >>>     'lr': [1,2,3,4]
                >>>     'epochs': [100, 200]
                >>> }
            The number of verification steps in this case also depends on the method of combination

            Parameters:
                - When one_vs_one=True, all available parameters are combined with each other
                with a friend, in this example, 8 steps are obtained

                - When one_vs_one=False, the parameters are taken by columns, while
                if there are not enough values in some list, then its last one is taken
                the value in the list. In this example, 4 steps are obtained
            inputs:
                train - training dataset, first column is answer labels
                eval_input - evaluation dataset without answers
                eval_answers - answer array in the same order as eval_input
                               size = (1, N)
                params - dict consisted of lists of the iterated parameters.
                        every value must be a list, even singular vals
                one_vs_one - parameters combination method, True is One vertus One;
                            False is columswise combination.

        """
        self._metric_data = []
        self._models = []
        curr_params = dict()
        if one_vs_one:
            # Check if there is a single value
            if len(list(params.values())) <= 1:
                pairs = list(*params.values())
            else:
                pairs = list(product(*list(params.values())))

            if self._responsive_bar:
                len_model_ticks = self.model_class(
                    self.defaults).define_tick(None, additive=len(eval_ans))
            else:
                len_model_ticks = 1
            with alive_bar(len(list(pairs)*len_model_ticks), title=f'Проверка модели {self.model_class.__name__}', force_tty=True, bar='filling') as bar:
                # Unpack parameters
                for vals in pairs:
                    for i, key in enumerate(params.keys()):
                        try:
                            curr_params[key] = vals[i]
                        except TypeError:
                            curr_params[key] = vals

                    print('-----With parameters-----')
                    for key, val in curr_params.items():
                        print(f'{key} = {val}')

                    self._parameters_data.append(list(curr_params.values()))
                    self._run_method(train, eval_input,
                                     eval_ans, curr_params, bar)
                    print('-----End with-----')
                    bar()
        else:
            iter_lens = [len(val) for val in params.values()]
            if self._responsive_bar:
                len_model_ticks = self.model_class(
                    self.defaults).define_tick(None, additive=len(eval_ans))
            else:
                len_model_ticks = 1
            max_len = max(iter_lens)
            with alive_bar(max_len*len_model_ticks, title=f'Проверка модели {self.model_class.__name__}', force_tty=True, bar='filling') as bar:
                for i in range(max_len):
                    for pos, key in enumerate(params.keys()):
                        this_len = iter_lens[pos]
                        try:
                            curr_params[key] = params[key][min(
                                this_len - 1, i)]
                        except TypeError:
                            curr_params[key] = params[key]

                    print('-----With parameters-----')
                    for key, val in curr_params.items():
                        print(f'{key} = {val}')

                    self._parameters_data.append(list(curr_params.values()))
                    self._run_method(train, eval_input,
                                     eval_ans, curr_params, bar)
                    print('-----End with-----')
                    bar()

        print("===============RESULTS=================")
        pos = self._highest_metric_pos(self._metric_data)
        print(f'On iteration {pos}:')
        print(f"With hyperparameters: {self._parameters_data[pos]}")
        print(f'Got metrics: {self._metric_data[pos]}')

    def _run_method(self, train: np.ndarray, eval_input: np.ndarray, eval_ans: np.ndarray, params: dict, bar_obj: Callable):
        params_to_pass = self._mix_params(self.defaults, params)
        self.model = self.model_class(params_to_pass)

        if self._responsive_bar:
            self.model.define_tick(bar_obj, len(eval_ans))

        print('~fit complete in ', end='')
        self._run_train(train)

        print('~eval complete in ', end='')
        answer = self._run_eval(eval_input)
        self._comma_metrics(answer, eval_ans)
        self._models.append(self.model)

    def _mix_params(self, main, invasive):
        """
            The method is changing the dictionary with parameters

            It makes changes to the main dictionary with parameters 
            from another dictionary. The main dictionary does not change at the same time.

            inputs:
                main: dict - dict to be  inserted values into
                invasive: dict - mixed in values
            output - new dict with mixed values
        """
        maincpy = main.copy()
        for key, val in invasive.items():
            maincpy[key] = val
        return maincpy

    def _comma_metrics(self, preds, evals):
        buff = []
        for metric in self.metrics:
            res = metric(preds, evals)
            print(f"    {metric.__name__} = {res:.3f}")
            buff.append(res)
        self._metric_data.append(buff)

    def _highest_metric_pos(self, metrics):
        score = [math.prod(vals) for vals in metrics]
        return score.index(max(score))

    def get_models(self):
        return self._models

    def get_metrics(self):
        return self._metric_data

    def get_params(self):
        return self._parameters_data

    @timer
    def _run_train(self, train: np.ndarray):
        self.model.fit(train)

    @timer
    def _run_eval(self, eval_input: np.ndarray):
        return self.model.predict(eval_input)
