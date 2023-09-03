import numpy as np
from model_base import BaseModel


class KnnModel(BaseModel):
    """
       The model is to the nearest neighbors.

       During training, it remembers the training data.
       When predicting, outputs the class of the nearest point for the transmitted
       points from the training data array.

       parameters:
           Arbitrary:
               data_converter - function-converter for input data
               k - number of neighbors

    """
    def __init__(self, custom_params=None) -> None:
        super().__init__(custom_params)
        self.assert_have(['k'])

    def fit(self, train_data):
        self.answer, self.train_data = self._splice_data(train_data)
        self.n = self.train_data.shape[0]

        return self

    def predict(self, new_points):
        self.data = new_points
        predictions = np.zeros(self.data.shape[0])
        k = getattr(self, 'k')
        for i, point in enumerate(self.data):
            distances = self._calculate_distances(point)
            # Find the K smallest distances to the points and their classes
            label_neighbors = self.answer[np.argpartition(distances, k)[:k]]
            # Write the most frequent answer
            predictions[i] = np.bincount(
                label_neighbors.astype("int64")).argmax()

            if hasattr(self, '_tick'):
                self._tick()
        return predictions

    def _calculate_distances(self, new_point):
        """
            The method calculates the Euclidean distances from the transmitted point to
            for each point in the training dataset and outputs them.

            inputs:
                new_point: np.ndarray - 1 row of data to calculate distance
            output:
                euclidean_distance: np.ndarray - array of distances (N_train,)
        """
        # Copy the entry point `n` times
        new_point = np.resize(new_point, (self.n, new_point.shape[0]))
        euclidean_distance = np.sum((self.train_data - new_point) ** 2, axis=1)
        return euclidean_distance
