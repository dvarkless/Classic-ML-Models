import numpy as np


def convert_to_emnist(img: np.ndarray):
    """
        Converts the image to match those of
        the emnist-letters dataset

        input: np.ndarray - shape(N,)
        output: np.ndarray - shape(N,)
    """
    img = np.resize(img, (28, 28)).astype(np.uint8).T
    return get_plain_data(img)


def get_plain_data(img):
    """
        The function aligns the resulting array and converts its values to 8-bit (0-255)

        input:
            img - input array to convert
        output:
            out - converted array (np.ndarray) (type: np.uint8)

    """
    return img.flatten().astype(np.uint8)


def threshold_mid(img):
    """
        The function applies a threshold function to each value in the array:
            >> threshold = 127
            >> pixel_value = 0 if pixel_value <= threshold else 255

        input:
            img - input array to convert
        output:
            out - converted array (np.ndarray) (type: np.uint8)

    """
    img = get_plain_data(img)
    img[img > 127] = 255
    img[img <= 127] = 0
    return img


def threshold_low(img):
    """
         The function applies a threshold function to each value in the array:
             >> threshold = 50
             >> pixel_value = 0 if pixel_value <= threshold else 255

         input:
             img - input array to convert
         output:
             out - converted array (np.ndarray) (type: np.uint8)

     """
    img = get_plain_data(img)
    img[img > 50] = 255
    img[img <= 50] = 0
    return img


def threshold_high(img):
    """
        The function applies a threshold function to each value in the array:
            >> threshold = 200
            >> pixel_value = 0 if pixel_value <= threshold else 255

        input:
            img - input array to convert
        output:
            out - converted array (np.ndarray) (type: np.uint8)

    """
    img = get_plain_data(img)
    img[img > 200] = 255
    img[img <= 200] = 0
    return img


class PCA_transform:
    """
        Applies the Principal Component Method (PCA) to the data array to reduce its dimension

        To set the vector of coefficients, you need to call the constructor of the class with the value
        the desired output dimension and call the method .fill(dataset) by passing to it
        training data set.

        Next, you can use an instance of the class as a function to convert individual dataset points.
        ref: "https://stackoverflow.com/questions/58666635/implementing-pca-with-numpy"

        use case:
            >> transformer = PCA_transform(3) # we want our output dataset to have 3 dims
            >> transformer.fit(dataset) # dataset.shape = (N, k) N - sample number, k - dimensions
            >> # transformer = PCA_transform(3).fit(dataset) # same thing
            >> 
            >> converted_ds1 = np.vectorize(transformer)(dataset)
            >> converted_ds2 = np.vectorize(transformer)(test_data) # use it for validation dataset too!

    """

    def __init__(self, n_dims) -> None:
        self.n_dims = n_dims

    def __call__(self, img):
        img = get_plain_data(img)
        return img @ self.PCA_vector

    def __repr__(self):
        return f'<PCA_transform({self.n_dims});vector:{self.PCA_vector.shape}>'

    def fit(self, dataset, answer_column=True):
        """
            Adjusting the converter to the transmitted data.
            
            Calculates and remembers the transformational vector
            based on the transmitted data set.

            Required method to call.

            input:
                dataset: np.ndarray - input dataset
                answer_column: bool - True if there is a column with answer
                labels in the dataset

            output - PCA_transform instance
        """
        if answer_column:
            dataset = dataset[:, 1:]
        dataset = np.array(list(map(get_plain_data, dataset)))
        dataset_cov = np.cov(dataset.T)
        e_values, e_vectors = np.linalg.eigh(dataset_cov)

        e_ind_order = np.flip(e_values.argsort())
        e_values = e_values[e_ind_order]
        self.PCA_vector = e_vectors[:, e_ind_order]
        self.PCA_vector = self.PCA_vector[:, :self.n_dims]

        return self
