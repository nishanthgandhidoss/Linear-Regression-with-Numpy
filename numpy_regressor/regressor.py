# Import Pandas
import pandas as pd

# Import Numpy
import numpy as np

# Import Random
import random

# Import DataProcessing
from .data_processing import DataProcessing

class Regression:
    """Perform regression on data

    Looks over the predictor and target outputs with linear and Gaussian
    basis function with different lamda values and doing the model selection
    based on k-fold cross validation on the metrics of least mean sum of
    squares error value.

    Arguments:
        k_fold: Number of folds of the data used for the model selection
        lamda_values: List of float values generally of increasing log scale
            that are used in the closed solution to find best optimal lamda
            for the final model.
    """

    # Creating the data Processing object for training data
    data_process = DataProcessing()

    def __init__(self, k_fold=7):
        self.k_fold = k_fold
        self.lamda_values = [0, 0.0001, 0.001, 0.01, 0.1, 0, 1, 10, 100]
        self.kernels = ["linear", "gaussian"]
        self.parameters_list = []
        self.n_points = 10
        self.model_param = []

    def lmse(self, actual, pred):
        """ Find the Prediction LMSE

        LMSE -  Least Mean Square Error is the average squared difference
        between the prediction and actual values. The inputs are actual and
        predicted values that outputs the LMSE rounded off to two digits.

        Arguments:
            actual: Actual value from the dataset. continuous value.
            pred: Predicted value for the data. continuous value.

        Returns:
            lmse: Least Mean Square Error value
        """
        # Computing the LMSE and rounding of to two digits
        lmse = round(np.sum((actual - pred) ** 2) / len(actual), 2)
        return lmse

    def gaussian_kernel(self, data, n_points=10):
        """ Apply gaussian function on the data

        Dataset with x feature vector is passed over the n - gaussian function that
        is computed using random pick of observation from the dataset and converted
        into n feauture vectors from x features. This makes the data to be around your
        n data points you selected.

        Arguments:
            data: Data to be changed into gaussian form
            n_points: Number of gaussian kernels/functions/points to refine into

        Returns:
            gaussian: N feature vectors represetation of the input dataset.
        """
        # Set the sigma value to be no. of original features
        sigma = data.shape[1]
        # Setting the random seed and sampling the points from the data
        random.seed(100)
        points = random.sample(list(data.index), n_points)
        # Initialize the gaussian kernel feature vector
        gaussian = np.zeros((data.shape[0], n_points))
        col = 0
        # Iterate through the n points to come up with n gaussian feauture vectors
        for point in points:
            gaussian[:, col] = np.exp(-np.linalg.norm(data - data.loc[point, :], 2, axis=1) ** 2
                                      / (2. * sigma ** 2))
            col += 1
        return gaussian

    def cv_fit(self, x_train, y_train, k_fold):
        """ Model Selection using Cross validation

        Selecting the best fit regression model and hyper parameters such
        as lamda and basis functions using cross validation on the feauture
        vectors and our output variable(s) and linear regression's closed
        form solution.

        Arguments:
            x_train: Predictors of the Training set
            y_train: Output of the Training set
            k_fold: Number of fold for the cross validation

        Returns:
            model_param: A dictinoary consists of selected model weights,
                lamda value, lmse value, and kernel functions
        """

        # Initialize variables
        weights = np.zeros((x_train.shape[1],))
        best_weights = weights
        identity = np.identity(x_train.shape[1])
        best_lmse = float("inf")
        best_lamda = 0.
        best_kernel = ""

        # K Fold cross validation index generetion
        cv = self.data_process.k_fold(data=x_train, k=k_fold)

        # Looping through the basis function
        for kernel in self.kernels:
            # Chacking the Gaussian function
            if kernel is "gaussian":
                x_train = self.gaussian_kernel(pd.DataFrame(x_train), self.n_points)
                identity = np.identity(x_train.shape[1])
            # Looping through Lamda
            for lamda in self.lamda_values:
                lmse = 0
                for train_cv, test_cv in cv:
                    # Split up the train and test for each iteration
                    x_train_cv = x_train[train_cv]
                    y_train_cv = y_train[train_cv]
                    x_test_cv = x_train[test_cv]
                    y_test_cv = y_train[test_cv]
                    # Closed form solution for each lamda values
                    inverse = np.linalg.inv(lamda * identity + np.dot(x_train_cv.T, x_train_cv))
                    data_term = np.dot(x_train_cv.T, y_train_cv)
                    weights = np.dot(inverse, data_term)
                    # Predict with the test set
                    prediction = np.dot(x_test_cv, weights)
                    # Find the LMSE value for the set
                    lmse += (np.sum((y_test_cv - prediction) ** 2) / len(y_test_cv))
                # LMSE for each lamda
                lmse = lmse / k_fold
                # Preparing the parameter list for further analysis
                # Storing the result in class variable
                parameters = {'basis_function': kernel,
                              'lamda': lamda,
                              'lmse': lmse,
                              'weights': weights}
                self.parameters_list.append(parameters)
                # Update the best parameters
                if best_lmse > lmse:
                    best_lmse = lmse
                    best_lamda = lamda
                    best_weights = weights
                    best_kernel = kernel
        model_param = {"weights": best_weights, "kernel": best_kernel, "lamda": best_lamda, "lmse": best_lmse}
        self.model_param.append(model_param)
        return model_param

    def my_regression(self, trainX, testX, noutputs):
        """ Make Prediction for continuous output(s)

        Takes training set and performs cross validation to select the best
        parameter (lamda value of the closed form solution, basis function)
        for the model selection and use that model on the testing set to
        output the prediction for the testing data.

        Arguments:
            trainX - an [ntrain x (nfeature + noutputs)] array that contains
                the features in the first 'nfeature' columns and the outputs
                in the last 'noutput' columns
            testX - an [ntest x nfeature] array of test data for which the
                predictions are made
            noutputs - the number of output columns in trainX

        Returns:
            prediction: [ntest x noutputs] array, which contains the prediction
                values for the testX data
        """
        # Normalize the train and test data
        trainX.iloc[:, 0:-noutputs] = self.data_process.z_score_norm(trainX.iloc[:, 0:-noutputs])
        testX = self.data_process.z_score_norm(testX)
        # Split the train data into predictors and target variables
        x_train, y_train = self.data_process.pred_target_split(trainX, noutputs=noutputs)
        # Include the bias term in the feature vector at the last
        x_train["bias"] = 1
        testX["bias"] = 1
        # Convert Dataframe to Numpy array
        x_train = x_train.values
        y_train = y_train.values
        # Intializing the prediction array with zeros
        prediction = np.zeros((testX.shape[0], noutputs))
        # Get the best fit for the data with cross validation
        model = self.cv_fit(x_train, y_train, self.k_fold)
        # Check if the kernel is gaussian
        if model["kernel"] is "gaussian":
            # Redefine in gaussian feature
            gaus = self.gaussian_kernel(pd.DataFrame(np.concatenate((x_train, testX.values))),
                                        self.n_points)
            x_train_gaus = gaus[0:len(x_train), :]
            x_test_gaus = gaus[len(x_train):, :]
            # Initialize the identity matrix
            identity = np.identity(x_train_gaus.shape[1])
            # Train with the whole training data
            inverse = np.linalg.inv(model["lamda"] * identity + np.dot(x_train_gaus.T, x_train_gaus))
            data_term = np.dot(x_train_gaus.T, y_train)
            weights = np.dot(inverse, data_term)
            # Predict with the test data
            # Inverse/rescale for the z-score norm
            prediction = np.dot(x_test_gaus, weights)
        else:
            # Initialize the identity matrix
            identity = np.identity(x_train.shape[1])
            # Train with the whole training data
            inverse = np.linalg.inv(model["lamda"] * identity + np.dot(x_train.T, x_train))
            data_term = np.dot(x_train.T, y_train)
            weights = np.dot(inverse, data_term)
            # Predict with the test data
            # Inverse/rescale for the z-score norm
            prediction = np.dot(testX.values, weights)
        return prediction