# Import Pandas
import pandas as pd

# K FOld
from sklearn.cross_validation import KFold


class DataProcessing:
    """Data Processing opertation

    Processing of data such as Normalization, spliting the data,
    folding the data can be performed on the pandas dataframe

    Arguments:
        Pandas dataframe that need to processed
    """

    def __init__(self):
        """Variable Initialization

        Initialize the class variables which can be accessed using
        the self or class objects

        Arguments:
            None
        Returns:
             None
        """
        pass

    def z_score_norm(self, data):
        """Normalize the data using z-score

        Normalize the data which is mostly a pandas dataframe and
        outputs normalized dataframe of the same using z-score
        normalization.https://en.wikipedia.org/wiki/Standard_score

        Arguments:
            data: Pandas Dataframe to be normalized

        Returns:
            Normalized Pandas Dataframe
        """
        # Finding the z-score using mean and standard deviation of each column
        # Broadcasting is applied over the data
        normalized_df = (data - data.mean()) / data.std()
        return pd.DataFrame(normalized_df)

    def train_test_split(self, data, cut_at=0.75):
        """ Data is split into train and test set

        Split the data set into two random chuncks of train and test set
        using the cut_at value. The cut_at value literally means how much
        percentage of data you want to be in the training set and rest
        of those goes to the testing set.

        Arguments:
            data: Pandas Dataframe to be split into train and test
            cut_at: Threshold for what percentage of data needs top be
                in the training set. Ranges from 0 to 1. Default to 0.75

        Returns:
            train: Training set of the data
            test: Testing set of the data
        """
        # Random sample of the train set is acquired
        train = data.sample(frac=cut_at, random_state=200)
        # Rest are put in the test set
        test = data.drop(train.index)
        return train, test

    def pred_target_split(self, data, noutputs=1):
        """ Split the Predictors and output variables

        The Input data variable is take and predictors and n-output vectors
        are drawn out of the complete data. Note that this function works only
        when the output varibles are end/last column of the data.

        Arguments:
            data: Pandas Dataframe to be split into Predictors and target
            noutputs: Number of output variables in the dataset. Default to 1

        Returns:
            predictors: Predictors of the data
            outputs: Output variable of the data
        """
        # Subsetting the predictors
        predictors = data.iloc[:, 0:-noutputs]
        # Subsetting the outputs
        outputs = data.iloc[:, -noutputs:]
        return predictors, outputs

    def k_fold(self, data, k=7):
        """ Create K folds of the data

        Data is split into K number of chunks depending on the k argument of
        the function that can be used up for the cross validation purpose.

        Arguments:
            data: Pandas Dataframe to be split into K Folds
            k: Number of folds. Default to 7

        Returns:
            index: Row index for each train and test set of the folds
        """
        # Creating K-Fold cross validation index.
        index = KFold(len(data), n_folds=k)
        return index
