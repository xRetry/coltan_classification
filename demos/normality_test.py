from analysis.dataset import DatasetAnalyser, Dataset
from core.functions import transformation


def test_normality():
    """
    Performs an Anderson-Darling normality test for all samples and attributes in the dataset and plots the result.
    """
    dataset = Dataset()
    DatasetAnalyser(dataset).test_normality(title='Without Transformation')
    DatasetAnalyser(dataset).test_normality(func_trans=transformation.log, title='Log Transformed')


if __name__ == '__main__':
    test_normality()
