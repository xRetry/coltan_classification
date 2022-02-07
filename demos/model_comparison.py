import numpy as np

from core.dataset import Dataset
from core.mines import AggregationMine, BayesianUniMine
from core.models import MineModel
from core import normalizers, estimators
from core.functions import classification, transformation
from analysis.model import ModelAnalyser, CrossValParameters
from analysis import loss
import analysis.plotting.model as plotting


def _create_params(n_attributes):
    mean_prior = np.ones(n_attributes) * 5
    std_prior = np.ones(n_attributes)
    model_classes, model_params = ModelAnalyser.params_generator(
        ModelClass=MineModel,
        MineClass=[AggregationMine, AggregationMine, BayesianUniMine, BayesianUniMine],
        EstimatorClass=estimators.MeanUniEstimator,
        func_eval=[AggregationMine.eval_norm2, AggregationMine.eval_ttest, BayesianUniMine.eval_pdf_predictive, BayesianUniMine.eval_ztest],
        func_transform=transformation.log,
        NormalizerClass=normalizers.NoNormalizer,
        func_classification=classification.max_score,
        threshold=0.5,
        mine_kwargs=[dict(), dict(), dict(loc=mean_prior, scale=std_prior, kappa=0, nu=-1), dict(loc=mean_prior, scale=std_prior, kappa=0, nu=-1)],
        eval_kwargs=[dict()] + [dict(func_aggr=np.product)]*3
    )
    return model_classes, model_params


def compare_models():
    """
    Compares the performance of predefined models on the Coltan dataset.
    """
    # Create list of all datasets to be used
    datasets = [
        Dataset('/data/ctpa-data.csv', group_by_mine=False),
        #Dataset('/data/ctpa-data.csv', group_by_mine=True)
    ]
    # Compare all models for every dataset
    for ds in datasets:
        # Create model parameters
        model_classes, model_params = _create_params(ds.n_attributes)
        # Create cross-val parameters
        cv_params = CrossValParameters(
            ModelClass=model_classes,
            model_params=model_params,
            dataset=ds,
            pct_test=1e-6,
            func_loss=loss.accuracy
        )
        # Cross-validate all models on the current dataset
        cv_result = ModelAnalyser.cross_validate(cv_params)
        # Create list of correct and wrong predictions
        accs = [(np.concatenate(p) == np.concatenate(l)).astype(int) for p, l in zip(cv_result.predictions, cv_result.labels)]
        # Plot result
        plotting.plot_crossval_distributions(accs, cv_result.conf_ints, ['L2 Norm', 't-Test', 'PDF Predictive'])


if __name__ == '__main__':
    compare_models()
