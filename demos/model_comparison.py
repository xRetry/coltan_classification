import numpy as np
from core.dataset import Dataset
from core.mines import AggregationMine, BayesianUniMine
from core.models import MineModel
from core import normalizers, estimators
from core.functions import classification, transformation

from analysis.model import ModelAnalyser, CrossValParameters
from analysis import loss


def _create_params(n_attributes):
    mean_prior = np.ones(n_attributes) * 5
    std_prior = np.zeros(n_attributes)
    model_classes, model_params = ModelAnalyser.params_generator(
        ModelClass=MineModel,
        MineClass=[AggregationMine, AggregationMine, BayesianUniMine],
        EstimatorClass=estimators.MeanUniEstimator,
        func_eval=[AggregationMine.eval_norm2, AggregationMine.eval_ttest, BayesianUniMine.eval_pdf_predictive],
        func_transform=transformation.log,
        NormalizerClass=normalizers.NoNormalizer,
        func_classification=classification.max_score,
        threshold=0.5,
        mine_kwargs=[dict(), dict(), dict(loc=mean_prior, scale=std_prior, kappa=0, nu=-1)]
        # eval_kwargs=dict(func_aggr=np.mean)
    )
    return model_classes, model_params


def compare_models():
    datasets = [
        Dataset('/data/ctpa-data.csv', group_by_mine=False),
        #Dataset('/data/ctpa-data.csv', group_by_mine=True)
    ]

    for ds in datasets:
        model_classes, model_params = _create_params(ds.n_attributes)

        cv_params = CrossValParameters(
            ModelClass=model_classes,
            model_params=model_params,
            dataset=ds,
            pct_test=0.000001,
            func_loss=loss.accuracy
        )

        cv_result = ModelAnalyser.cross_validate(cv_params)
        print()
        print(cv_result.conf_ints)

        losses = cv_result.losses_per_fold
        pass


if __name__ == '__main__':
    compare_models()
