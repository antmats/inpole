from os.path import join

import pandas as pd
import numpy as np

from .models.utils import expects_groups
from .data.utils import (
    get_aggregation_index,
    check_fit_preprocessor,
    get_feature_names
)
from .models import (
    RiskSlimClassifier,
    FasterRiskClassifier,
    FRLClassifier,
    RuleFitClassifier,
    DecisionTreeClassifier
)
from .data import get_data_handler_from_config, is_treatment_switch
from .data.utils import drop_shifted_columns
from .pipeline import create_pipeline
from . import NET_ESTIMATORS, RECURRENT_NET_ESTIMATORS


ALL_NET_ESTIMATORS = NET_ESTIMATORS | RECURRENT_NET_ESTIMATORS


def is_net_estimator(estimator_name):
    return estimator_name in ALL_NET_ESTIMATORS


def train(config, estimator_name):
    pipeline = create_pipeline(config, estimator_name)
    preprocessor, estimator = pipeline.named_steps.values()

    data_handler = get_data_handler_from_config(config)

    if data_handler.aggregate_history:
        assert not expects_groups(estimator)

    data_train, data_valid, _ = data_handler.get_splits()
    X_train, y_train = data_train
    X_valid, y_valid = data_valid

    if estimator_name.startswith('truncated'):
        X_train = drop_shifted_columns(X_train)
        X_valid = drop_shifted_columns(X_valid)

    if not expects_groups(estimator) and not data_handler.aggregate_history:
        X_train = X_train.drop(columns=data_handler.GROUP)
        X_valid = X_valid.drop(columns=data_handler.GROUP)

    fit_params = {}

    if data_handler.aggregate_history:
        agg_index = get_aggregation_index(preprocessor, X_train, y_train)
        fit_params['preprocessor__feature_selector__aggregator__agg_index'] = agg_index
    
    if is_net_estimator(estimator_name) or isinstance(estimator, DecisionTreeClassifier):
        fit_params['estimator__X_valid'] = X_valid
        fit_params['estimator__y_valid'] = y_valid
    
    if isinstance(estimator, RiskSlimClassifier):
        feature_names = get_feature_names(preprocessor, X_train, y_train)
        outcome_name = data_handler.TREATMENT
        fit_params['estimator__feature_names'] = feature_names
        fit_params['estimator__outcome_name'] = outcome_name
    
    if isinstance(estimator, FasterRiskClassifier) or isinstance(estimator, RuleFitClassifier):
        feature_names = get_feature_names(preprocessor, X_train, y_train)
        fit_params['estimator__feature_names'] = feature_names
    
    if isinstance(estimator, FRLClassifier):
        preprocessor = check_fit_preprocessor(preprocessor, X_train, y_train)
        fit_params['estimator__preprocessor'] = preprocessor

    pipeline.fit(X_train, y_train, **fit_params)
    
    return pipeline


def collect_scores(model, X, y, metrics, columns=[], data=[], labels=None):
    if labels is None:
        labels = model.classes_.tolist()
    for metric, kwargs in metrics:
        score = model.score(X, y, metric=metric, **kwargs)
        if isinstance(score, np.ndarray):
            data.extend(score)
            for suffix in labels:
                columns.append(f'{metric}_{suffix}')
        else:
            data.append(score)
            columns.append(metric)
    return pd.Series(data=data, index=columns)


def predict(
    config,
    pipeline,
    estimator_name,
    subset,
    metrics,
    switches_only=False
):    
    data_handler = get_data_handler_from_config(config)
    _, data_valid, data_test = data_handler.get_splits()

    if subset == 'valid':
        X, y = data_valid
    elif subset == 'test':
        X, y = data_test
    
    if estimator_name.startswith('truncated'):
        X = drop_shifted_columns(X)
    
    if not expects_groups(pipeline[-1]) and not data_handler.aggregate_history:
       X = X.drop(columns=data_handler.GROUP)

    metrics = [
        (metric, {}) if isinstance(metric, str) else tuple(*metric.items())
        for metric in metrics
    ]

    columns = ['estimator_name', 'subset']
    data = [estimator_name, subset]
    labels = data_handler.get_labels()

    if switches_only:
        is_switch = is_treatment_switch(config, index=X.index)
        sample_weight = is_switch.astype(float)
        for _metric, kwargs in metrics:
            kwargs['sample_weight'] = sample_weight
        data[-1] += '_s'
    
    scores = collect_scores(pipeline, X, y, metrics, columns, data, labels)

    scores_file = join(config['results']['path'], 'scores.csv')
    try:
        df = pd.read_csv(scores_file)
        df = pd.concat([df, scores.to_frame().T], ignore_index=True)
        df.to_csv(scores_file, index=False)
    except FileNotFoundError:
        df = pd.DataFrame(scores)
        df.T.to_csv(scores_file, index=False)
