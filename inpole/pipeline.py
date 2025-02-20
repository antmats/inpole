import copy
from os.path import join

import torch
import joblib
import pandas as pd
import sklearn.pipeline as pipeline
from sklearn.utils import _print_elapsed_time
from amhelpers.amhelpers import seed_hash

from .models import hparam_registry as hprm
from .data.utils import *
from .data import get_data_handler_from_config
from . import (
    NET_ESTIMATORS,
    RECURRENT_NET_ESTIMATORS,
    OTHER_ESTIMATORS
)


ALL_NET_ESTIMATORS = NET_ESTIMATORS | RECURRENT_NET_ESTIMATORS
CPLEX_PARAM_VALUE_BOUND = 20e8


continuous_feature_transformation = {
    'dt': None,
    'dummy': None,
    'fasterrisk': 'discretize',
    'frl': 'discretize',
    'lr': 'scale',
    'mlp': 'scale',
    'pronet': 'scale',
    'prosenet': 'scale',
    'rdt': 'discretize',
    'rnn': 'scale',
    'lstm': 'scale',
    'rulefit': 'scale',
    'riskslim': 'discretize',
    'sdt': 'discretize',
    'truncated_prosenet': 'scale',
    'truncated_rdt': 'discretize',
    'truncated_rnn': 'scale',
}


_default_net_params = {
    #module
    'criterion': torch.nn.CrossEntropyLoss,
    'optimizer': torch.optim.Adam,
    #lr
    #max_epochs
    #batch_size
    'iterator_train': torch.utils.data.DataLoader,
    'iterator_train__shuffle': True,
    'iterator_valid': torch.utils.data.DataLoader,
    'dataset': StandardDataset,
    'train_split': None,
    'callbacks': None,
    'predict_nonlinearity': 'auto',
    'warm_start': False,
    'verbose': 1,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}


def _get_estimator_params(config, estimator_name, input_dim=None, output_dim=None):
    experiment = config['experiment']
    hparams_seed = config['hparams']['seed']
    if hparams_seed == 0:
        hparams = hprm.default_hparams(estimator_name, experiment)
    else:
        data_seed = config['data']['seed']
        seed = seed_hash(hparams_seed, data_seed)
        hparams = hprm.random_hparams(estimator_name, experiment, seed)
    
    if estimator_name in ALL_NET_ESTIMATORS:
        params = copy.deepcopy(_default_net_params)        
        params.update(
            {
                'results_path': config['results']['path'],
                'seed': config['estimators']['seed'],
                'module__input_dim': input_dim,
                'module__output_dim': output_dim,
            }
        )
    else:
        seed = config['estimators']['seed']
        if estimator_name == 'riskslim' and seed > CPLEX_PARAM_VALUE_BOUND:
            seed = int(seed - CPLEX_PARAM_VALUE_BOUND)
        params = {'random_state': seed}
    
    params.update(hparams)

    if estimator_name in config['estimators']:
        params.update(config['estimators'][estimator_name])

    return params


def _create_estimator(
    config,
    estimator_name,
    **kwargs
):
    params = _get_estimator_params(config, estimator_name, **kwargs)

    if estimator_name in NET_ESTIMATORS:
        return NET_ESTIMATORS[estimator_name](**params)
    
    if estimator_name in RECURRENT_NET_ESTIMATORS:
        if estimator_name.startswith('truncated'):
            params['dataset'] = TruncatedHistoryDataset
            params['dataset__periods'] = config['data']['shift_periods']
        else:
            params['dataset'] = SequentialDataset
        params['iterator_train__collate_fn'] = pad_pack_sequences
        params['iterator_valid__collate_fn'] = pad_pack_sequences
        return RECURRENT_NET_ESTIMATORS[estimator_name](**params)
    
    if estimator_name in OTHER_ESTIMATORS:
        return OTHER_ESTIMATORS[estimator_name](**params)


class Pipeline(pipeline.Pipeline):
    def __init__(self, steps, *, memory=None, verbose=False):
        super().__init__(steps, memory=memory, verbose=verbose)

    def fit(self, X, y=None, **fit_params):
        fit_params_steps = self._check_fit_params(**fit_params)
        Xt = self._fit(X, y, **fit_params_steps)
        with _print_elapsed_time('Pipeline', self._log_message(len(self.steps) - 1)):
            if self._final_estimator != 'passthrough':
                fit_params_last_step = fit_params_steps[self.steps[-1][0]]
                Xt_valid = fit_params_last_step.get('X_valid', None)
                if Xt_valid is not None:
                    for _, _, transform in self._iter(with_final=False):
                        Xt_valid = transform.transform(Xt_valid)
                    fit_params_last_step['X_valid'] = Xt_valid
                self._final_estimator.fit(Xt, y, **fit_params_last_step)
        return self
    
    def score(self, X, y=None, **score_params):
        estimator = self.steps[-1][1]
        if isinstance(X, torch.utils.data.Dataset):
            return estimator.score(X, y, **score_params)
        for _, _, transform in self._iter(with_final=False):
            X = transform.transform(X)
        return estimator.score(X, y, **score_params)


def create_pipeline(config, estimator_name):
    data_handler = get_data_handler_from_config(config)

    # Should this seed depend on the estimator?
    seed = config['hparams']['seed']
    cont_feat_trans = continuous_feature_transformation[estimator_name]
    preprocessor = data_handler.get_preprocessor(cont_feat_trans, seed)

    if estimator_name in ALL_NET_ESTIMATORS:
        # Infer input/output dimensions from training data.
        X_train, y_train = data_handler.get_splits()[0]
        if estimator_name.startswith('truncated'):
            X_train = drop_shifted_columns(X_train)
        input_dim = len(get_feature_names(preprocessor, X_train, y_train))
        output_dim = len(set(y_train))
    else:
        input_dim = output_dim = None
    
    estimator = _create_estimator(
        config,
        estimator_name,
        input_dim=input_dim,
        output_dim=output_dim
    )

    steps = [('preprocessor', preprocessor), ('estimator', estimator)]
    return Pipeline(steps)
    

def load_best_pipeline(
    experiment_path,
    trial,
    estimator_name,
    sweep_parameter_value=None,
    return_results_path=False
):
    scores_path = join(experiment_path, 'scores.csv')
    scores = pd.read_csv(scores_path)
    if sweep_parameter_value is None:
        sweep = 'sweep'
        mask = scores.trial.eq(trial) & scores.estimator_name.eq(estimator_name)
    else:
        sweep_parameter = scores.columns[0]
        sweep = f'sweep_{sweep_parameter}_{sweep_parameter_value}'
        mask = (
            scores[sweep_parameter].eq(sweep_parameter_value)
            & scores.trial.eq(trial)
            & scores.estimator_name.eq(estimator_name)
        )
    if mask.sum() == 0:
        raise FileNotFoundError(
            f"No results found for model '{estimator_name}' in trial {trial}."
        )
    experiment = scores[mask].exp.iat[0]
    results_path = join(experiment_path, sweep, f'trial_{trial:02d}', experiment)
    pipeline_path = join(results_path, 'pipeline.pkl')
    pipeline = joblib.load(pipeline_path)
    if return_results_path:
        return pipeline, results_path
    return pipeline
