import os
import re
from pathlib import Path
from os.path import join

import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from seaborn._statistics import EstimateAggregator
from sklearn.decomposition import PCA
from amhelpers.config_parsing import load_config
from sklearn.tree._export import _MPLTreeExporter
from sklearn.tree._reingold_tilford import Tree
from pandas.api.types import is_categorical_dtype, is_bool_dtype

from inpole.pipeline import _get_estimator_params, load_best_pipeline
from inpole.models.models import get_model_complexity
from inpole.data import get_data_handler_from_config


__all__ = [
    'visualize_encodings',
    'get_params_and_scores',
    'get_model_complexities_and_scores',
    'plot_model_complexity',
    'plot_tree',
    'describe_categorical',
    'describe_numerical',
    'display_dataframe',
    'get_all_scores',
    'get_scoring_table',
    'compare_ra_models',
    'get_table_sections',
    'get_wide_table',
    'describe_dataset',
    'get_cpr_scores',
]


def visualize_encodings(
    encodings, prototype_indices, frac=0.1, figsize=(6,4), annotations=None,
    hue=None, hue_key=None, **kwargs
):
    
    pca = PCA(n_components=2).fit(encodings)
    
    # Transform the encodings.
    encodings_pca = pca.transform(encodings)
    encodings = {
        'PC 1': encodings_pca[:, 0],
        'PC 2': encodings_pca[:, 1],
        'Prototype': 'No',
        hue_key: hue
    }
    encodings = pd.DataFrame(encodings)
    
    # Sample a fraction of the encodings.
    encodings = encodings.sample(frac=frac, axis='index')
    
    # Transform the prototypes.
    prototypes_pca = encodings_pca[prototype_indices]
    prototypes_hue = hue[prototype_indices] if hue is not None else None
    prototypes = {
        'PC 1': prototypes_pca[:, 0], 
        'PC 2': prototypes_pca[:, 1],
        'Prototype': 'Yes',
        hue_key: prototypes_hue
    }
    prototypes = pd.DataFrame(prototypes)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    common_kwargs = {'x': 'PC 1', 'y': 'PC 2', 'ax': ax}
    if hue is not None:
        common_kwargs['hue'] = hue_key
    common_kwargs.update(kwargs)
    
    sns.scatterplot(
        data=encodings, alpha=0.7, size='Prototype', sizes=(20, 100),
        size_order=['Yes', 'No'], **common_kwargs
    )
    
    n_prototypes = prototypes_pca.shape[0]
    common_kwargs.pop('legend', None)
    sns.scatterplot(
        data=prototypes, alpha=1, s=n_prototypes*[100], legend=False,
        **common_kwargs
    )
    
    # Annotate the prototypes.
    for i, a in enumerate(prototypes_pca, start=1):
        try:
            xytext = (a[0]+annotations[i][0], a[1]+annotations[i][1])
            ax.annotate(
                i, xy=a, xytext=xytext, arrowprops={'arrowstyle': '-'}
            )
        except TypeError:
            ax.annotate(i, xy=(a[0]+0.1, a[1]))

    title = 'Prototype' if hue is None else None
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), title=title)

    return fig, ax


def get_params_and_scores(sweep_path, estimator_name, trials=None):
    params, scores = [], []

    if trials is None:
        # Get all sorted trial directories.
        trial_dirs = sorted(os.listdir(sweep_path))
    else:
        trial_dirs = [join(sweep_path, f'trial_{trial:02d}') for trial in trials]
    
    for trial_dir in trial_dirs:
        trial_path = join(sweep_path, trial_dir)
        
        for d in sorted(Path(trial_path).iterdir()):
            exp = str(d).split('/')[-1]  # `exp` is on the form estimator_XX
            name = '_'.join(exp.split('_')[:-1])  # Get the estimator name
            
            if name == estimator_name:
                try:
                    scores_path = join(d, 'scores.csv')
                    _scores = pd.read_csv(scores_path)
                    scores.append(_scores)
                except FileNotFoundError:
                    continue

                config_path = join(d, 'config.yaml')
                _config = load_config(config_path)
                _params = _get_estimator_params(_config, estimator_name)
                _params.pop('random_state', None)
                params.append(_params)
    
    return params, scores


def get_model_complexities_and_scores(
    trial_path, estimator_name, subset='test', metric='auc'
):
    complexities, scores = [], []

    for experiment_dir in os.listdir(trial_path):
        exp = experiment_dir.split('/')[-1]  # `exp` is on the form estimator_XX
        name = '_'.join(exp.split('_')[:-1])  # Get the estimator name
        
        if name == estimator_name:
            pipeline_path = join(trial_path, experiment_dir, 'pipeline.pkl')
            if os.path.exists(pipeline_path):
                pipeline = joblib.load(pipeline_path)
                estimator = pipeline.named_steps['estimator']
                complexity = get_model_complexity(estimator)
            else:
                complexity = np.nan
            complexities.append(complexity)
            
            scores_path = join(trial_path, experiment_dir, 'scores.csv')
            if os.path.exists(scores_path):
                s = pd.read_csv(scores_path)
                mask = s.subset == subset
                if name in ['rdt', 'truncated_rdt']:
                    mask &= s.estimator_name == f'{name}_aligned'
                score = s[mask][metric].item()
            else:
                score = np.nan
            scores.append(score)

    return np.array(complexities), np.array(scores)


def plot_model_complexity(inputs, subset='test', metric='auc'):
    for estimator, trial_path, bins, label, color, ax in inputs:
        complexities, scores = get_model_complexities_and_scores(
            trial_path, estimator, subset, metric
        )
        
        assert len(complexities) == len(scores)
        
        if len(complexities) > 0:
            if bins is not None:
                indices = np.digitize(complexities, bins)
                x, y = [], []
                xticks, xticklabels = [], []
                for i in range(1, len(bins)):
                    le, re = bins[i-1], bins[i]
                    m = le + (re - le) / 2
                    xticks.append(m)
                    xticklabels.append(f'{le}–{re}')
                    if i in indices:
                        s = max(scores[indices==i])
                        x.append(m)
                        y.append(s)
                ax.plot(x, y, 'ko-', color=color, label=label)
                ax.set_xticks(xticks)
                ax.set_xticklabels(xticklabels, rotation=90)
            else:
                unique = np.unique(complexities)
                unique = unique[~np.isnan(unique)]
                max_scores = [max(scores[complexities==x]) for x in unique]
                ax.plot(unique, max_scores, 'ko-', color=color, label=label)
                ax.set_xticks(unique)


class TreeExporter(_MPLTreeExporter):
    def __init__(
        self,
        max_depth=None,
        feature_names=None,
        class_names=None,
        filled=False,
        proportion=False,
        rounded=False,
        precision=3,
        fontsize=None,
        node_ids_to_include=None,
    ):
        self.node_ids_to_include = node_ids_to_include
        super().__init__(
            max_depth=max_depth,
            feature_names=feature_names,
            class_names=class_names,
            label='all',
            filled=filled,
            impurity=False,
            node_ids=False,
            proportion=proportion,
            rounded=rounded,
            precision=precision,
            fontsize=fontsize,
        )
    
    def _make_tree(self, node_id, et, criterion, depth=0):
        name = self.node_to_str(et, node_id, criterion=criterion)
        if (
            self.node_ids_to_include is not None
            and node_id not in self.node_ids_to_include
        ):
            name = self.node_to_str(et, node_id, criterion=criterion)
            if not name.startswith('samples'):
                splits = name.split('\n')
                splits[0] = 'null'
                name = '\n'.join(splits)
            return Tree(name, node_id)
        return super()._make_tree(node_id, et, criterion, depth)

    def recurse(self, node, tree, ax, max_x, max_y, depth=0):
        kwargs = dict(
            bbox=self.bbox_args.copy(),
            ha="center",
            va="center",
            zorder=100 - 10 * depth,
            xycoords="axes fraction",
            arrowprops=self.arrow_args.copy(),
        )
        kwargs["arrowprops"]["edgecolor"] = plt.rcParams["text.color"]

        if self.fontsize is not None:
            kwargs["fontsize"] = self.fontsize

        xy = ((node.x + 0.5) / max_x, (max_y - node.y - 0.5) / max_y)

        if self.max_depth is None or depth <= self.max_depth:
            if self.filled:
                kwargs["bbox"]["fc"] = self.get_fill_color(tree, node.tree.node_id)
            else:
                kwargs["bbox"]["fc"] = ax.get_facecolor()

            if node.parent is None:
                ax.annotate(node.tree.label, xy, **kwargs)
            else:
                xy_parent = (
                    (node.parent.x + 0.5) / max_x,
                    (max_y - node.parent.y - 0.5) / max_y,
                )
                if node.tree.label.startswith('null'):
                    kwargs["bbox"]["fc"] = "lightgrey"
                    label = node.tree.label.replace('null', '(...)')
                    ax.annotate(label, xy_parent, xy, **kwargs)
                else:
                    ax.annotate(node.tree.label, xy_parent, xy, **kwargs)
            if not node.tree.label.startswith('null'):
                for child in node.children:
                    self.recurse(child, tree, ax, max_x, max_y, depth=depth + 1)
        else:
            xy_parent = (
                (node.parent.x + 0.5) / max_x,
                (max_y - node.parent.y - 0.5) / max_y,
            )
            kwargs["bbox"]["fc"] = "lightgrey"
            ax.annotate("\n  (...)  \n", xy_parent, xy, **kwargs)


def plot_tree(
    decision_tree,
    max_depth=None,
    feature_names=None,
    filled=True,
    proportion=True,
    rounded=True,
    precision=2,
    fontsize=None,
    ax=None,
    node_ids_to_include=None,
    label_mapper={},
    formatter=None,
    annotate_arrows=False,
):
    exporter = TreeExporter(
        max_depth=max_depth,
        feature_names=feature_names,
        filled=filled,
        proportion=proportion,
        rounded=rounded,
        precision=precision,
        fontsize=fontsize,
        node_ids_to_include=node_ids_to_include,
    )
    annotations = exporter.export(decision_tree, ax=ax)

    if ax is None:
        ax = plt.gca()

    x0, y0 = annotations[0].get_position()
    x1, y1 = annotations[1].get_position()

    renderer = ax.figure.canvas.get_renderer()
    for annotation in annotations:
        text = annotation.get_text()
        if text.startswith('samples'):
            # Leaf node
            if formatter is not None:
                s, v = text.split('\n')
                s, v = formatter(s, v)
                text = '\n'.join([s, v])
        elif text.startswith('\n'):
            # (...)
            pass
        else:
            # Inner node
            l, s, v = text.split('\n')
            if l in label_mapper:
                l = label_mapper[l]
            elif re.match(r'\w+\s+<=\s+\w+', l):
                l1, l2 = l.split(' <= ')
                l1 = label_mapper.get(l1, l1)
                l2 = float(l2)
                l = l1 + ' $\leq$ ' + '{:.{prec}f}'.format(l2, prec=precision)
            if formatter is not None:
                s, v = formatter(s, v)
            text = '\n'.join([l, s, v])
        annotation.set_text(text)
        annotation.set(ha='center')
        annotation.draw(renderer)

    if annotate_arrows:
        kwargs = dict(ha='center', va='center', fontsize=fontsize)
        ax.annotate('True', (x1 + (x0-x1) / 2, y0 - (y0-y1) / 3), **kwargs)
        ax.annotate('False', (x0 + (x0-x1) / 2, y0 - (y0-y1) / 3), **kwargs)


def describe_categorical(D):
    assert (D.apply(is_categorical_dtype) | D.apply(is_bool_dtype)).all()
    
    index_tuples = []
    out = {'Counts': [], 'Proportions': []}

    for v in D:
        s = D[v]

        if is_bool_dtype(s):
            s = s.astype('category')
            s = s.cat.rename_categories({True: 'yes', False: 'no'})
        
        table = pd.Categorical(s).describe()
        
        # Exclude NaNs when computing proportions.
        N = table.counts.values[:-1].sum() if -1 in table.index.codes \
            else table.counts.values.sum()
        proportion = [round(100 * x, 1) for x in table.counts / N]
        table.insert(1, 'proportion', proportion)

        try:
            from inpole.data.corevitas import COREVITAS_DATA
            categories = COREVITAS_DATA.variables[s.name].pop('categories', None)
        except ModuleNotFoundError:
            categories = None
        except KeyError:
            categories = None
        
        if categories is not None:
            table.index = table.index.rename_categories(
                categories
            )

        for c in table.index:
            index_tuples.append((v, N, c))
        out['Counts'].extend(table.counts)
        out['Proportions'].extend(table.proportion)

    index = pd.MultiIndex.from_tuples(
        index_tuples, names=['Variable', 'No. samples', 'Value']
    )
    return pd.DataFrame(out, index=index)


def describe_numerical(data):
    table = data.describe().T
    table.rename(columns={'count': 'No. samples'}, inplace=True)
    return table.drop(
        columns=['mean', 'std', 'min', 'max']
    )


def display_dataframe(
    df,
    caption=None,
    new_index=None,
    new_columns=None,
    hide_index=False,
    precision=2
):
    def set_style(styler):
        if caption is not None:
            styler.set_caption(caption)
        if new_index is not None:
            styler.relabel_index(new_index, axis=0)
        if new_columns is not None:
            styler.relabel_index(new_columns, axis=1)
        if hide_index:
            styler.hide(axis='index')
        styler.format(precision=precision)
        return styler
    
    display_everything = (
        'display.max_rows', None,
        'display.max_columns', None,
    )
    
    with pd.option_context(*display_everything):
        return df.style.pipe(set_style)


def get_all_scores(all_experiment_paths):
    all_scores = []
    
    for experiment, experiment_paths in all_experiment_paths.items():
        if experiment_paths is None:
            continue
        for state, experiment_path in experiment_paths:
            if experiment_path is None:
                continue
            scores_path = os.path.join(experiment_path, 'scores.csv')
            if not os.path.exists(scores_path):
                print(f"No scores available for {experiment_path}.")
                continue
            scores = pd.read_csv(scores_path)
    
            # Load a pipeline to compute the dimensionality of the input.
            estimator = 'rnn' if state == '$H_t$' else 'lr'
            pipeline, results_path = load_best_pipeline(
                experiment_path, 1, estimator, return_results_path=True)
    
            # Load data handler.
            config_path = os.path.join(results_path, 'config.yaml')
            config = load_config(config_path)
            data_handler = get_data_handler_from_config(config)
            
            scores['data'] = experiment
            scores['state'] = state
            if r'\bar{H}_t' in state:
                scores['reduction'] = data_handler.reduction
            else:
                scores['reduction'] = 'none'
            state_dim = pipeline.n_features_in_
            if data_handler.GROUP in pipeline.feature_names_in_:
                state_dim -= 1
            scores['state_dim'] = state_dim
            all_scores.append(scores)
    
    all_scores = pd.concat(all_scores)
    all_scores.rename(columns={'estimator_name': 'estimator'}, inplace=True)
    
    return all_scores


def get_scoring_table(
    all_scores,
    groupby=['data', 'state', 'reduction', 'state_dim', 'estimator'],
    metric='auc',
    include_cis=False,
    exclude_models=[],
    experiment_order=None,
    model_order=None,
    index=None,
):
    g = all_scores[all_scores.subset == 'test'].groupby(groupby)
    
    agg = EstimateAggregator(np.mean, 'ci', n_boot=1000, seed=0)
    
    table = g.apply(agg, var=metric)
    table = table * 100  # Convert to percentage

    if include_cis:
        a = r'\begin{tabular}[c]{@{}c@{}}'
        b = r'\end{tabular}'
        f = lambda r: a + rf"{r[metric]:.1f}\\({r[f'{metric}min']:.1f}, {r[f'{metric}max']:.1f})" + b
    else:
        f = lambda r: f'{r[metric]:.1f}'
    table[metric] = table[[metric, f'{metric}min', f'{metric}max']].apply(f, axis=1)
    table = table.drop(columns=[f'{metric}min', f'{metric}max'])
    
    table = table.unstack(-1)
    table.columns = table.columns.droplevel()  # Drop metric level
    
    sequence_models = ['prosenet', 'rdt', 'rdt_aligned', 'rdt_pruned', 'rnn']
    for c in sequence_models:
        c_truncated = f'truncated_{c}'
        if c_truncated in table:
            table[c].fillna(table[c_truncated], inplace=True)
            table.drop(columns=c_truncated, inplace=True)
    
    exclude_models = [m for m in exclude_models if m in table.columns]
    table = table.drop(columns=exclude_models)
        
    table = table.fillna('-')

    if experiment_order is not None:
        table = table.reindex(experiment_order, level=0)

    if model_order is not None:
        table = table[[m for m in model_order if m in table.columns]]
        table = table.rename(columns=model_order)
    
    if index is not None:
        table = table.reindex(index, level=1)

    return table


def compare_ra_models(
    probas,
    data_path,
    state_true='$H_t$',
    estimator_true='rnn',
    state_pred='$A_{t-1}$',
    estimator_pred='lr',
    num_trials=5,
    switches_only=False,
    compare_with_gt=False,
    return_proba=False,
):
    from inpole.data import RAData
    
    y_true_all, y_pred_all = [], []
    yp_true_all, yp_pred_all = [], []
    X_all, y_all = [], []
    
    for trial in range(1, num_trials + 1):
        data_handler = RAData(path=data_path, seed=trial)
        _train_data, _valid_data, test_data = data_handler.get_splits()
        X, y = test_data
        X['stage'] = X.groupby(data_handler.GROUP).cumcount() + 1
        
        yp_true = probas[
            probas.State.eq(state_true)
            & probas.Estimator.eq(estimator_true) 
            & probas.Trial.eq(trial)
        ].Probas.item()
    
        yp_pred = probas[
            probas.State.eq(state_pred) 
            & probas.Estimator.eq(estimator_pred) 
            & probas.Trial.eq(trial)
        ].Probas.item()
    
        y_true = np.argmax(yp_true, axis=1)
        y_pred = np.argmax(yp_pred, axis=1)

        if compare_with_gt:
            labels = data_handler.get_labels()
            yp_true = np.eye(len(labels))[y]
            y_true = y

        if switches_only:
            labels = data_handler.get_labels()
            switch = (labels[y] != X.prev_therapy)
            
            yp_true = yp_true[switch]
            yp_pred = yp_pred[switch]
            
            y_true = y_true[switch]
            y_pred = y_pred[switch]
            
            X = X[switch]
            y = y[switch]
        
        X['correct'] = y_true == y_pred

        yp_true_all.append(yp_true)
        yp_pred_all.append(yp_pred)
        
        y_true_all.append(y_true)
        y_pred_all.append(y_pred)
        
        X_all.append(X)
        y_all.append(y)

    yp_true_all = np.concatenate(yp_true_all)
    yp_pred_all = np.concatenate(yp_pred_all)
    
    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    
    X_all = pd.concat(X_all)
    y_all = np.concatenate(y_all)

    _, unique_indices = np.unique(X_all.index, return_index=True)
    unique_indices = np.sort(unique_indices)

    yp_true_all = yp_true_all[unique_indices]
    yp_pred_all = yp_pred_all[unique_indices]
    
    y_true_all = y_true_all[unique_indices]
    y_pred_all = y_pred_all[unique_indices]

    X_all = X_all.iloc[unique_indices]
    y_all = y_all[unique_indices]

    if return_proba:            
        return y_true_all, y_pred_all, yp_true_all, yp_pred_all, X_all, y_all
    else:
        return y_true_all, y_pred_all, X_all, y_all


def get_table_sections(table, sequence_models):
    """Get two table sections, one for sequence models and one for non-sequence models."""

    # Remove sequence models.
    table1 = table.drop(columns=sequence_models)
    table1 = table1[table1.index.get_level_values('state') != '$H_t$']
    table1 = get_wide_table(table1)

    # Keep sequence models.
    table2 = table.drop(columns=[c for c in table.columns if not c in sequence_models])
    table2 = table2[table2.index.get_level_values('state') == '$H_t$']
    table2 = get_wide_table(table2)

    return table1, table2


def get_wide_table(table):
    assert isinstance(table.index, pd.MultiIndex)
    assert table.index.names[0] == 'data'
    subtables = []
    for experiment in table.index.get_level_values('data').unique():
        subtable = table.xs(experiment, level='data')
        subtable = subtable.replace('-', np.nan).dropna(axis=0, how='all')
        subtable = subtable.replace('-', np.nan).dropna(axis=1, how='all')
        subtable.columns = pd.MultiIndex.from_product([[experiment], subtable.columns])
        subtables.append(subtable)
    return pd.concat(subtables, axis=1)


def describe_dataset(data, c_group, c_age, c_gender, i_female, v_female=None):
    if i_female is None:
        assert data[c_gender].dtype == 'category' and v_female is not None
    data_grouped = data.groupby(c_group)

    n_patients = data_grouped.ngroups

    age_bl = data_grouped[c_age].first()
    age_bl_median = age_bl.median()
    age_bl_iqr = (age_bl.quantile(0.25), age_bl.quantile(0.75))

    gender_bl = data_grouped[c_gender].first()
    if i_female is None:
        i_female = data[c_gender].cat.categories.get_loc(v_female)
    female_bl_count = gender_bl.value_counts(sort=False).iloc[i_female]
    female_bl_pct = 100 * gender_bl.value_counts(sort=False, normalize=True).iloc[i_female]

    n_stages_median = data_grouped.size().median()
    n_stages_iqr = (data_grouped.size().quantile(0.25), data_grouped.size().quantile(0.75))

    print(f"Patients, n: {n_patients}")
    print(f"Age in years, median (IQR): {age_bl_median:.1f} ({age_bl_iqr[0]:.1f}, {age_bl_iqr[1]:.1f})")
    print(f"Female, n (%): {female_bl_count} ({female_bl_pct:.1f})")
    print(f"Stages, median IQR): {n_stages_median:.1f} ({n_stages_iqr[0]:.1f}, {n_stages_iqr[1]:.1f})")


def get_cpr_scores(results_path):
    from sklearn.metrics import roc_auc_score
    from amhelpers.metrics import ece as ece_score

    all_scores = []

    def get_auc_and_ece(root, file):
        data = joblib.load(os.path.join(root, file))
        return (
            roc_auc_score(data['true'], data['preds']),
            ece_score(data['true'].flatten(), data['preds']),
        )

    for root, _, files in os.walk(results_path):
        for file in files:
            if 'rnn' in file.lower() and file.endswith('pkl'):
                auc, ece = get_auc_and_ece(root, file)
                all_scores.append(['RNN', auc, ece])

            if 'lstm' in file.lower() and file.endswith('pkl'):
                auc, ece = get_auc_and_ece(root, file)
                all_scores.append(['LSTM', auc, ece])

    all_scores = pd.DataFrame(all_scores, columns=['Encoder', 'AUROC', 'ECE'])

    return all_scores
