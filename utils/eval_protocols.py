import numpy as np
from sklearn.linear_model import Ridge
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score, f1_score


def fit_svm(features, y, MAX_SAMPLES=100000):
    nb_classes = np.unique(y, return_counts=True)[1].shape[0]
    train_size = features.shape[0]

    svm = SVC(C=np.inf, gamma="scale")
    if train_size // nb_classes < 5 or train_size < 50:
        return svm.fit(features, y)
    else:
        grid_search = GridSearchCV(
            svm,
            {
                "C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, np.inf],
                "kernel": ["rbf"],
                "degree": [3],
                "gamma": ["scale"],
                "coef0": [0],
                "shrinking": [True],
                "probability": [False],
                "tol": [0.001],
                "cache_size": [200],
                "class_weight": [None],
                "verbose": [False],
                "max_iter": [10000000],
                "decision_function_shape": ["ovr"],
                "random_state": [None],
            },
            cv=5,
            n_jobs=5,
        )
        # If the training set is too large, subsample MAX_SAMPLES examples
        if train_size > MAX_SAMPLES:
            split = train_test_split(
                features, y, train_size=MAX_SAMPLES, random_state=0, stratify=y
            )
            features = split[0]
            y = split[2]

        grid_search.fit(features, y)
        return grid_search.best_estimator_


def fit_lr(features, y, MAX_SAMPLES=100000):
    # If the training set is too large, subsample MAX_SAMPLES examples
    if features.shape[0] > MAX_SAMPLES:
        split = train_test_split(
            features, y, train_size=MAX_SAMPLES, random_state=0, stratify=y
        )
        features = split[0]
        y = split[2]

    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(random_state=0, max_iter=1000000, multi_class="ovr"),
    )
    pipe.fit(features, y)
    return pipe


def fit_knn(features, y):
    pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1))
    pipe.fit(features, y)
    return pipe


def fit_ridge(train_features, train_y, valid_features, valid_y, MAX_SAMPLES=100000):
    # If the training set is too large, subsample MAX_SAMPLES examples
    if train_features.shape[0] > MAX_SAMPLES:
        split = train_test_split(
            train_features, train_y, train_size=MAX_SAMPLES, random_state=0
        )
        train_features = split[0]
        train_y = split[2]
    if valid_features.shape[0] > MAX_SAMPLES:
        split = train_test_split(
            valid_features, valid_y, train_size=MAX_SAMPLES, random_state=0
        )
        valid_features = split[0]
        valid_y = split[2]

    alphas = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    valid_results = []
    for alpha in alphas:
        lr = Ridge(alpha=alpha).fit(train_features, train_y)
        valid_pred = lr.predict(valid_features)
        score = (
            np.sqrt(((valid_pred - valid_y) ** 2).mean())
            + np.abs(valid_pred - valid_y).mean()
        )
        valid_results.append(score)
    best_alpha = alphas[np.argmin(valid_results)]

    lr = Ridge(alpha=best_alpha)
    lr.fit(train_features, train_y)
    return lr


def eval_classification(model, train_data, train_labels, test_data, test_labels, eval_protocol='linear'):
    assert train_labels.ndim == 1 or train_labels.ndim == 2
    train_repr = model.encode(train_data, encoding_window='full_series' if train_labels.ndim == 1 else None)
    test_repr = model.encode(test_data, encoding_window='full_series' if train_labels.ndim == 1 else None)

    if eval_protocol == 'linear':
        fit_clf = fit_lr
    elif eval_protocol == 'svm':
        fit_clf = fit_svm
    elif eval_protocol == 'knn':
        fit_clf = fit_knn
    elif eval_protocol == 'ridge':
        fit_clf = fit_ridge
    else:
        assert False, 'unknown evaluation protocol'

    def merge_dim01(array):
        return array.reshape(array.shape[0] * array.shape[1], *array.shape[2:])

    if train_labels.ndim == 2:
        train_repr = merge_dim01(train_repr)
        train_labels = merge_dim01(train_labels)
        test_repr = merge_dim01(test_repr)
        test_labels = merge_dim01(test_labels)

    if eval_protocol == 'ridge':
        clf = fit_clf(train_repr, train_labels,test_repr,test_labels)
    else:
        clf = fit_clf(train_repr, train_labels)


    acc = clf.score(test_repr, test_labels)
    y_pred = clf.predict(test_repr)
    # Calculate macro F1 score
    macro_f1 = f1_score(test_labels, y_pred, average='macro')

    if eval_protocol == 'linear':
        y_score = clf.predict_proba(test_repr)
    else:
        y_score = clf.decision_function(test_repr)

    test_labels_onehot = label_binarize(test_labels, classes=np.arange(train_labels.max() + 1))
    auprc = average_precision_score(test_labels_onehot, y_score)

    return y_score, {'Acc': acc, 'averagePre': auprc, 'Macro-F1': macro_f1}
