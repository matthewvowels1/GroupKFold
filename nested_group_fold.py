import numpy as np

N = 50
y = np.random.randn(N, 1)
X = np.random.randn(N, 6)
group = np.round(np.random.uniform(0,24, N)).reshape(-1,1)
k = 5
nest_k = 4


def RandomGroupKFold_split(groups, n, seed=0):  # noqa: N802
    """
    Random analogous of sklearn.model_selection.GroupKFold.split.

    :return: list of (train, test) indices
    """
    groups = pd.Series(groups)
    ix = np.arange(len(groups))
    unique = np.unique(groups)
    np.random.RandomState(seed).shuffle(unique)
    result = []
    for split in np.array_split(unique, n):
        mask = groups.isin(split)
        train, test = ix[~mask], ix[mask]
        result.append((train, test))

    return result


k = 5
nest_k = 4
runs = 6
seed = 0 
for i in range(runs):
    seed += 1
    indices = np.arange(len(X))
    incides = np.random.choice(indices, len(X))
    
    X = X[indices]
    y = y[indices]
    group = group[indices]
    
    
    group_kfold = RandomGroupKFold_split(groups=group[:,0], n=k, seed=seed)

    for train_index, test_index in group_kfold:

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        g_train, g_test = group[train_index], group[test_index]

        nested_group_kfold = RandomGroupKFold_split(groups=g_train[:,0], n=nest_k, seed=seed)

        for nest_train_index, nest_test_index in nested_group_kfold:

            X_train_nest, X_test_nest = X_train[nest_train_index], X_train[nest_test_index]
            y_train_nest, y_test_nest = y_train[nest_train_index], y_train[nest_test_index]
            g_train_nest, g_test_nest = g_train[nest_train_index], g_train[nest_test_index]


