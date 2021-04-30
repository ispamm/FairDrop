import numpy as np
import torch
from sklearn.multioutput import MultiOutputClassifier
from torch_sparse import SparseTensor
from sklearn.metrics import (
    roc_auc_score,
    make_scorer,
    balanced_accuracy_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection, pipeline, metrics

# Metrics
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
)
from itertools import combinations_with_replacement


def encode_classes(col):
    """
    Input:  categorical vector of any type
    Output: categorical vector of int in range 0-num_classes
    """
    classes = set(col)
    classes_dict = {c: i for i, c in enumerate(classes)}
    labels = np.array(list(map(classes_dict.get, col)), dtype=np.int32)
    return labels


def onehot_classes(col):
    """
    Input:  categorical vector of int in range 0-num_classes
    Output: one-hot representation of the input vector
    """
    col2onehot = np.zeros((col.size, col.max() + 1), dtype=float)
    col2onehot[np.arange(col.size), col] = 1
    return col2onehot


def get_edge_embeddings(z, edge_index):
    return z[edge_index[0]] * z[edge_index[1]]


def get_link_labels(pos_edge_index, neg_edge_index):
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float)
    link_labels[: pos_edge_index.size(1)] = 1.0
    return link_labels


def train_n2v(model, loader, optimizer, device):
    model.train()

    total_loss = 0

    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()

        loss = model.loss(pos_rw.to(device), neg_rw.to(device))

        loss.backward()

        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def train_rn2v(
    model, loader, optimizer, device, pos_edge_index_tr, y_aux, round1, round2, N
):

    keep = torch.where(round1, y_aux, round2)

    row, col = pos_edge_index_tr[:, keep]
    model.adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N)).to("cpu")

    model.train()

    total_loss = 0

    for pos_rw, neg_rw in loader:

        optimizer.zero_grad()

        loss = model.loss(pos_rw.to(device), neg_rw.to(device))

        loss.backward()

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def train_rn2v_adaptive(
    model, loader, optimizer, device, pos_edge_index_tr, y_aux, rand, N
):

    keep = torch.where(rand, y_aux, ~y_aux)

    row, col = pos_edge_index_tr[:, keep]
    model.adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N)).to("cpu")

    model.train()

    total_loss = 0

    for pos_rw, neg_rw in loader:

        optimizer.zero_grad()

        loss = model.loss(pos_rw.to(device), neg_rw.to(device))

        loss.backward()

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def emb_fairness(XB, YB):
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
        XB, YB, test_size=0.3, stratify=YB
    )

    log = model_selection.GridSearchCV(
        pipeline.Pipeline(
            [
                (
                    "logi",
                    LogisticRegression(
                        multi_class="multinomial", solver="saga", max_iter=9000
                    ),
                )
            ]
        ),
        param_grid={"logi__C": [1, 10, 100]},
        cv=4,
        scoring="balanced_accuracy",
    )

    mlp = model_selection.GridSearchCV(
        pipeline.Pipeline(
            [
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32), solver="adam", max_iter=1000
                    ),
                )
            ]
        ),
        param_grid={
            "mlp__alpha": [0.001, 0.0001, 0.00001],
            "mlp__learning_rate_init": [0.01, 0.001],
        },
        cv=4,
        scoring="balanced_accuracy",
    )

    rf = model_selection.GridSearchCV(
        pipeline.Pipeline([("rf", RandomForestClassifier())]),
        param_grid={"rf__max_depth": [2, 4]},
        cv=4,
        scoring="balanced_accuracy",
    )

    c_dict = {
        "LogisticRegression": log,
        "MLPClassifier": mlp,
        "RandomForestClassifier": rf,
    }
    r_dict = {"RB EMB": []}
    for name, alg in c_dict.items():
        print(f"Evaluating RB with: {name}")
        alg.fit(X_train, Y_train)
        clf = alg.best_estimator_
        clf.fit(X_train, Y_train)
        score = metrics.get_scorer("balanced_accuracy")(clf, X_test, Y_test)
        r_dict["RB EMB"].append(score)

    return r_dict


def emblink_fairness(XB, YB, pos_edge_index_tr, pos_edge_index_te):

    X_train = np.hstack((XB[pos_edge_index_tr[0]], XB[pos_edge_index_tr[1]]))
    X_test = np.hstack((XB[pos_edge_index_te[0]], XB[pos_edge_index_te[1]]))
    YB = YB.reshape(-1, 1)
    Y_train = np.hstack((YB[pos_edge_index_tr[0]], YB[pos_edge_index_tr[1]]))
    Y_test = np.hstack((YB[pos_edge_index_te[0]], YB[pos_edge_index_te[1]]))

    def double_accuracy(y, y_pred, **kwargs):
        return (
            balanced_accuracy_score(y[:, 0], y_pred[:, 0])
            + balanced_accuracy_score(y[:, 1], y_pred[:, 1])
        ) / 2

    scorer = make_scorer(double_accuracy)

    log = MultiOutputClassifier(
        LogisticRegression(multi_class="multinomial", solver="saga", max_iter=1000)
    )
    mlp = MultiOutputClassifier(
        MLPClassifier(hidden_layer_sizes=(64, 32), solver="adam", max_iter=1000)
    )
    rf = MultiOutputClassifier(RandomForestClassifier(max_depth=4))

    c_dict = {
        "LogisticRegression": log,
        "MLPClassifier": mlp,
        "RandomForestClassifier": rf,
    }
    r_dict = {"RB LINK": []}
    for name, alg in c_dict.items():
        print(f"Evaluating LINK RB with: {name}")
        alg.fit(X_train, Y_train)
        score = scorer(alg, X_test, Y_test)
        r_dict["RB LINK"].append(score)

    return r_dict


def fair_metrics(gt, y, group):
    metrics_dict = {
        "DPd": demographic_parity_difference(gt, y, sensitive_features=group),
        "EOd": equalized_odds_difference(gt, y, sensitive_features=group),
    }
    return metrics_dict


def prediction_fairness(test_edge_idx, test_edge_labels, te_y, group):
    te_dyadic_src = group[test_edge_idx[0]]
    te_dyadic_dst = group[test_edge_idx[1]]

    # SUBGROUP DYADIC
    u = list(combinations_with_replacement(np.unique(group), r=2))

    te_sub_diatic = []
    for i, j in zip(te_dyadic_src, te_dyadic_dst):
        for k, v in enumerate(u):
            if (i, j) == v or (j, i) == v:
                te_sub_diatic.append(k)
                break
    te_sub_diatic = np.asarray(te_sub_diatic)
    # MIXED DYADIC 
    
    te_mixed_dyadic = te_dyadic_src != te_dyadic_dst
    # GROUP DYADIC
    te_gd_dict = fair_metrics(
        np.concatenate([test_edge_labels, test_edge_labels], axis=0),
        np.concatenate([te_y, te_y], axis=0),
        np.concatenate([te_dyadic_src, te_dyadic_dst], axis=0),
    )

    te_md_dict = fair_metrics(test_edge_labels, te_y, te_mixed_dyadic)

    te_sd_dict = fair_metrics(test_edge_labels, te_y, te_sub_diatic)

    fair_list = [
        te_md_dict["DPd"],
        te_md_dict["EOd"],
        te_gd_dict["DPd"],
        te_gd_dict["EOd"],
        te_sd_dict["DPd"],
        te_sd_dict["EOd"],
    ]

    return fair_list


def link_fairness(
    Z, pos_edge_index_tr, pos_edge_index_te, neg_edge_index_tr, neg_edge_index_te, group
):

    train_edge_idx = np.concatenate([pos_edge_index_tr, neg_edge_index_tr], axis=-1)
    train_edge_embs = get_edge_embeddings(Z, train_edge_idx)
    train_edge_labels = get_link_labels(pos_edge_index_tr, neg_edge_index_tr)

    test_edge_idx = np.concatenate([pos_edge_index_te, neg_edge_index_te], axis=-1)
    test_edge_embs = get_edge_embeddings(Z, test_edge_idx)
    test_edge_labels = get_link_labels(pos_edge_index_te, neg_edge_index_te)

    log = model_selection.GridSearchCV(
        pipeline.Pipeline(
            [
                (
                    "logi",
                    LogisticRegression(
                        multi_class="multinomial", solver="saga", max_iter=9000
                    ),
                )
            ]
        ),
        param_grid={"logi__C": [1, 10, 100]},
        cv=4,
        scoring="balanced_accuracy",
    )

    mlp = model_selection.GridSearchCV(
        pipeline.Pipeline(
            [
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32), solver="adam", max_iter=1000
                    ),
                )
            ]
        ),
        param_grid={
            "mlp__alpha": [0.0001, 0.00001],
            "mlp__learning_rate_init": [0.01, 0.001],
        },
        cv=4,
        scoring="balanced_accuracy",
    )

    rf = model_selection.GridSearchCV(
        pipeline.Pipeline([("rf", RandomForestClassifier())]),
        param_grid={"rf__max_depth": [2, 4]},
        cv=4,
        scoring="balanced_accuracy",
    )

    # GROUP DYADIC (one class is involved more in the generation of links)
    te_dyadic_src = group[test_edge_idx[0]]
    te_dyadic_dst = group[test_edge_idx[1]]

    # SUBGROUP DYADIC
    u = list(combinations_with_replacement(np.unique(group), r=2))
    # print(u)
    te_sub_diatic = []
    for i, j in zip(te_dyadic_src, te_dyadic_dst):
        for k, v in enumerate(u):
            if (i, j) == v or (j, i) == v:
                te_sub_diatic.append(k)
                break
    te_sub_diatic = np.asarray(te_sub_diatic)

    # MIXED DYADIC ( imbalanced intra-extra link creation )
    te_mixed_dyadic = te_dyadic_src != te_dyadic_dst

    c_dict = {
        "LogisticRegression": log,
        "MLPClassifier": mlp,
        "RandomForestClassifier": rf,
    }
    fair_dict = {
        "LogisticRegression": [],
        "MLPClassifier": [],
        "RandomForestClassifier": [],
    }

    for name, alg in c_dict.items():

        alg.fit(train_edge_embs, train_edge_labels)
        clf = alg.best_estimator_
        clf.fit(train_edge_embs, train_edge_labels)

        te_y = clf.predict(test_edge_embs)
        te_p = clf.predict_proba(test_edge_embs)[:, 1]

        auc = roc_auc_score(test_edge_labels, te_p)

        te_gd_dict = fair_metrics(
            np.concatenate([test_edge_labels, test_edge_labels], axis=0),
            np.concatenate([te_y, te_y], axis=0),
            np.concatenate([te_dyadic_src, te_dyadic_dst], axis=0),
        )

        te_md_dict = fair_metrics(test_edge_labels, te_y, te_mixed_dyadic)

        te_sd_dict = fair_metrics(test_edge_labels, te_y, te_sub_diatic)

        fair_dict[name] = [
            auc,
            # linkf,
            te_md_dict["DPd"],
            te_md_dict["EOd"],
            te_gd_dict["DPd"],
            te_gd_dict["EOd"],
            te_sd_dict["DPd"],
            te_sd_dict["EOd"],
        ]

    return fair_dict

