
from gnnutils import *
from models import *

from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold


# Settings
graphs_type = "export"  # "total", "export"
layered = True
multi_graph = False
graph_identifier = f"{'multi-graph-' if multi_graph else ''}{graphs_type}{'-layered' if (layered and not multi_graph) else ''}"
ablate = None#"Geo-Positional"  # None, "COI", "ECI", "Geo-Positional", "HHI", "TI", "Export Value", "Avg.PCI", "# Prod", "SRCA", "Trade Agreements", "Trustworthiness"

# GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

train_graphs, test_graphs = get_preloaded_graphs(
    path=f"../../data/5. Graphs Data/{'multi-graph/' if multi_graph else ''}{graphs_type}"
)

if layered or multi_graph:
    print("Adding layer embeddings...")
    layer_embeddings = pickle.load(open("product_space_embeddings.pickle", "rb"))
    all_graphs = append_layer_embedding(
        graphs=train_graphs + test_graphs,
        layer_embeddings=layer_embeddings,
        multi_graph=multi_graph,
    )
    train_graphs, test_graphs = all_graphs[: len(train_graphs)], all_graphs[len(train_graphs) :]
    print(f"New layer shape: {train_graphs[0].x.shape}")

if ablate:
    print(f"Shapes BEFORE ablation of {ablate}")
    print(f"Number of training graphs: {len(train_graphs)}")
    print(f"Number of testing graphs: {len(test_graphs)}")
    print(f"Shape of training graphs {train_graphs[0].x.shape}, {train_graphs[0].edge_attr.shape}")
    print(f"Shape of testing graphs {test_graphs[0].x.shape}, {test_graphs[0].edge_attr.shape}")

    train_graphs, test_graphs = ablate_attribute(
        train_graphs, test_graphs, attribute=ablate, multi_graph=multi_graph
    )

    print(f"Shapes AFTER ablation of {ablate}")
    print(f"Number of training graphs: {len(train_graphs)}")
    print(f"Number of testing graphs: {len(test_graphs)}")
    print(f"Shape of training graphs {train_graphs[0].x.shape}, {train_graphs[0].edge_attr.shape}")
    print(f"Shape of testing graphs {test_graphs[0].x.shape}, {test_graphs[0].edge_attr.shape}")


def stack_graph_data(graphs):
    x = torch.cat([graph.x for graph in graphs], dim=0).cpu().numpy()
    y = torch.cat([graph.y for graph in graphs], dim=0).cpu().numpy()
    return x, y


X_train, y_train = stack_graph_data(train_graphs)


def objective(trial):
    """Objective function for Optuna to optimize the RandomForest wrapper."""
    params = {
        "n_estimators": trial.suggest_categorical("n_estimators", [100, 200, 300]),
        "max_depth": trial.suggest_categorical("max_depth", [None, 10, 20, 30]),
        "min_samples_split": trial.suggest_categorical("min_samples_split", [2, 5, 10]),
        "min_samples_leaf": trial.suggest_categorical("min_samples_leaf", [1, 2, 4]),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=28)
    fold_loss = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), start=1):
        model = RandomForest(random_seed=28, **params)
        model.fit(X_train[train_idx], y_train[train_idx])

        val_scores = model.predict_proba(X_train[val_idx])[:, 1].cpu().numpy()

        tp = (y_train[val_idx] * val_scores).sum()  # True Positives
        fp = ((1 - y_train[val_idx]) * val_scores).sum()  # False Positives
        fn = (y_train[val_idx] * (1 - val_scores)).sum()  # False Negatives

        precision = tp / (tp + fp + 1e-8)  # Avoid division by zero
        recall = tp / (tp + fn + 1e-8)

        f1_positive = (2 * precision * recall) / (precision + recall + 1e-8)

        loss = (1 - f1_positive)
    
        fold_loss.append(loss)

    mean_loss = float(np.mean(fold_loss))
    print(f"Mean loss: {mean_loss:.4f}")
    return mean_loss


# Define the storage file (saved locally)
db_file = f"optuna_study-RandomForest-{ablate}.db"
storage = optuna.storages.RDBStorage(
    f"sqlite:///{db_file}", engine_kwargs={"connect_args": {"timeout": 30}}
)

# Run Optuna hyperparameter search
study = optuna.create_study(
    study_name=f"RandomForest-{'mg-' if multi_graph else ''}{graphs_type}{'-l' if layered else ''}{f'-ablate_{ablate}' if ablate else ''}",
    direction="minimize",
    storage=storage,
    load_if_exists=True,
)
MAX_TRIALS = 20
remaining_trials = MAX_TRIALS - len(study.trials)

n_jobs = 1  # To be safe, just run one job at a time
study.optimize(
    objective,
    n_trials=remaining_trials,
    gc_after_trial=True,
    n_jobs=n_jobs,
    timeout=172800,
)

# Get the best hyperparameters
best_params = study.best_params
print("Best Hyperparameters:", best_params)
print("Best Objective Value:", study.best_value)
