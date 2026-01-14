
from gnnutils import *
from models import *

# Settings
graphs_type = "total" # "total", "export"
layered = True
multi_graph = True
graph_identifier = f"{'multi-graph-' if multi_graph else ''}{graphs_type}{'-layered' if (layered and not multi_graph) else ''}"
ablate = "Geo-Positional"  # None, "COI", "ECI", "Geo-Positional", "HHI", "TI", "Export Value", "Avg.PCI", "# Prod", "SRCA", "Trade Agreements", "Trustworthiness"

# GPU if possible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)

train_graphs, test_graphs = get_preloaded_graphs(path=f"../../data/5. Graphs Data/{'multi-graph/' if multi_graph else ''}{graphs_type}")

if (layered or multi_graph):
    print("Adding layer embeddings...")
    # Read layer embeddings
    layer_embeddings = pickle.load(open("product_space_embeddings.pickle", "rb"))
    all_graphs = append_layer_embedding(graphs=train_graphs+test_graphs, layer_embeddings=layer_embeddings, multi_graph=multi_graph)
    train_graphs, test_graphs = train_graphs[:len(train_graphs)], all_graphs[len(train_graphs):]
    print(f"New layer shape: {train_graphs[0].x.shape}")

if ablate:

    print(f"Shapes BEFORE ablation of {ablate}")
    print(f"Number of training graphs: {len(train_graphs)}")
    print(f"Number of testing graphs: {len(test_graphs)}")
    print(f"Shape of training graphs {train_graphs[0].x.shape}, {train_graphs[0].edge_attr.shape}")
    print(f"Shape of testing graphs {test_graphs[0].x.shape}, {test_graphs[0].edge_attr.shape}")

    train_graphs, test_graphs = ablate_attribute(train_graphs, test_graphs, attribute=ablate, multi_graph=multi_graph)

    print(f"Shapes AFTER ablation of {ablate}")
    print(f"Number of training graphs: {len(train_graphs)}")
    print(f"Number of testing graphs: {len(test_graphs)}")
    print(f"Shape of training graphs {train_graphs[0].x.shape}, {train_graphs[0].edge_attr.shape}")
    print(f"Shape of testing graphs {test_graphs[0].x.shape}, {test_graphs[0].edge_attr.shape}")


def objective(trial):
    """Objective function for Optuna to optimize."""
    # Hyperparameters to tune
    n_layers = trial.suggest_int("n_layers", 1, 3, step=1)
    hidden_channels = trial.suggest_categorical("hidden_channels", [8, 16, 32, 64, 128])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=False)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=False)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "SGD", "RMSprop", "Adagrad"])

    # GPU if possible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_classes = 1
    num_features = test_graphs[0].num_features

    # Initialize model with sampled hyperparameters
    model = GCN(num_features=num_features, hidden_channels=hidden_channels, num_classes=num_classes, dropout=dropout,\
                 n_layers=n_layers)
    model = model.to(device)  # move model to GPU

    # Select optimizer based on the sampled parameters
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "SGD":
        momentum = trial.suggest_float("momentum", 0.5, 0.99)  # Only for SGD
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == "RMSprop":
        momentum = trial.suggest_float("momentum", 0.5, 0.99)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == "Adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    pos_weight = get_pos_weight(train_graphs=train_graphs)
    criterion = SoftF1Loss(pos_weight=pos_weight).to(device)
    
    # Create batches ensuring positive labels
    batches = create_batches(train_graphs=train_graphs)

    kf = KFold(n_splits=5, shuffle=True, random_state=28)
    val_losses = []

    for i, (train_idx, val_idx) in enumerate(kf.split(batches)): # Split graphs with positive labels

        train_subset = [batches[j] for j in train_idx] # Train subset
        val_subset = [batches[j] for j in val_idx] # Validation subset

        # --- Training Loop ---
        model.train()
        for epoch in range(50):  # 50 epochs per fold
            print(f"K: {i} - Epoch: {epoch}")

            for batch_graphs in train_subset:
                batch = Batch.from_data_list(batch_graphs).to(device) 
                optimizer.zero_grad()
                out = model(batch)
                loss = criterion(out, batch.y.float())
                loss.backward()
                optimizer.step()

        # --- Validation Loop ---
        model.eval()
        val_loss = 0
        num_val_graphs = sum(len(batch_graphs) for batch_graphs in val_subset)  # Total graphs in validation

        with torch.no_grad():
            for batch_graphs in val_subset:
                batch = Batch.from_data_list(batch_graphs).to(device)   # Convert to batch
                out = model(batch)
                val_loss += criterion(out, batch.y.float()).item() * len(batch_graphs)  # Scale loss
        
        val_losses.append(val_loss / num_val_graphs)  # Average over all graphs

    return sum(val_losses) / len(val_losses)


# Define the storage file (saved locally)
db_file = f"optuna_study-GCN-{ablate}.db"
storage = optuna.storages.RDBStorage(f"sqlite:///{db_file}", engine_kwargs={"connect_args": {"timeout": 30}})

# Run Optuna hyperparameter search
study = optuna.create_study(study_name=f"GCN-{'mg-' if multi_graph else ''}{graphs_type}{'-l' if layered else ''}{f'-ablate_{ablate}' if ablate else ''}",\
                             direction="minimize", storage=storage, load_if_exists=True)
MAX_TRIALS = 100
remaining_trials = MAX_TRIALS - len(study.trials)

# if multi_graph:
#     n_jobs = 1
# else:
#     n_jobs = 4
n_jobs = 1 # To be safe, just run one job at a time
study.optimize(objective, n_trials=remaining_trials, gc_after_trial=True, n_jobs=n_jobs, timeout=172800)

# Get the best hyperparameters
best_params = study.best_params
print("Best Hyperparameters:", best_params)