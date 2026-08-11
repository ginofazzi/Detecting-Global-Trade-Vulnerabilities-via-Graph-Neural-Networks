from typing import Callable, Dict, Optional, Sequence, Union

import pandas as pd
import torch


BASELINE_NAMES = ("inverse_partner_outdegree", "import_concentration_hhi")


def _num_nodes_from_edges(edge_index: torch.Tensor, num_nodes: Optional[int]) -> int:
    if num_nodes is not None:
        return int(num_nodes)
    if edge_index.numel() == 0:
        return 0
    return int(edge_index.max().item()) + 1


def _edge_weight_from_graph(graph, edge_attr_weight_col: int = 0) -> torch.Tensor:
    if hasattr(graph, "edge_weight") and graph.edge_weight is not None:
        return graph.edge_weight
    if hasattr(graph, "edge_attr") and graph.edge_attr is not None:
        return graph.edge_attr[:, edge_attr_weight_col]
    raise ValueError("Graph must have either edge_weight or edge_attr.")


def _graph_num_nodes(graph) -> int:
    if getattr(graph, "num_nodes", None) is not None:
        return int(graph.num_nodes)
    if hasattr(graph, "x") and graph.x is not None:
        return int(graph.x.size(0))
    return _num_nodes_from_edges(graph.edge_index, None)


def _prepare_edge_weight(
    edge_weight: torch.Tensor,
    *,
    weight_transform: Optional[str] = None,
) -> torch.Tensor:
    edge_weight = edge_weight.flatten()

    if weight_transform is None:
        weights = edge_weight
    elif weight_transform == "exp":
        weights = torch.exp(edge_weight)
    elif weight_transform == "expm1":
        weights = torch.expm1(edge_weight)
    else:
        raise ValueError("weight_transform must be one of None, 'exp', or 'expm1'.")

    if torch.any(weights < 0):
        raise ValueError(
            "Baseline risk metrics require non-negative edge weights. "
            "Use a suitable weight_transform or inspect graph.edge_weight."
        )
    return weights


def inverse_partner_outdegree_risk(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    num_nodes: Optional[int] = None,
    eps: float = 1e-12,
    weight_transform: Optional[str] = None,
) -> torch.Tensor:
    """
    Compute each importer's exposure to suppliers with low weighted out-degree.

    Edge convention:
      edge_index[0] = exporter/source/supplier
      edge_index[1] = importer/target/buyer
      edge_weight = export value from source to target

    For importer j, this returns:
      sum_i share(i -> j) * 1 / out_strength(i)

    where share(i -> j) is i's share of j's imports in this graph.
    Nodes with no imports receive 0.
    """
    source, target = edge_index
    num_nodes = _num_nodes_from_edges(edge_index, num_nodes)
    weights = _prepare_edge_weight(edge_weight, weight_transform=weight_transform)

    out_strength = torch.zeros(num_nodes, dtype=weights.dtype, device=weights.device)
    out_strength.scatter_add_(0, source, weights)

    supplier_risk = 1.0 / (out_strength + eps)
    risk_contribution = weights * supplier_risk[source]

    weighted_risk_sum = torch.zeros_like(out_strength)
    weighted_risk_sum.scatter_add_(0, target, risk_contribution)

    total_imports = torch.zeros_like(out_strength)
    total_imports.scatter_add_(0, target, weights)

    node_risk = torch.zeros_like(out_strength)
    has_imports = total_imports > 0
    node_risk[has_imports] = weighted_risk_sum[has_imports] / total_imports[has_imports]
    return node_risk


def import_concentration_hhi(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    num_nodes: Optional[int] = None,
    weight_transform: Optional[str] = None,
) -> torch.Tensor:
    """
    Compute import concentration as HHI over incoming supplier shares.

    For importer j, this returns:
      sum_i (imports_from_i_to_j / total_imports_j) ** 2

    A node sourcing all imports from one supplier has HHI 1. A more diversified
    node has lower HHI. Nodes with no imports receive 0.
    """
    _, target = edge_index
    num_nodes = _num_nodes_from_edges(edge_index, num_nodes)
    weights = _prepare_edge_weight(edge_weight, weight_transform=weight_transform)

    total_imports = torch.zeros(num_nodes, dtype=weights.dtype, device=weights.device)
    total_imports.scatter_add_(0, target, weights)

    shares = torch.zeros_like(weights)
    has_target_imports = total_imports[target] > 0
    shares[has_target_imports] = (
        weights[has_target_imports] / total_imports[target[has_target_imports]]
    )

    hhi = torch.zeros_like(total_imports)
    hhi.scatter_add_(0, target, shares.square())
    return hhi


def graph_baseline_risks(
    graph,
    edge_attr_weight_col: int = 0,
    weight_transform: Optional[str] = None,
) -> torch.Tensor:
    """
    Return a [num_nodes, 2] tensor with both simple graph-native baselines.
    """
    edge_weight = _edge_weight_from_graph(graph, edge_attr_weight_col=edge_attr_weight_col)
    num_nodes = _graph_num_nodes(graph)

    inverse_outdegree = inverse_partner_outdegree_risk(
        graph.edge_index,
        edge_weight,
        num_nodes=num_nodes,
        weight_transform=weight_transform,
    )
    hhi = import_concentration_hhi(
        graph.edge_index,
        edge_weight,
        num_nodes=num_nodes,
        weight_transform=weight_transform,
    )
    return torch.stack([inverse_outdegree, hhi], dim=1)


BASELINE_FUNCTIONS: Dict[str, Callable[..., torch.Tensor]] = {
    "inverse_partner_outdegree": inverse_partner_outdegree_risk,
    "import_concentration_hhi": import_concentration_hhi,
}


def graph_baseline_risk(
    graph,
    baseline: str,
    edge_attr_weight_col: int = 0,
    weight_transform: Optional[str] = None,
) -> torch.Tensor:
    if baseline not in BASELINE_FUNCTIONS:
        raise ValueError(f"Unknown baseline {baseline}. Choose one of {list(BASELINE_FUNCTIONS)}.")

    edge_weight = _edge_weight_from_graph(graph, edge_attr_weight_col=edge_attr_weight_col)
    return BASELINE_FUNCTIONS[baseline](
        graph.edge_index,
        edge_weight,
        num_nodes=_graph_num_nodes(graph),
        weight_transform=weight_transform,
    )


def baseline_result_frame(
    graph,
    country_ids: Sequence[int],
    *,
    digits: int,
    year: int,
    graph_type: str,
    commodity: Union[str, int],
    baseline: str,
    edge_attr_weight_col: int = 0,
    weight_transform: Optional[str] = None,
) -> pd.DataFrame:
    risk = graph_baseline_risk(
        graph,
        baseline=baseline,
        edge_attr_weight_col=edge_attr_weight_col,
        weight_transform=weight_transform,
    ).detach().cpu()

    y = graph.y.detach().cpu()
    if len(country_ids) != risk.numel():
        raise ValueError(
            f"country_ids has length {len(country_ids)}, but graph has {risk.numel()} nodes."
        )

    return pd.DataFrame(
        {
            "digits": digits,
            "year": year,
            "graph_type": graph_type,
            "commodity": str(commodity),
            "country_id": list(country_ids),
            "y": y.numpy(),
            "risk": risk.numpy(),
        }
    )


def baseline_graphs_to_frame(
    graphs: Sequence,
    country_ids_by_graph: Sequence[Sequence[int]],
    metadata: Sequence[dict],
    *,
    baseline: str,
    edge_attr_weight_col: int = 0,
    weight_transform: Optional[str] = None,
) -> pd.DataFrame:
    frames = []
    for graph, country_ids, meta in zip(graphs, country_ids_by_graph, metadata):
        frames.append(
            baseline_result_frame(
                graph,
                country_ids,
                baseline=baseline,
                edge_attr_weight_col=edge_attr_weight_col,
                weight_transform=weight_transform,
                **meta,
            )
        )
    return pd.concat(frames, ignore_index=True)
