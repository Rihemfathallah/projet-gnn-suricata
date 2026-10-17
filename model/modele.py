"""Modele GNN multi-tetes : encodeur GATv2 partage + 3 tetes (graphe, externes, internes)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool


class Encodeur(nn.Module):
    """2 a 3 couches GATv2 avec connexions residuelles + LayerNorm (limite le sur-lissage)."""

    def __init__(self, dim_in, dim_arete, cache=64, couches=2, heads=4, dropout=0.2):
        super().__init__()
        self.entree = nn.Linear(dim_in, cache)
        self.convs = nn.ModuleList(
            [GATv2Conv(cache, cache // heads, heads=heads, edge_dim=dim_arete) for _ in range(couches)]
        )
        self.normes = nn.ModuleList([nn.LayerNorm(cache) for _ in range(couches)])
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr, return_attention=False):
        h = F.relu(self.entree(x))
        attentions = []
        for conv, norme in zip(self.convs, self.normes):
            if return_attention:
                out, att = conv(h, edge_index, edge_attr, return_attention_weights=True)
                attentions.append(att)
            else:
                out = conv(h, edge_index, edge_attr)
            h = norme(h + F.dropout(F.relu(out), self.dropout, self.training))  # residuel
        return (h, attentions) if return_attention else h


class ModeleGNN(nn.Module):
    def __init__(self, dim_in, dim_arete, n_verdicts=3, n_types=5, n_roles_internes=3,
                 cache=64, couches=2, heads=4, dropout=0.2):
        super().__init__()
        self.encodeur = Encodeur(dim_in, dim_arete, cache, couches, heads, dropout)
        self.tete_verdict = nn.Linear(2 * cache, n_verdicts)       # pooling moyenne + max
        self.tete_type = nn.Linear(2 * cache, n_types)
        self.tete_externe = nn.Linear(cache, 2)                    # attaquante ou non
        self.tete_interne = nn.Linear(cache, n_roles_internes)     # sain / cible / impacte

    def forward(self, data, return_attention=False):
        res = self.encodeur(data.x, data.edge_index, data.edge_attr, return_attention)
        h, att = res if return_attention else (res, None)
        batch = data.batch if getattr(data, "batch", None) is not None else \
            torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        g = torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], dim=1)
        sortie = {
            "verdict": self.tete_verdict(g),
            "type": self.tete_type(g),
            "externe": self.tete_externe(h),
            "interne": self.tete_interne(h),
        }
        if return_attention:
            sortie["attention"] = att
        return sortie


def focal_loss(logits, cibles, gamma=2.0, poids=None):
    """Focal loss : donne plus de poids aux exemples difficiles (classes desequilibrees)."""
    ce = F.cross_entropy(logits, cibles, weight=poids, reduction="none")
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()


def perte_totale(sortie, data, poids=(1.0, 1.0, 1.0)):
    """Perte multi-taches = somme ponderee des pertes des 3 tetes (masques sur les noeuds)."""
    a, b, c = poids
    perte_graphe = F.cross_entropy(sortie["verdict"], data.y_verdict) + \
        F.cross_entropy(sortie["type"], data.y_type)
    perte_ext = focal_loss(sortie["externe"][data.mask_ext], data.y_ext[data.mask_ext])
    perte_int = focal_loss(sortie["interne"][data.mask_int], data.y_int[data.mask_int])
    return a * perte_graphe + b * perte_ext + c * perte_int
