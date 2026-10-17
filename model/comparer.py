"""Comparaison GAT (GATv2) / GraphSAGE sur le dataset de P2, a hyperparametres identiques.

Chaque modele est entraine sur plusieurs graines ; on affiche la moyenne et l'ecart-type des F1
sur le TEST (le meilleur modele de chaque entrainement est choisi sur la VALIDATION).

Usage :  python -W ignore comparer.py --data gnn-ready --graines 5 | tee resultats_comparaison.txt
"""
import argparse, random, statistics, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv

from modele import ModeleGNN, perte_totale
from train_reel import adapter, evaluer, TYPES


class EncodeurSAGE(nn.Module):
    """Meme structure que l'encodeur GAT (residuel + LayerNorm), mais avec SAGEConv.
    SAGEConv n'utilise PAS les attributs d'aretes (edge_attr) : c'est la difference a noter."""

    def __init__(self, dim_in, cache=64, couches=2, dropout=0.2):
        super().__init__()
        self.entree = nn.Linear(dim_in, cache)
        self.convs = nn.ModuleList([SAGEConv(cache, cache) for _ in range(couches)])
        self.normes = nn.ModuleList([nn.LayerNorm(cache) for _ in range(couches)])
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr, return_attention=False):
        h = F.relu(self.entree(x))
        for conv, norme in zip(self.convs, self.normes):
            h = norme(h + F.dropout(F.relu(conv(h, edge_index)), self.dropout, self.training))
        return (h, []) if return_attention else h


def construire(nom, dim_x, dim_e):
    m = ModeleGNN(dim_x, dim_e, 3, len(TYPES), 3)
    if nom == "GraphSAGE":
        m.encodeur = EncodeurSAGE(dim_x)
    return m


def entrainer(nom, graine, tr, va, te, epoques):
    random.seed(graine); torch.manual_seed(graine)
    m = construire(nom, tr[0].x.size(1), tr[0].edge_attr.size(1))
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    meilleur, etat = -1.0, None
    for _ in range(epoques):
        m.train()
        for lot in DataLoader(tr, batch_size=16, shuffle=True):
            opt.zero_grad(); perte_totale(m(lot), lot).backward(); opt.step()
        f1, _, _ = evaluer(m, va)
        score = sum(f1.values()) / len(f1)
        if score >= meilleur:
            meilleur, etat = score, {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(etat)
    f1, P, Y = evaluer(m, te)
    attaques_ok = sum(1 for p, y in zip(P["verdict"], Y["verdict"]) if y == 2 and p == 2)
    return f1, attaques_ok, sum(p.numel() for p in m.parameters())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="gnn-ready")
    ap.add_argument("--graines", type=int, default=5); ap.add_argument("--epoques", type=int, default=60)
    a = ap.parse_args()
    tr = adapter(torch.load(f"{a.data}/train.pt", weights_only=False))
    va = adapter(torch.load(f"{a.data}/val.pt", weights_only=False))
    te = adapter(torch.load(f"{a.data}/test.pt", weights_only=False))
    n_att = sum(1 for g in te if g.y.item() == 2)
    print(f"test : {len(te)} graphes dont {n_att} attaques | {a.graines} graines | {a.epoques} epoques\n")

    resultats = {}
    for nom in ("GAT", "GraphSAGE"):
        t0 = time.time(); runs = []
        for graine in range(a.graines):
            runs.append(entrainer(nom, graine, tr, va, te, a.epoques))
            print(f"  {nom} graine {graine} : " + " ".join(f"{k}={v:.2f}" for k, v in runs[-1][0].items()), flush=True)
        resultats[nom] = (runs, time.time() - t0)

    print("\n=== F1 macro sur le TEST : moyenne +- ecart-type ===")
    print(f"{'modele':10s} {'verdict':>13s} {'type':>13s} {'source':>13s} {'interne':>13s} {'attaques OK':>12s} {'params':>8s} {'temps':>7s}")
    for nom, (runs, temps) in resultats.items():
        cols = []
        for k in ("verdict", "type", "source", "interne"):
            v = [r[0][k] for r in runs]
            cols.append(f"{statistics.mean(v):.2f}+-{statistics.pstdev(v):.2f}")
        att = statistics.mean(r[1] for r in runs)
        print(f"{nom:10s} " + " ".join(f"{c:>13s}" for c in cols) + f" {att:>9.1f}/{n_att}" + f" {runs[0][2]:>8d} {temps:>6.0f}s")
    print("\nNote : GraphSAGE n'utilise pas les attributs d'aretes (flux / dependance / volume / direction).")


if __name__ == "__main__":
    main()
