"""Explicabilite : pourquoi le modele a rendu ce verdict pour UNE fenetre ?

Deux methodes complementaires :
  1. Poids d'attention du GAT : quelles aretes le modele « regarde » le plus.
  2. GNNExplainer (torch_geometric.explain) : quelles aretes et quelles features font basculer le verdict.

Usage :
    python -W ignore expliquer.py --data gnn-ready --split test                       # 1re attaque bien detectee
    python -W ignore expliquer.py --data gnn-ready --split test --fenetre real-syn-flood_0010   # une autre fenetre, par son identifiant
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer

from modele import ModeleGNN
from train_reel import adapter, TYPES, VERDICTS

FEATURES = ["paquets_s", "octets_s", "requetes_s", "nb_flux", "connexions_ouvertes", "ratio_syn",
            "duree_moy_connexion", "nb_sources_uniques", "nb_destinations", "nb_ports", "entropie_url",
            "entropie_ua", "variabilite_intervalles", "taille_moy_paquets", "cpu", "memoire",
            "latence_p95", "taux_erreurs", "nb_alertes", "gravite_max_alerte", "instant_premiere_anomalie"]
NOMS_X = FEATURES + ["present_externe", "present_interne"] + \
    ["type_" + t for t in ["externe", "gateway", "service", "database", "cache", "pod", "k8s_node"]]


class SortieVerdict(nn.Module):
    """Adaptateur : GNNExplainer attend un modele (x, edge_index, ...) -> logits."""
    def __init__(self, modele):
        super().__init__(); self.m = modele

    def forward(self, x, edge_index, edge_attr):
        return self.m(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))["verdict"]


@torch.no_grad()
def attention_par_arete(modele, g):
    """Attention moyenne (sur les tetes et les couches) de chaque arete reelle (sans les boucles)."""
    modele.eval()
    sortie = modele(g, return_attention=True)
    total = torch.zeros(g.edge_index.size(1)); n = 0
    for ei, alpha in sortie["attention"]:                  # une paire par couche
        a = alpha.mean(1)
        index = {(int(s), int(d)): k for k, (s, d) in enumerate(g.edge_index.t().tolist())}
        for k, (s, d) in enumerate(ei.t().tolist()):
            if s != d and (s, d) in index:
                total[index[(s, d)]] += a[k]
        n += 1
    return total / n


def nom_arete(g, k):
    s, d = g.edge_index[:, k].tolist()
    return f"{g.node_ids[s]} -> {g.node_ids[d]}"


def expliquer(modele, g, top=8, epoques=300):
    modele.eval()
    with torch.no_grad():
        s = modele(g)
        p = F.softmax(s["verdict"], 1)[0]
    classe = p.argmax().item()
    print(f"Fenetre {g.graph_id} : verdict predit = {VERDICTS[classe]} (p={p[classe]:.2f}), "
          f"vrai = {VERDICTS[g.y.item()]}, type predit = {TYPES[s['type'].argmax().item()]}, "
          f"vrai type = {TYPES[g.y_type.item()]}")

    print(f"\n[1] Aretes les plus regardees par le GAT (attention moyenne) :")
    att = attention_par_arete(modele, g)
    for k in att.argsort(descending=True)[:top].tolist():
        print(f"   {att[k]:.3f}  {nom_arete(g, k)}")

    print(f"\n[2] GNNExplainer (verdict « {VERDICTS[classe]} ») :")
    torch.manual_seed(0)
    explainer = Explainer(
        model=SortieVerdict(modele), algorithm=GNNExplainer(epochs=epoques),
        explanation_type="model", node_mask_type="attributes", edge_mask_type="object",
        model_config=dict(mode="multiclass_classification", task_level="graph", return_type="raw"))
    ex = explainer(g.x, g.edge_index, edge_attr=g.edge_attr)
    print("   Aretes les plus importantes :")
    for k in ex.edge_mask.argsort(descending=True)[:top].tolist():
        print(f"   {ex.edge_mask[k]:.3f}  {nom_arete(g, k)}")
    imp = ex.node_mask.sum(0)
    print("   Features les plus importantes (somme sur les noeuds) :")
    for j in imp.argsort(descending=True)[:top].tolist():
        print(f"   {imp[j]:.2f}  {NOMS_X[j]}")
    noeuds = ex.node_mask.sum(1)
    print("   Noeuds les plus importants :")
    for i in noeuds.argsort(descending=True)[:5].tolist():
        print(f"   {noeuds[i]:.2f}  {g.node_ids[i]}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="gnn-ready")
    ap.add_argument("--split", default="test"); ap.add_argument("--fenetre", default=None)
    ap.add_argument("--modele", default="meilleur_modele_reel.pt"); ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args()
    graphes = adapter(torch.load(f"{a.data}/{a.split}.pt", weights_only=False))
    modele = ModeleGNN(graphes[0].x.size(1), graphes[0].edge_attr.size(1), 3, len(TYPES), 3)
    modele.load_state_dict(torch.load(a.modele)); modele.eval()
    if a.fenetre:
        g = next((x for x in graphes if x.graph_id == a.fenetre), None)
        if g is None:
            raise SystemExit(f"Fenetre {a.fenetre} introuvable dans {a.split}.pt")
    else:   # par defaut : une vraie attaque que le modele detecte avec confiance (p >= 0.9)
        with torch.no_grad():
            g = next(x for x in graphes if x.y.item() == 2 and F.softmax(modele(x)["verdict"], 1)[0, 2] >= 0.9)
    expliquer(modele, g, a.top)


if __name__ == "__main__":
    main()
