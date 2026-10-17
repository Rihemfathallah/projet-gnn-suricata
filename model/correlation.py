"""Fonction finale de la partie P3 : correlation_ids_suricata (mode « gnn »).

Entree  : les fenetres de trafic (graphes PyG au format de P2) dans l'ordre du temps.
Sortie  : une liste d'incidents DoS / DDoS (type, cible, debut, fin, sources, confiance, explication).

IMPORTANT : cette fonction prend des FENETRES DEJA CONSTRUITES (le format de P2). La conversion
« fichiers eve.json des deux sondes -> fenetres » appartient a la chaine commune de P2 : tant
qu'elle n'est pas branchee, on ne peut pas appeler la fonction directement avec les alertes brutes.
Le mode « regles » est la version de P2 : on lui passe sa fonction via `regles=`.

Usage :  python -W ignore correlation.py --data gnn-ready --split test
"""
import argparse, collections, json
import torch
import torch.nn.functional as F
from torch_geometric.explain import Explainer, GNNExplainer

from modele import ModeleGNN
from train_reel import adapter, TYPES
from incidents import predire_fenetre, regrouper, run_de, rang_de
from expliquer import SortieVerdict, attention_par_arete, nom_arete, NOMS_X

_CACHE = {}


def _charger(chemin, g):
    if chemin not in _CACHE:
        m = ModeleGNN(g.x.size(1), g.edge_attr.size(1), 3, len(TYPES), 3)
        m.load_state_dict(torch.load(chemin)); m.eval()
        _CACHE[chemin] = m
    return _CACHE[chemin]


def _explication(modele, g, inc, p, epoques=150):
    """Phrase d'explication : attention du GAT (aretes) + GNNExplainer (features)."""
    att = attention_par_arete(modele, g)
    aretes = [nom_arete(g, k) for k in att.argsort(descending=True)[:3].tolist()]
    torch.manual_seed(0)
    ex = Explainer(
        model=SortieVerdict(modele), algorithm=GNNExplainer(epochs=epoques),
        explanation_type="model", node_mask_type="attributes", edge_mask_type="object",
        model_config=dict(mode="multiclass_classification", task_level="graph", return_type="raw"),
    )(g.x, g.edge_index, edge_attr=g.edge_attr)
    imp = ex.node_mask.sum(0)
    feats = [NOMS_X[j] for j in imp.argsort(descending=True)[:4].tolist()
             if not NOMS_X[j].startswith(("present_", "type_"))][:3]
    srcs = (f"{len(inc['sources'])} sous-reseau(x) source(s) predit(s) (indicateurs, pas une attribution "
            f"certaine : les adresses peuvent etre usurpees)") if inc["sources"] else "sources non determinees"
    return (f"{inc['nb_fenetres']} fenetres d'attaque entre les fenetres {inc['debut']} et {inc['fin']}, vers {inc['cible']} ; "
            f"{srcs} ; "
            f"fenetre la plus sure : {g.graph_id} (p={p:.2f}) ; "
            f"aretes les plus regardees : {', '.join(aretes)} ; "
            f"features dominantes (GNNExplainer) : {', '.join(feats)}")


def correlation_ids_suricata(fenetres, mode="gnn", modele="meilleur_modele_reel.pt",
                             tolerance=2, explication=True, regles=None):
    """Fusionne les fenetres d'attaque en incidents DoS / DDoS.

    fenetres    : liste de graphes PyG (format P2), un ou plusieurs runs, dans n'importe quel ordre
    mode        : « gnn » (ce fichier) ou « regles » (fonction de P2, passee via `regles`)
    tolerance   : nombre de fenetres « calmes » acceptees au milieu d'un incident
    explication : calcule l'explication (GNNExplainer, quelques secondes par incident)
    """
    if mode == "regles":
        if regles is None:
            raise NotImplementedError("mode « regles » : passez la fonction de P2 avec regles=...")
        return regles(fenetres)
    if mode != "gnn":
        raise ValueError(f"mode inconnu : {mode!r} (attendu : 'gnn' ou 'regles')")
    if not fenetres:
        return []

    m = _charger(modele, fenetres[0])
    runs = collections.defaultdict(list)
    for g in fenetres:
        runs[run_de(g)].append(g)

    sortie, numero = [], 0
    for run, gs in sorted(runs.items()):
        preds = {rang_de(g): (g, predire_fenetre(m, g)) for g in gs}
        for inc in regrouper([p for _, p in preds.values()], tolerance):
            numero += 1
            inc["incident"] = f"inc-{numero:04d}"
            inc["run"] = run
            inc["sondes"] = None        # non disponible : le dataset actuel ne separe pas IDS 1 / IDS 2
            att = [(g, p) for r, (g, p) in preds.items()
                   if inc["debut"] <= r <= inc["fin"] and p["verdict"] == "attaque"]
            g, p = max(att, key=lambda gp: gp[1]["p_attaque"])
            inc["explication"] = _explication(m, g, inc, p["p_attaque"]) if explication else ""
            sortie.append(inc)
    return sortie


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="gnn-ready")
    ap.add_argument("--split", default="test"); ap.add_argument("--modele", default="meilleur_modele_reel.pt")
    a = ap.parse_args()
    graphes = adapter(torch.load(f"{a.data}/{a.split}.pt", weights_only=False))
    res = correlation_ids_suricata(graphes, modele=a.modele)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
