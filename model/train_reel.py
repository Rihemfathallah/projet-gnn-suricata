"""Entrainement sur le dataset reel livre par P2 (gnn-ready/train.pt, val.pt, test.pt).

Le decoupage train / val / test est deja fait PAR RUN (aucune fenetre d'un meme run n'est
dans deux ensembles) : ne le refaites pas.

Usage :  python -W ignore train_reel.py --data gnn-ready
"""
import argparse, collections, random
import torch
from sklearn.metrics import f1_score
from torch_geometric.loader import DataLoader

from modele import ModeleGNN, perte_totale

VERDICTS = ["normal", "pic_legitime", "attaque"]
TYPES = ["none", "ddos_udp_flood", "ddos_icmp_flood", "ddos_syn_flood", "ddos_http_flood",
         "slowloris", "slow_post", "bruteforce_distribue", "scan_distribue"]
NOEUDS = ["cible", "impacte", "sain"]   # ordre de y_node chez P2 : 0 cible, 1 impacte, 2 sain


def adapter(graphes):
    """Renomme les attributs de P2 vers ceux que lit modele.py."""
    for g in graphes:
        g.y_verdict = g.y                       # verdict du graphe
        g.y_ext = g.y_source.long()             # source attaquante (0/1) -- noeuds externes
        g.y_int = g.y_node                      # 0 cible / 1 impacte / 2 sain -- noeuds internes
        g.mask_ext = g.is_ext
        g.mask_int = ~g.is_ext
    return graphes


@torch.no_grad()
def evaluer(modele, graphes):
    modele.eval()
    P = collections.defaultdict(list); Y = collections.defaultdict(list)
    for lot in DataLoader(graphes, batch_size=32):
        s = modele(lot)
        P["verdict"] += s["verdict"].argmax(1).tolist(); Y["verdict"] += lot.y_verdict.tolist()
        P["type"] += s["type"].argmax(1).tolist(); Y["type"] += lot.y_type.tolist()
        m = lot.mask_ext
        P["source"] += s["externe"][m].argmax(1).tolist(); Y["source"] += lot.y_ext[m].tolist()
        m = lot.mask_int
        P["interne"] += s["interne"][m].argmax(1).tolist(); Y["interne"] += lot.y_int[m].tolist()
    res = {}
    for k in P:   # F1 macro sur les classes qui existent vraiment dans les vrais labels
        res[k] = f1_score(Y[k], P[k], labels=sorted(set(Y[k])), average="macro", zero_division=0)
    return res, P, Y


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="gnn-ready")
    ap.add_argument("--epoques", type=int, default=60); ap.add_argument("--graine", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.graine); torch.manual_seed(a.graine)

    tr = adapter(torch.load(f"{a.data}/train.pt", weights_only=False))
    va = adapter(torch.load(f"{a.data}/val.pt", weights_only=False))
    te = adapter(torch.load(f"{a.data}/test.pt", weights_only=False))
    dim_x, dim_e = tr[0].x.size(1), tr[0].edge_attr.size(1)
    print(f"graphes : train={len(tr)} val={len(va)} test={len(te)} | dim_x={dim_x} dim_arete={dim_e}")

    modele = ModeleGNN(dim_x, dim_e, n_verdicts=3, n_types=len(TYPES), n_roles_internes=3)
    opt = torch.optim.AdamW(modele.parameters(), lr=2e-3, weight_decay=1e-4)
    meilleur, chemin = -1.0, "meilleur_modele_reel.pt"

    for ep in range(1, a.epoques + 1):
        modele.train(); total = 0.0
        for lot in DataLoader(tr, batch_size=16, shuffle=True):
            opt.zero_grad(); perte = perte_totale(modele(lot), lot); perte.backward(); opt.step()
            total += perte.item()
        f1, _, _ = evaluer(modele, va)
        score = sum(f1.values()) / len(f1)
        if score >= meilleur:
            meilleur = score; torch.save(modele.state_dict(), chemin)
        if ep == 1 or ep % 10 == 0:
            print(f"epoque {ep:3d} perte={total:7.3f} | F1 val : " + " ".join(f"{k}={v:.2f}" for k, v in f1.items()))

    modele.load_state_dict(torch.load(chemin))
    f1, P, Y = evaluer(modele, te)
    print("\n=== TEST (meilleur modele sur la validation) ===")
    for k, v in f1.items():
        print(f"  F1 macro {k:8s} {v:.3f}")
    # Detail utile : matrice de confusion du verdict
    from sklearn.metrics import confusion_matrix
    print("\nConfusion verdict (lignes = vrai, colonnes = predit ; normal / pic / attaque) :")
    print(confusion_matrix(Y["verdict"], P["verdict"], labels=[0, 1, 2]))


if __name__ == "__main__":
    main()
