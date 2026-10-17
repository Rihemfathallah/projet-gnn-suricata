"""Regroupement des fenetres predites en INCIDENTS (un incident = une attaque du debut a la fin).

Principe : on lit les fenetres d'un meme run dans l'ordre. Les fenetres consecutives classees
« attaque » qui visent la meme cible sont fusionnees. Un trou de `tolerance` fenetres (par defaut 2)
est accepte ; au-dela, l'incident est ferme.

Usage :
    python -W ignore incidents.py --data gnn-ready --split test --modele meilleur_modele_reel.pt
(lancer d'abord train_reel.py pour obtenir meilleur_modele_reel.pt)
"""
import argparse, collections
import torch
import torch.nn.functional as F

from modele import ModeleGNN
from train_reel import adapter, TYPES, VERDICTS


def run_de(g):             # 'real-http-flood_0006' -> 'real-http-flood'
    return g.graph_id.rsplit("_", 1)[0]


def rang_de(g):            # 'real-http-flood_0006' -> 6 (ordre de la fenetre dans le run)
    return int(g.graph_id.rsplit("_", 1)[1])


@torch.no_grad()
def predire_fenetre(modele, g):
    """Prediction du modele pour UNE fenetre, sous une forme lisible."""
    modele.eval()
    s = modele(g)
    p_verdict = F.softmax(s["verdict"], 1)[0]
    verdict = VERDICTS[p_verdict.argmax().item()]
    type_att = TYPES[s["type"].argmax(1).item()]
    ext = g.is_ext.nonzero().flatten().tolist()
    internes = (~g.is_ext).nonzero().flatten().tolist()
    p_ext = F.softmax(s["externe"], 1)[:, 1]
    p_int = F.softmax(s["interne"], 1)                      # colonnes : cible / impacte / sain
    sources = [g.node_ids[i] for i in ext if p_ext[i] > 0.5]
    impactes = [g.node_ids[i] for i in internes if p_int[i].argmax().item() == 1]
    cibles = [i for i in internes if p_int[i].argmax().item() == 0]
    cible = g.node_ids[max(cibles, key=lambda i: p_int[i, 0].item())] if cibles else None
    return dict(rang=rang_de(g), verdict=verdict, p_attaque=p_verdict[2].item(), type=type_att,
                cible=cible, sources=sources, impactes=impactes)


def _majorite(valeurs):
    valeurs = [v for v in valeurs if v is not None]
    return collections.Counter(valeurs).most_common(1)[0][0] if valeurs else None


def _frequents(listes, n):
    """Elements presents dans au moins la moitie des n fenetres."""
    c = collections.Counter(x for l in listes for x in set(l))
    return sorted(k for k, v in c.items() if v >= n / 2)


def regrouper(fenetres, tolerance=1):
    """fenetres : liste de dicts (sortie de predire_fenetre) d'UN run. Renvoie la liste des incidents."""
    fenetres = sorted(fenetres, key=lambda f: f["rang"])
    incidents, cour, trou = [], [], 0

    def fermer():
        if cour:
            incidents.append(cour[:])
            cour.clear()

    for f in fenetres:
        if f["verdict"] == "attaque":
            if cour and f["cible"] and cour[-1]["cible"] and f["cible"] != cour[-1]["cible"]:
                fermer()                                     # autre cible : nouvel incident
            cour.append(f); trou = 0
        elif cour:
            trou += 1
            if trou > tolerance:
                fermer(); trou = 0
    fermer()
    sortie = []
    for k, ws in enumerate(incidents, 1):
        types = [w["type"] for w in ws if w["type"] != "none"]
        sortie.append(dict(
            incident=f"inc-{k:04d}", type=_majorite(types) or "inconnu",
            confiance=round(sum(w["p_attaque"] for w in ws) / len(ws), 3),
            cible=_majorite([w["cible"] for w in ws]),
            debut=ws[0]["rang"], fin=ws[-1]["rang"], nb_fenetres=len(ws),
            sources=_frequents([w["sources"] for w in ws], len(ws)),
            impactes=_frequents([w["impactes"] for w in ws], len(ws))))
    return sortie


def incidents_reels(graphes, tolerance=1):
    """Les vrais incidents d'un run, construits a partir des labels (meme regle de regroupement)."""
    fen = [dict(rang=rang_de(g), verdict="attaque" if g.y.item() == 2 else "autre",
                p_attaque=1.0, type=TYPES[g.y_type.item()],
                cible=next((g.node_ids[i] for i in range(g.num_nodes) if g.y_node[i] == 0 and not g.is_ext[i]), None),
                sources=[g.node_ids[i] for i in range(g.num_nodes) if g.y_source[i] == 1],
                impactes=[]) for g in graphes]
    return regrouper(fen, tolerance)


def recouvrement(a, b):
    """Nombre de fenetres communes a deux incidents."""
    return max(0, min(a["fin"], b["fin"]) - max(a["debut"], b["debut"]) + 1)


def chevauche(a, b):
    return recouvrement(a, b) > 0


def evaluer_incidents(par_run_pred, par_run_reel):
    n_pred = n_reel = tp_pred = tp_reel = type_ok = cible_ok = 0
    retards, fenetres_att = [], 0
    for run in par_run_reel:
        P, R = par_run_pred.get(run, []), par_run_reel[run]
        n_pred += len(P); n_reel += len(R); fenetres_att += sum(p["nb_fenetres"] for p in P)
        for p in P:
            tp_pred += any(chevauche(p, r) for r in R)
        for r in R:
            m = sorted((p for p in P if chevauche(p, r)), key=lambda p: -recouvrement(p, r))
            if m:
                tp_reel += 1; p = m[0]          # on compare avec l'incident predit qui recouvre le plus
                type_ok += p["type"] == r["type"]; cible_ok += p["cible"] == r["cible"]
                retards.append(p["debut"] - r["debut"])
    return dict(
        incidents_predits=n_pred, incidents_reels=n_reel,
        fragmentation=round(n_pred / n_reel, 2) if n_reel else None,   # 1.0 = un incident predit par vrai incident
        precision=tp_pred / n_pred if n_pred else 0.0, rappel=tp_reel / n_reel if n_reel else 0.0,
        type_correct=f"{type_ok}/{tp_reel}", cible_correcte=f"{cible_ok}/{tp_reel}",
        compression=round(fenetres_att / n_pred, 1) if n_pred else None,
        retard_moyen_fenetres=round(sum(retards) / len(retards), 2) if retards else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="gnn-ready"); ap.add_argument("--split", default="test")
    ap.add_argument("--modele", default="meilleur_modele_reel.pt"); ap.add_argument("--tolerance", type=int, default=2)
    a = ap.parse_args()

    graphes = adapter(torch.load(f"{a.data}/{a.split}.pt", weights_only=False))
    modele = ModeleGNN(graphes[0].x.size(1), graphes[0].edge_attr.size(1), 3, len(TYPES), 3)
    modele.load_state_dict(torch.load(a.modele))

    runs = collections.defaultdict(list)
    for g in graphes:
        runs[run_de(g)].append(g)
    pred, reel = {}, {}
    for run, gs in sorted(runs.items()):
        pred[run] = regrouper([predire_fenetre(modele, g) for g in gs], a.tolerance)
        reel[run] = incidents_reels(gs, a.tolerance)

    print(f"=== Incidents predits ({a.split}) ===")
    for run in sorted(runs):
        print(f"\n[{run}] {len(runs[run])} fenetres")
        for inc in pred[run]:
            print(f"  {inc['incident']} {inc['type']:16s} cible={inc['cible']} fenetres {inc['debut']}-{inc['fin']} "
                  f"({inc['nb_fenetres']}) confiance={inc['confiance']} sources={inc['sources']} impactes={inc['impactes']}")
        if not pred[run]:
            print("  aucun incident")
        for r in reel[run]:
            print(f"  (reel : {r['type']} cible={r['cible']} fenetres {r['debut']}-{r['fin']})")
    print("\n=== Evaluation des incidents ===")
    for k, v in evaluer_incidents(pred, reel).items():
        print(f"  {k:22s} {v}")


if __name__ == "__main__":
    main()
