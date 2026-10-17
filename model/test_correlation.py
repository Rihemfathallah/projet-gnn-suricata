"""Tests de correlation_ids_suricata. Lancer :  python -W ignore test_correlation.py
(ou avec pytest :  python -m pytest -W ignore test_correlation.py -q)
Pre-requis : gnn-ready/ et meilleur_modele_reel.pt dans le dossier courant (voir train_reel.py)."""
import random
import torch

from train_reel import adapter
from correlation import correlation_ids_suricata
from incidents import run_de

_TEST = adapter(torch.load("gnn-ready/test.pt", weights_only=False))


def un_run(nom):
    return [g for g in _TEST if run_de(g) == nom]


def test_run_normal_aucun_incident():
    assert correlation_ids_suricata(un_run("real-healthy"), explication=False) == []


def test_pic_legitime_pas_de_faux_incident():
    # Attention : reussit parce que le modele classe le pic en « normal », pas parce qu'il le reconnait.
    assert correlation_ids_suricata(un_run("real-peak"), explication=False) == []


def test_http_flood_un_seul_incident():
    res = correlation_ids_suricata(un_run("real-http-flood-r3"), explication=False)
    assert len(res) == 1
    assert res[0]["type"] == "ddos_http_flood"
    assert res[0]["cible"] == "boutique/frontend"


def test_syn_flood_un_seul_incident():
    res = correlation_ids_suricata(un_run("real-syn-flood"), explication=False)
    assert len(res) == 1
    assert res[0]["type"] == "ddos_syn_flood"
    assert res[0]["cible"] == "boutique/frontend"


def test_format_de_sortie():
    res = correlation_ids_suricata(un_run("real-http-flood-r3"), explication=True)
    inc = res[0]
    for cle in ("incident", "run", "type", "confiance", "cible", "debut", "fin", "nb_fenetres",
                "sondes", "sources", "impactes", "explication"):
        assert cle in inc, f"cle manquante : {cle}"
    assert 0.0 <= inc["confiance"] <= 1.0
    assert inc["debut"] <= inc["fin"] and inc["nb_fenetres"] >= 1
    assert isinstance(inc["sources"], list) and isinstance(inc["impactes"], list)
    assert inc["cible"] in inc["explication"] and len(inc["explication"]) > 50


def test_ordre_des_fenetres_sans_importance():
    f = un_run("real-syn-flood")
    melange = f[:]; random.Random(0).shuffle(melange)
    a = correlation_ids_suricata(f, explication=False)
    b = correlation_ids_suricata(melange, explication=False)
    assert [(i["type"], i["debut"], i["fin"]) for i in a] == [(i["type"], i["debut"], i["fin"]) for i in b]


def test_plusieurs_runs_donnent_plusieurs_incidents():
    res = correlation_ids_suricata(_TEST, explication=False)
    assert sorted(i["run"] for i in res) == ["real-http-flood-r3", "real-syn-flood"]
    assert [i["incident"] for i in res] == ["inc-0001", "inc-0002"]


def test_liste_vide():
    assert correlation_ids_suricata([]) == []


def test_modes():
    f = un_run("real-healthy")
    try:
        correlation_ids_suricata(f, mode="regles"); assert False
    except NotImplementedError:
        pass
    assert correlation_ids_suricata(f, mode="regles", regles=lambda x: ["regles-p2"]) == ["regles-p2"]
    try:
        correlation_ids_suricata(f, mode="nimporte"); assert False
    except ValueError:
        pass


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    echecs = 0
    for nom, f in tests:
        try:
            f(); print(f"  OK   {nom}")
        except Exception as e:
            echecs += 1; print(f"  ECHEC {nom} : {type(e).__name__} {e}")
    print(f"\n{len(tests) - echecs}/{len(tests)} tests reussis")
    raise SystemExit(1 if echecs else 0)
