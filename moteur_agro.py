"""Moteur agronomique : pluie saisonnière, scénarios, besoin en eau (FAO-56),
plan d'irrigation, prix, date de récolte, texte explicatif.
Les 5 fichiers dataset_M*.csv doivent être dans le même dossier que ce fichier."""
import os
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

DATA = Path(__file__).resolve().parent

# ============================================================ M1 : pluie saisonnière
VARS_M1 = ["sst_atlantique", "enso_index", "humidite_atm", "pluie_saison_precedente"]


def _prep_m1(df, locs):
    X = df[VARS_M1].copy()
    for l in locs:
        X["loc_" + l] = (df.localite == l).astype(int)
    return X


@lru_cache(maxsize=1)
def _m1():
    d = pd.read_csv(DATA / "dataset_M1_pluie_saisonniere.csv")
    locs = sorted(d.localite.unique())
    modele = Ridge(alpha=1.0).fit(_prep_m1(d, locs), d.pluie_saison_cible)
    resid = d.pluie_saison_cible - modele.predict(_prep_m1(d, locs))
    stats = d.groupby("localite").pluie_saison_cible.agg(["mean", "std"])
    return {"modele": modele, "locs": locs, "sigma": float(resid.std()), "stats": stats}


def localites():
    return list(_m1()["locs"])


def predire_saison(localite, sst=0.0, enso=0.0, humidite=45.0, pluie_precedente=None):
    o = _m1()
    if pluie_precedente is None:
        pluie_precedente = o["stats"].loc[localite, "mean"]
    ligne = pd.DataFrame([{"localite": localite, "sst_atlantique": sst, "enso_index": enso,
                           "humidite_atm": humidite, "pluie_saison_precedente": pluie_precedente}])
    pred = float(o["modele"].predict(_prep_m1(ligne, o["locs"]))[0])
    moy, ecart = o["stats"].loc[localite, "mean"], o["stats"].loc[localite, "std"]
    z = (pred - moy) / ecart
    cat = "sèche" if z < -0.5 else "humide" if z > 0.5 else "normale"
    return {"localite": localite, "pluie_prevue_mm": round(pred),
            "intervalle_80pct_mm": (round(pred - 1.28 * o["sigma"]), round(pred + 1.28 * o["sigma"])),
            "categorie": cat}


# ============================================================ M2 : scénarios de pluie
@lru_cache(maxsize=1)
def _profils():
    m2 = pd.read_csv(DATA / "dataset_M2_scenarios_journaliers.csv")
    return {loc: g.sort_values(["annee", "scenario_id", "jour_saison"])
                  .pluie_journaliere_mm.values.reshape(-1, 120)
            for loc, g in m2.groupby("localite")}


def generer_scenarios(localite, sst=0.0, enso=0.0, humidite=45.0, graine=42):
    sigma = _m1()["sigma"]
    pred = predire_saison(localite, sst, enso, humidite)["pluie_prevue_mm"]
    rng = np.random.default_rng(graine)
    profils = _profils()[localite]
    profil = profils[rng.integers(len(profils))]
    profil = profil / profil.sum()
    scenarios = []
    for nom, prob, z in [("sèche", 0.25, -1), ("normale", 0.50, 0), ("humide", 0.25, 1)]:
        cumul = max(pred + z * sigma, 300)
        scenarios.append({"nom": nom, "probabilite": prob,
                          "cumul_mm": round(cumul), "pluie_jour": profil * cumul})
    return scenarios


# ============================================================ M3 : besoin en eau (FAO-56)
KC = {"mais": (0.3, 1.2, 0.5), "riz": (1.05, 1.2, 0.9), "manioc": (0.3, 0.8, 0.5),
      "igname": (0.4, 1.0, 0.6), "tomate": (0.6, 1.15, 0.8)}
PHASES = {"mais": (6, 28, 18, 28, 18), "riz": (8, 35, 22, 28, 22), "igname": (18, 75, 0, 75, 38),
          "manioc": (12, 105, 0, 150, 75), "tomate": (6, 28, 18, 22, 12)}
NOM_CULTURE = {"mais": "maïs", "riz": "riz", "igname": "igname", "manioc": "manioc", "tomate": "tomate"}


@lru_cache(maxsize=1)
def _et0_moy():
    m3 = pd.read_csv(DATA / "dataset_M3_etc_fao56.csv")
    return m3.groupby("localite").et0_mm.mean()


def courbe_kc(culture):
    ini, mid, fin = KC[culture]
    levee, veg, flo, rem, mat = PHASES[culture]
    kc = [ini] * levee
    kc += list(np.linspace(ini, mid, veg + 1)[1:])
    kc += [mid] * (flo + rem)
    kc += list(np.linspace(mid, fin, mat + 1)[1:])
    return np.array(kc)


def calculer_etc(localite, culture):
    kc = courbe_kc(culture)
    et0 = _et0_moy()[localite]
    return {"duree_jours": len(kc), "et0_mm_jour": round(et0, 2),
            "etc_jour": kc * et0, "etc_total_mm": round(float((kc * et0).sum()))}


# ============================================================ Bilan hydrique et irrigation
FC, PWP, SEUIL, IRR_MAX = 100.0, 30.0, 20.0, 30.0   # mm
FACTEUR_ET0 = 1.0                                     # laisser 1.0 (l'ET0 du dataset M3 est élevée)


def pluie_efficace(p):
    return 0.8 * p if p <= 75 else 0.3 * p + 37.5


def bilan(pluie, etc):
    sw, lignes = FC, []
    for p, e in zip(pluie, etc):
        pe = pluie_efficace(p)
        sw = sw + pe - e
        drainage = max(0.0, sw - FC)
        sw = min(max(sw, PWP), FC)
        deficit = FC - sw
        irr = min(deficit, IRR_MAX) if deficit > SEUIL else 0.0
        sw += irr
        lignes.append((p, pe, e, irr, sw, drainage))
    return pd.DataFrame(lignes, columns=["pluie_mm", "pluie_eff_mm", "etc_mm",
                                         "irrigation_mm", "reserve_mm", "drainage_mm"]).round(1)


def plan_irrigation(localite, culture, superficie_ha, date_debut, sst=0.0, enso=0.0, humidite=45.0):
    etc = calculer_etc(localite, culture)
    n = min(etc["duree_jours"], 120)              # les scénarios de pluie couvrent 120 jours
    dates = pd.date_range(pd.Timestamp(date_debut), periods=n)
    res = {}
    for s in generer_scenarios(localite, sst, enso, humidite):
        t = bilan(s["pluie_jour"][:n], etc["etc_jour"][:n] * FACTEUR_ET0)
        t.insert(0, "date", dates)
        res[s["nom"]] = {"probabilite": s["probabilite"], "cumul_mm": s["cumul_mm"], "table": t}
    sem = pd.DataFrame({nom: r["table"].groupby(np.arange(n) // 7).irrigation_mm.sum()
                        for nom, r in res.items()}).round(0)
    sem["attendue"] = sum(r["probabilite"] * sem[nom] for nom, r in res.items()).round(0)
    sem.insert(0, "semaine_du", dates[::7].strftime("%d/%m/%Y"))
    tot = {nom: round(r["table"].irrigation_mm.sum()) for nom, r in res.items()}
    tot_att = round(sum(r["probabilite"] * tot[nom] for nom, r in res.items()))
    return {"localite": localite, "culture": culture, "superficie_ha": superficie_ha,
            "jours_planifies": n, "duree_cycle": etc["duree_jours"],
            "plan_jour": res["normale"]["table"], "resume_semaine": sem,
            "total_mm": tot, "total_attendu_mm": tot_att,
            "volume_attendu_m3": round(tot_att * 10 * superficie_ha), "scenarios": res}


# ============================================================ M4 : prix et récolte
STOCKAGE_MAX_MOIS = {"mais": 6, "riz": 6, "igname": 4, "manioc": 0, "tomate": 0}  # 0 = périssable
_cache_prix = {}


@lru_cache(maxsize=1)
def _m4():
    return pd.read_csv(DATA / "dataset_M4_prix_marche.csv")


def serie_prix(localite, culture):
    m4 = _m4()
    s = m4[(m4.localite == localite) & (m4.culture == culture)].sort_values(["annee", "mois"])
    return pd.Series(s.prix_moyen.values, index=pd.period_range("2015-01", periods=len(s), freq="M"))


def prevoir_prix(localite, culture, jusqu_a):
    """Prix mensuel prévu (FCFA/kg) de janvier 2025 jusqu'au mois 'jusqu_a' (ex. '2028-02').
    Modèle : tendance linéaire + effet du mois, ajusté sur 2015-2024."""
    cle = (localite, culture)
    if cle not in _cache_prix:
        y = serie_prix(localite, culture)
        t = np.arange(len(y)); mois = np.array([i.month for i in y.index])
        X = np.column_stack([t] + [(mois == k).astype(float) for k in range(1, 13)])
        coef = np.linalg.lstsq(X, y.values, rcond=None)[0]
        sigma = float((y.values - X @ coef).std())
        _cache_prix[cle] = (coef, sigma, len(y))
    coef, sigma, n = _cache_prix[cle]
    h = (pd.Period(jusqu_a, freq="M") - pd.Period("2024-12", freq="M")).n
    tf = np.arange(n, n + h); mf = np.array([(i % 12) + 1 for i in range(n, n + h)])
    Xf = np.column_stack([tf] + [(mf == k).astype(float) for k in range(1, 13)])
    prix = Xf @ coef
    idx = pd.period_range("2025-01", periods=h, freq="M")
    return pd.DataFrame({"prix": prix.round(0), "bas": (prix - 1.28 * sigma).round(0),
                         "haut": (prix + 1.28 * sigma).round(0)}, index=idx)


def planifier_recolte(localite, culture, date_debut, production_t=None, part_vente_immediate=0.45):
    duree = sum(PHASES[culture])
    date_recolte = pd.Timestamp(date_debut) + pd.Timedelta(days=duree)
    m_rec = date_recolte.to_period("M")
    stock_max = STOCKAGE_MAX_MOIS[culture]
    prix = prevoir_prix(localite, culture, str(m_rec + stock_max))
    fenetre = prix.loc[m_rec: m_rec + stock_max]
    p_rec = float(fenetre.prix.iloc[0])
    m_best = fenetre.prix.idxmax()
    p_best = float(fenetre.prix.max())
    gain_kg = p_best - p_rec
    rec = {"culture": culture, "localite": localite, "date_recolte": date_recolte.date(),
           "prix_recolte": p_rec, "mois_stockage": str(m_best), "prix_stockage": p_best,
           "gain_par_tonne": round(gain_kg * 1000), "fenetre": fenetre}
    if stock_max == 0 or gain_kg <= 0:
        rec["conseil"] = "vendre à la récolte"
    else:
        rec["conseil"] = f"stocker jusqu'en {m_best}"
    if production_t:
        stock_t = 0.0 if rec["conseil"] == "vendre à la récolte" else round(production_t * (1 - part_vente_immediate), 1)
        rec.update({"production_t": production_t, "a_stocker_t": stock_t,
                    "a_vendre_recolte_t": round(production_t - stock_t, 1),
                    "revenu_si_tout_vendu_recolte": round(production_t * 1000 * p_rec),
                    "revenu_avec_strategie": round((production_t - stock_t) * 1000 * p_rec + stock_t * 1000 * p_best)})
        rec["gain_total"] = rec["revenu_avec_strategie"] - rec["revenu_si_tout_vendu_recolte"]
    return rec


# ============================================================ Fonction unique
def recommander(localite, culture, superficie_ha, date_debut, production_t=None,
                sst=0.0, enso=0.0, humidite=45.0):
    if localite not in localites():
        raise ValueError(f"Localité inconnue : {localite}. Choix : {', '.join(localites())}")
    if culture not in KC:
        raise ValueError(f"Culture inconnue : {culture}. Choix : {', '.join(KC)}")
    if superficie_ha <= 0:
        raise ValueError("La superficie doit être supérieure à 0 hectare.")
    try:
        pd.Timestamp(date_debut)
    except Exception:
        raise ValueError("Date invalide. Format attendu : AAAA-MM-JJ (ex. 2027-05-15).")

    saison = predire_saison(localite, sst, enso, humidite)
    irr = plan_irrigation(localite, culture, superficie_ha, date_debut, sst, enso, humidite)
    rec = planifier_recolte(localite, culture, date_debut, production_t)
    sem = irr["resume_semaine"]
    pic = sem.loc[sem.attendue.idxmax()]

    resume = {
        "localite": localite, "culture": culture, "superficie_ha": float(superficie_ha),
        "date_semis": str(pd.Timestamp(date_debut).date()), "duree_cycle_jours": int(irr["duree_cycle"]),
        "date_recolte": str(rec["date_recolte"]),
        "saison": saison["categorie"], "pluie_saison_prevue_mm": int(saison["pluie_prevue_mm"]),
        "pluie_saison_intervalle_mm": [int(x) for x in saison["intervalle_80pct_mm"]],
        "jours_planifies_irrigation": int(irr["jours_planifies"]),
        "irrigation_attendue_mm": int(irr["total_attendu_mm"]),
        "irrigation_scenario_sec_mm": int(irr["total_mm"]["sèche"]),
        "irrigation_scenario_humide_mm": int(irr["total_mm"]["humide"]),
        "volume_eau_m3": int(irr["volume_attendu_m3"]),
        "semaine_la_plus_exigeante": {"debut": pic["semaine_du"], "irrigation_mm": int(pic["attendue"])},
        "jours_irrigation_scenario_normal": int((irr["plan_jour"].irrigation_mm > 0).sum()),
        "prix_recolte_fcfa_kg": int(rec["prix_recolte"]), "conseil_recolte": rec["conseil"],
    }
    if rec["conseil"] != "vendre à la récolte":
        resume.update({"mois_meilleur_prix": rec["mois_stockage"],
                       "prix_meilleur_fcfa_kg": int(rec["prix_stockage"]),
                       "gain_par_tonne_stockee_fcfa": int(rec["gain_par_tonne"])})
    if production_t:
        resume.update({"production_t": float(production_t), "a_vendre_recolte_t": float(rec["a_vendre_recolte_t"]),
                       "a_stocker_t": float(rec["a_stocker_t"]), "gain_total_fcfa": int(rec["gain_total"])})
    return {"saison": saison, "irrigation": irr, "recolte": rec, "resume": resume}


# ============================================================ Texte explicatif
CONSIGNES = """Tu es un conseiller agricole qui parle à un agriculteur du Bénin.
Tu reçois en JSON le résultat d'un moteur de calcul (pluie, irrigation, prix, récolte). Tu ne prédis RIEN toi-même : tu expliques ces chiffres.

Lexique des champs (à respecter exactement) :
- date_semis : date du semis, début de la culture. date_recolte : date de récolte prévue.
- saison : type de saison des pluies attendue (sèche, normale ou humide) ; pluie_saison_prevue_mm et pluie_saison_intervalle_mm : pluie totale de la saison et sa fourchette.
- irrigation_attendue_mm : eau à apporter en plus de la pluie, au total ; volume_eau_m3 : la même quantité en m3 pour toute la parcelle.
- irrigation_scenario_sec_mm / irrigation_scenario_humide_mm : eau à apporter si la saison est plus sèche / plus humide.
- jours_planifies_irrigation : nombre de jours couverts par le calendrier d'irrigation ; jours_irrigation_scenario_normal : nombre de jours où il faut réellement irriguer dans ce calendrier.
- semaine_la_plus_exigeante : semaine où il faut le plus d'eau (debut = premier jour de la semaine).

Règles strictes :
- Réponds en français simple, phrases courtes, sans jargon (ne dis pas ETc, ET0, SARIMA, scénario, probabilité).
- Utilise UNIQUEMENT les chiffres du JSON. N'invente aucun chiffre, aucune date, aucun conseil technique absent du JSON.
- Si une information est absente du JSON (production, gain, stockage), n'en parle pas.
- Ne dis jamais que l'eau est « répartie sur N jours » : dis « il faudra irriguer N jours au total ».
- Si jours_planifies_irrigation est inférieur à duree_cycle_jours, précise que le plan d'irrigation couvre seulement les premiers jours du cycle.
- Écris les dates au format jour/mois/année, les mois en lettres (janvier 2028, jamais 01/2028), les nombres à quatre chiffres ou plus avec un espace (110 000 ; 6 040), les prix en FCFA/kg.
- Structure : 1) La saison de pluie ; 2) L'eau à apporter ; 3) La récolte et la vente ; 4) Une phrase de prudence : ce sont des estimations, à confirmer avec un conseiller agricole.
- Maximum 170 mots. Pas de titres markdown, des paragraphes courts. Termine toujours par la phrase de prudence."""

MOIS = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août",
        "septembre", "octobre", "novembre", "décembre"]


def _date(iso):
    a, m, j = iso.split("-")
    return f"{j}/{m}/{a}"


def _n(x):
    """Nombre à la française : 40 000 ; 5,4"""
    if isinstance(x, float) and not float(x).is_integer():
        return f"{x:g}".replace(".", ",")
    return f"{int(x):,}".replace(",", " ")


def _mois(aaaa_mm):
    a, m = aaaa_mm.split("-")
    return f"{MOIS[int(m) - 1]} {a}"


def texte_hors_ligne(res):
    """Explication de secours (sans IA), à partir des mêmes chiffres."""
    cult = NOM_CULTURE[res["culture"]]
    de = "d'" if cult[0] in "aeiouéèh" else "de "
    t = [f"Votre parcelle {de}{cult} à {res['localite']} ({_n(res['superficie_ha'])} ha, semis le "
         f"{_date(res['date_semis'])}) : la saison de pluie devrait être {res['saison']}, avec environ "
         f"{_n(res['pluie_saison_prevue_mm'])} mm."]
    bas = min(res["irrigation_scenario_sec_mm"], res["irrigation_scenario_humide_mm"])
    haut = max(res["irrigation_scenario_sec_mm"], res["irrigation_scenario_humide_mm"])
    eau = (f"Prévoyez environ {res['irrigation_attendue_mm']} mm d'irrigation sur {res['jours_planifies_irrigation']} "
           f"jours, soit environ {_n(res['volume_eau_m3'])} m3 pour la parcelle (entre {bas} et {haut} mm selon la pluie). "
           f"La semaine la plus exigeante commence le {res['semaine_la_plus_exigeante']['debut']} "
           f"(environ {res['semaine_la_plus_exigeante']['irrigation_mm']} mm).")
    if res["jours_planifies_irrigation"] < res["duree_cycle_jours"]:
        eau += (f" Ce plan couvre seulement les {res['jours_planifies_irrigation']} premiers jours "
                f"d'un cycle de {res['duree_cycle_jours']} jours.")
    t.append(eau)
    v = (f"Récolte prévue vers le {_date(res['date_recolte'])}, avec un prix attendu de "
         f"{_n(res['prix_recolte_fcfa_kg'])} FCFA/kg.")
    if "mois_meilleur_prix" in res:
        v += (f" Le prix devrait être meilleur en {_mois(res['mois_meilleur_prix'])} ({_n(res['prix_meilleur_fcfa_kg'])} FCFA/kg) : "
              f"environ {_n(res['gain_par_tonne_stockee_fcfa'])} FCFA de plus par tonne stockée.")
    else:
        v += " Il vaut mieux vendre à la récolte."
    if "production_t" in res and "gain_total_fcfa" in res:
        v += (f" Sur {_n(res['production_t'])} t : vendre {_n(res['a_vendre_recolte_t'])} t à la récolte et stocker "
              f"{_n(res['a_stocker_t'])} t, pour un gain estimé de {_n(res['gain_total_fcfa'])} FCFA.")
    t.append(v)
    t.append("Ces chiffres sont des estimations : à confirmer avec un conseiller agricole.")
    return "\n\n".join(t)


MODELES = ["gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-2.5-flash"]
NOMS_SECRET = ["GEMINI_API_KEY", "Gemini API Key", "GOOGLE_API_KEY"]


def _cle_api():
    """Cherche la clé : secrets Streamlit, secrets Colab, puis variables d'environnement."""
    try:
        import streamlit as st
        for nom in NOMS_SECRET:
            if nom in st.secrets:
                return st.secrets[nom]
    except Exception:
        pass
    try:
        from google.colab import userdata
        for nom in NOMS_SECRET:
            try:
                return userdata.get(nom)
            except Exception:
                continue
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def expliquer(resume):
    """Texte en français simple (Gemini si la clé existe, sinon texte de secours)."""
    cle = _cle_api()
    if not cle:
        return ("[Mode hors ligne : clé API introuvable — vérifiez le secret GEMINI_API_KEY]\n\n"
                + texte_hors_ligne(resume))
    derniere = None
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=cle)
        for modele in MODELES:
            try:
                rep = client.models.generate_content(
                    model=modele,
                    contents=json.dumps(resume, ensure_ascii=False),
                    config=types.GenerateContentConfig(system_instruction=CONSIGNES,
                                                       temperature=0.3, max_output_tokens=8192))
                fin = str(rep.candidates[0].finish_reason) if rep.candidates else ""
                if "MAX_TOKENS" in fin:
                    derniere = ValueError("réponse tronquée")
                    continue
                if rep.text and rep.text.strip():
                    return rep.text
                derniere = ValueError("réponse vide")
            except Exception as e:
                derniere = e
    except Exception as e:
        derniere = e
    return f"[Mode hors ligne : appel à Gemini impossible ({type(derniere).__name__})]\n\n" + texte_hors_ligne(resume)


DEMOS = [("Borgou", "mais", 2, "2027-05-15", 5.4),
         ("Littoral", "tomate", 0.5, "2027-11-10", None),
         ("Donga", "igname", 1.5, "2027-03-15", None)]

if __name__ == "__main__":
    for loc, cul, ha, d, prod in DEMOS:
        print(expliquer(recommander(loc, cul, ha, d, prod)["resume"]))
        print("=" * 50)
