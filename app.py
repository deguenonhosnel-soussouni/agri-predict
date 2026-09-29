import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st

import moteur_agro as m

st.set_page_config(page_title="Conseiller agricole – Bénin", page_icon="🌾", layout="wide")
st.title("🌾 Conseiller agricole – Bénin")
st.caption("Prototype : pluie de la saison, eau à apporter, prix et date de récolte. "
           "Les modèles sont entraînés sur des données simulées : les résultats sont des estimations.")

# ------------------------------------------------------------------ formulaire
with st.sidebar:
    st.header("Votre parcelle")
    localite = st.selectbox("Localité (département)", m.localites(), index=None, placeholder="Choisir…")
    culture = st.selectbox("Culture", list(m.KC), index=None, placeholder="Choisir…",
                           format_func=lambda c: m.NOM_CULTURE[c].capitalize())
    superficie = st.number_input("Superficie (hectares)", min_value=0.1, max_value=50.0,
                                 value=None, step=0.1, placeholder="ex. 2")
    date_semis = st.date_input("Date de semis", value=None, format="DD/MM/YYYY")
    production = st.number_input("Production estimée (tonnes) — facultatif", min_value=0.1,
                                 value=None, step=0.1, placeholder="ex. 5,4")
    with st.expander("Conditions climatiques (facultatif)"):
        st.caption("Laissez ces valeurs à zéro pour une saison « moyenne ».")
        sst = st.slider("Température de l'océan Atlantique (écart à la normale)", -1.5, 1.5, 0.0, 0.1)
        enso = st.slider("El Niño / La Niña (indice)", -2.0, 2.0, 0.0, 0.1)
        hum = st.slider("Humidité de l'air (%)", 30.0, 60.0, 45.0, 1.0)
    pret = all([localite, culture, superficie, date_semis])
    lancer = st.button("Calculer", type="primary", disabled=not pret)
    if not pret:
        st.caption("Renseignez la localité, la culture, la superficie et la date de semis.")

# ------------------------------------------------------------------ calcul
if lancer:
    try:
        with st.spinner("Calcul en cours…"):
            res = m.recommander(localite, culture, float(superficie), str(date_semis),
                                float(production) if production else None,
                                sst=sst, enso=enso, humidite=hum)
            texte = m.expliquer(res["resume"])
        st.session_state["res"], st.session_state["texte"] = res, texte
    except ValueError as e:
        st.session_state.pop("res", None)
        st.error(str(e))

res = st.session_state.get("res")
if res is None:
    st.info("Renseignez votre parcelle dans le menu de gauche, puis cliquez sur **Calculer**.")
    st.stop()

r, irr, rec, saison = res["resume"], res["irrigation"], res["recolte"], res["saison"]

# ------------------------------------------------------------------ indicateurs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Saison de pluie", r["saison"].capitalize(), f"{r['pluie_saison_prevue_mm']} mm")
c2.metric("Eau à apporter", f"{r['irrigation_attendue_mm']} mm", f"{r['volume_eau_m3']} m³", delta_color="off")
c3.metric("Récolte prévue", pd.Timestamp(r["date_recolte"]).strftime("%d/%m/%Y"), f"cycle de {r['duree_cycle_jours']} j",
          delta_color="off")
c4.metric("Prix à la récolte", f"{r['prix_recolte_fcfa_kg']} FCFA/kg", r["conseil_recolte"], delta_color="off")

tab1, tab2, tab3, tab4 = st.tabs(["💬 Conseil", "💧 Irrigation", "🌧️ Pluie", "💰 Prix et récolte"])

with tab1:
    texte = st.session_state.get("texte", "")
    if texte.startswith("[Mode hors ligne"):
        entete, _, corps = texte.partition("\n\n")
        st.caption(entete.strip("[]"))
        st.write(corps)
    else:
        st.write(texte)

with tab2:
    if r["jours_planifies_irrigation"] < r["duree_cycle_jours"]:
        st.warning(f"Le plan couvre les {r['jours_planifies_irrigation']} premiers jours d'un cycle de "
                   f"{r['duree_cycle_jours']} jours (portée des prévisions de pluie).")
    st.subheader("Eau à apporter par semaine (mm)")
    sem = irr["resume_semaine"].set_index("semaine_du")
    st.bar_chart(sem[["sèche", "normale", "humide"]])
    st.caption("Trois cas : saison plus sèche, normale ou plus humide. "
               f"Total attendu : {r['irrigation_attendue_mm']} mm "
               f"(entre {min(irr['total_mm'].values())} et {max(irr['total_mm'].values())} mm).")
    st.subheader("Jours où irriguer (saison normale)")
    j = irr["plan_jour"]
    j = j[j.irrigation_mm > 0][["date", "pluie_mm", "etc_mm", "irrigation_mm"]].copy()
    j["date"] = j["date"].dt.strftime("%d/%m/%Y")
    j.columns = ["Date", "Pluie (mm)", "Besoin de la culture (mm)", "Irriguer (mm)"]
    if j.empty:
        st.success("Aucune irrigation nécessaire dans ce scénario.")
    else:
        st.dataframe(j, hide_index=True)
    st.caption("1 mm d'eau sur 1 hectare = 10 m³.")

with tab3:
    st.subheader("Trois scénarios de pluie pour la saison")
    lignes = [{"Scénario": nom.capitalize(), "Probabilité": f"{int(s['probabilite'] * 100)} %",
               "Pluie totale (mm)": s["cumul_mm"]} for nom, s in irr["scenarios"].items()]
    st.dataframe(pd.DataFrame(lignes), hide_index=True)
    cumul = pd.DataFrame({nom.capitalize(): np.cumsum(s["table"].pluie_mm.values)
                          for nom, s in irr["scenarios"].items()})
    cumul.index = irr["scenarios"]["normale"]["table"]["date"].dt.strftime("%d/%m")
    st.line_chart(cumul)
    st.caption(f"Prévision M1 : {saison['pluie_prevue_mm']} mm, fourchette "
               f"{saison['intervalle_80pct_mm'][0]}–{saison['intervalle_80pct_mm'][1]} mm. "
               "On ne prédit pas la pluie d'un jour précis : on prédit la saison et on simule des scénarios.")

with tab4:
    st.subheader("Prix prévu du kilo (FCFA)")
    f = rec["fenetre"].copy()
    f.index = f.index.strftime("%m/%Y")
    st.line_chart(f.rename(columns={"prix": "Prix prévu", "bas": "Bas", "haut": "Haut"}))
    st.caption("Fourchette indicative (elle couvre environ 7 cas sur 10 dans nos tests).")
    st.write(f"**Conseil : {rec['conseil']}.**")
    if "mois_meilleur_prix" in r:
        a, mo = r["mois_meilleur_prix"].split("-")
        st.write(f"Meilleur prix attendu : **{r['prix_meilleur_fcfa_kg']} FCFA/kg** en {mo}/{a}, "
                 f"soit **{r['gain_par_tonne_stockee_fcfa']:,} FCFA de plus par tonne stockée**.".replace(",", " "))
    if "production_t" in r:
        d1, d2, d3 = st.columns(3)
        d1.metric("À vendre à la récolte", f"{r['a_vendre_recolte_t']} t".replace(".", ","))
        d2.metric("À stocker", f"{r['a_stocker_t']} t".replace(".", ","))
        d3.metric("Gain estimé", f"{r['gain_total_fcfa']:,} FCFA".replace(",", " "))
    else:
        st.caption("Indiquez votre production estimée pour calculer le gain total.")

st.divider()
st.caption("Limites : données simulées ; prix connus jusqu'en décembre 2024 ; hypothèses de stockage "
           "(maïs et riz 6 mois, igname 4 mois, manioc et tomate non stockables) à valider avec un agronome.")
