# Cours 4.3 — Post-traitement et agrégation de transcriptions

**Module 4 · Computer Vision appliquée aux manuscrits médiévaux · MD5**

---

> *« Un seul modèle fait des erreurs systématiques. Plusieurs modèles font des erreurs différentes. L'agrégation intelligente de plusieurs sources imparfaites produit une source moins imparfaite — c'est le principe fondateur de la philologie médiévale appliqué à l'apprentissage automatique. »*

---

## Introduction : de la reconnaissance à la transcription consolidée

Les sections 4.1 et 4.2 ont produit deux sorties parallèles pour chaque page de manuscrit :

- Le **pipeline modulaire** (SAM + TrOCR) : transcriptions de haute qualité, format JSON, traitement GPU.
- Le **pipeline Kraken intégré** (BLLA + HTR) : transcriptions rapides, format PAGE XML, traitement CPU, avec ordre de lecture et polygones.

Ces deux sorties ne sont pas redondantes — elles sont **complémentaires**. TrOCR, grâce à son décodeur GPT-2, gère mieux le contexte linguistique long et les abréviations ambiguës. Kraken, grâce à son extraction par baseline, produit des images de lignes géométriquement plus fidèles et est plus robuste sur les lignes courbées. Leurs erreurs ne se produisent pas aux mêmes endroits.

L'agrégation de ces deux sources — et potentiellement des transcriptions humaines de la section 1.3 — est l'objet de cette section. Elle produit une transcription consolidée, enrichie de scores de confiance par ligne et de flags signalant les passages incertains, qui constitue la livraison finale au module NLP.

Cette section est aussi le lieu de traiter les **cas limites** du pipeline : lignes trop courtes, mélange de langues, zones dégradées, confidences anormalement basses. Un pipeline de production robuste ne se contente pas de transcrire — il sait aussi *quand ne pas transcrire*, et comment signaler ses doutes.

---

## 1. Le principe des estimateurs faibles → estimateur fort

### 1.1 Rappel philologique

La section 1.2 a introduit cette idée dans le contexte de la transcription humaine : plusieurs annotateurs divergents, mis en accord par un mécanisme de consensus, produisent une transcription plus fiable que chacun pris isolément. C'est exactement le principe que les éditeurs critiques médiévaux appliquent depuis des siècles : comparer les *témoins* (les différents manuscrits qui transmettent un même texte), identifier les leçons communes, et retenir comme probable ce que plusieurs sources indépendantes attestent.

En apprentissage automatique, ce principe s'appelle les **méthodes d'ensemble** (*ensemble methods*). Il repose sur une hypothèse : les erreurs de modèles indépendants sont distribuées de façon quasi-aléatoire — elles ne se produisent pas aux mêmes positions dans les séquences transcrites. Si cette hypothèse est vérifiée, la probabilité que plusieurs modèles se trompent *de la même façon au même endroit* est bien inférieure à la probabilité d'erreur de chaque modèle pris seul.

**La condition d'efficacité des ensembles :** les modèles doivent être *suffisamment différents* pour que leurs erreurs soient décorrélées. Deux instances du même modèle TrOCR avec des seeds différents feraient les mêmes erreurs systématiques — leur agrégation n'apporterait rien. TrOCR (décodeur autorégressif, vocabulaire BPE anglais) et Kraken (LSTM-CTC, vocabulaire appris sur données médiévales) ont des architectures, des loss functions, et des données d'entraînement suffisamment différents pour que leurs erreurs soient effectivement décorrélées.

### 1.2 Vérification empirique de la décorrélation des erreurs

Avant d'agréger, il est utile de vérifier que les erreurs sont effectivement décorrélées sur notre corpus de validation.

```python
import numpy as np
import editdistance
from collections import defaultdict
import matplotlib.pyplot as plt

def analyser_correlation_erreurs(
    predictions_modele_a: list[str],
    predictions_modele_b: list[str],
    references: list[str],
    noms: tuple = ("Modèle A", "Modèle B")
) -> dict:
    """
    Analyse la corrélation entre les erreurs de deux modèles.
    Pour chaque ligne, calcule si les deux modèles se trompent
    simultanément ou de façon indépendante.

    Returns:
        Dict avec fractions d'erreurs corrélées / décorrélées
        et matrice de contingence.
    """
    # Classifier chaque ligne : correct (0) ou erreur (1) pour chaque modèle
    erreurs_a = [
        1 if editdistance.eval(p, r) > 0 else 0
        for p, r in zip(predictions_modele_a, references)
    ]
    erreurs_b = [
        1 if editdistance.eval(p, r) > 0 else 0
        for p, r in zip(predictions_modele_b, references)
    ]

    # Matrice de contingence
    #                   Modèle B correct  Modèle B erroné
    # Modèle A correct  [0,0]             [0,1]
    # Modèle A erroné   [1,0]             [1,1]
    matrice = defaultdict(int)
    for a, b in zip(erreurs_a, erreurs_b):
        matrice[(a, b)] += 1

    n = len(references)
    resultats = {
        "les_deux_corrects"  : matrice[(0, 0)] / n,
        "seulement_a_errone" : matrice[(1, 0)] / n,
        "seulement_b_errone" : matrice[(0, 1)] / n,
        "les_deux_errones"   : matrice[(1, 1)] / n,
    }

    # Coefficient de corrélation phi (comme Pearson pour variables binaires)
    n00, n01 = matrice[(0,0)], matrice[(0,1)]
    n10, n11 = matrice[(1,0)], matrice[(1,1)]
    denom = np.sqrt((n00+n01)*(n10+n11)*(n00+n10)*(n01+n11))
    phi = (n00*n11 - n01*n10) / (denom + 1e-8)
    resultats["phi"] = float(phi)

    print(f"\n=== Corrélation des erreurs : {noms[0]} vs {noms[1]} ===")
    print(f"  Les deux corrects     : {resultats['les_deux_corrects']:.1%}")
    print(f"  Seul {noms[0]:10s} erroné : {resultats['seulement_a_errone']:.1%}")
    print(f"  Seul {noms[1]:10s} erroné : {resultats['seulement_b_errone']:.1%}")
    print(f"  Les deux erronés      : {resultats['les_deux_errones']:.1%}")
    print(f"  Coefficient phi       : {phi:.3f}  "
          f"({'forte corrélation' if abs(phi) > 0.5 else 'faible corrélation — ensemble utile'})")

    # Visualisation
    fig, ax = plt.subplots(figsize=(6, 5))
    mat_data = np.array([[n00, n01], [n10, n11]])
    im = ax.imshow(mat_data, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"{noms[1]} correct", f"{noms[1]} erroné"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels([f"{noms[0]} correct", f"{noms[0]} erroné"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{mat_data[i,j]}\n({mat_data[i,j]/n:.1%})",
                    ha="center", va="center", fontsize=10,
                    color="white" if mat_data[i,j] > mat_data.max()/2 else "black")
    ax.set_title(f"Matrice de contingence des erreurs\nφ = {phi:.3f}")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig("figures/correlation_erreurs.png", dpi=150)
    plt.show()

    return resultats
```

**Interprétation :** un coefficient phi proche de 0 indique des erreurs décorrélées — l'agrégation sera efficace. Un phi proche de 1 indique que les deux modèles échouent sur les mêmes lignes — l'agrégation n'apportera pas grand-chose. Si phi est élevé, cela suggère que les deux modèles ont un mode de défaillance commun (peut-être lié aux dégradations physiques du document) qu'il faudra traiter autrement.

---

## 2. L'alignement de séquences : comment comparer deux transcriptions

### 2.1 Le problème de l'alignement

Deux transcriptions d'une même ligne peuvent diverger non seulement par des substitutions de caractères, mais aussi par des insertions ou des suppressions qui décalent l'alignement de tous les caractères suivants.

Exemple :
```
Référence : "li rois Artus regnoit"
TrOCR     : "li rois Artus regnort"    → substitution de 'i' par 'r' en fin
Kraken    : "li ois Artus regnoit"     → suppression du 'r' de 'rois'
```

Pour voter caractère par caractère, il faut d'abord **aligner** les séquences — déterminer quelle position de TrOCR correspond à quelle position de Kraken. Sans alignement, un décalage par insertion crée des erreurs artificielles sur tous les caractères suivants.

### 2.2 La distance de Levenshtein et l'alignement de Needleman-Wunsch

La distance de Levenshtein (vue en section 1.2) donne un *score* mais pas l'*alignement* lui-même. L'algorithme de **Needleman-Wunsch** (développé en bioinformatique pour l'alignement de séquences d'ADN, et ici réappliqué aux transcriptions) produit à la fois le score et l'alignement optimal.

L'alignement est une mise en correspondance de chaque caractère d'une séquence avec soit un caractère de l'autre séquence (appariement), soit un espace (gap — insertion ou suppression) :

```
Référence : l  i     r  o  i  s     A  r  t  u  s
TrOCR     : l  i     r  o  i  s     A  r  t  u  s
Kraken    : l  i  -  -  o  i  s     A  r  t  u  s
                ↑  ↑
            gaps = le 'r' a été supprimé
```

```python
def needleman_wunsch(
    seq1: str,
    seq2: str,
    score_match: int = 2,
    score_mismatch: int = -1,
    score_gap: int = -1
) -> tuple[str, str, int]:
    """
    Alignement global optimal de deux séquences (Needleman-Wunsch, 1970).
    Retourne les deux séquences alignées (avec '-' pour les gaps) et le score.

    Ce même algorithme est utilisé en bioinformatique pour aligner des séquences
    d'ADN ou de protéines — son application aux transcriptions est directe.

    Args:
        seq1, seq2       : séquences à aligner
        score_match      : score pour un appariement correct
        score_mismatch   : pénalité pour une substitution
        score_gap        : pénalité pour un gap (insertion/suppression)

    Returns:
        seq1_alignee, seq2_alignee : séquences avec gaps insérés
        score                      : score optimal de l'alignement
    """
    m, n = len(seq1), len(seq2)

    # ── Initialisation de la matrice de programmation dynamique ─────────────
    dp = np.zeros((m + 1, n + 1), dtype=int)
    dp[:, 0] = [score_gap * i for i in range(m + 1)]
    dp[0, :] = [score_gap * j for j in range(n + 1)]

    # ── Remplissage ─────────────────────────────────────────────────────────
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            score_diag = (score_match
                         if seq1[i-1] == seq2[j-1]
                         else score_mismatch)
            dp[i, j] = max(
                dp[i-1, j-1] + score_diag,   # Substitution ou appariement
                dp[i-1, j]   + score_gap,    # Gap dans seq2
                dp[i, j-1]   + score_gap,    # Gap dans seq1
            )

    # ── Retrace (traceback) ─────────────────────────────────────────────────
    seq1_alignee, seq2_alignee = [], []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            score_diag = (score_match
                         if seq1[i-1] == seq2[j-1]
                         else score_mismatch)
        if (i > 0 and j > 0
                and dp[i, j] == dp[i-1, j-1] + score_diag):
            seq1_alignee.append(seq1[i-1])
            seq2_alignee.append(seq2[j-1])
            i -= 1; j -= 1
        elif i > 0 and dp[i, j] == dp[i-1, j] + score_gap:
            seq1_alignee.append(seq1[i-1])
            seq2_alignee.append("-")
            i -= 1
        else:
            seq1_alignee.append("-")
            seq2_alignee.append(seq2[j-1])
            j -= 1

    return ("".join(reversed(seq1_alignee)),
            "".join(reversed(seq2_alignee)),
            int(dp[m, n]))


def aligner_multiple(sequences: list[str]) -> list[str]:
    """
    Alignement multiple progressif de N séquences.
    Stratégie : aligner d'abord les deux séquences les plus proches,
    puis ajouter les suivantes une par une.
    """
    if len(sequences) == 1:
        return sequences
    if len(sequences) == 2:
        a1, a2, _ = needleman_wunsch(sequences[0], sequences[1])
        return [a1, a2]

    # Calculer la matrice de distances entre toutes les paires
    n = len(sequences)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = editdistance.eval(sequences[i], sequences[j])
            distances[i, j] = distances[j, i] = d

    # Aligner d'abord les deux séquences les plus proches
    idx_i, idx_j = np.unravel_index(
        np.argmin(distances + np.eye(n) * 9999), (n, n)
    )
    a1, a2, _ = needleman_wunsch(sequences[idx_i], sequences[idx_j])
    alignees = [None] * n
    alignees[idx_i] = a1
    alignees[idx_j]  = a2

    # Ajouter les autres séquences
    restantes = [k for k in range(n) if k not in (idx_i, idx_j)]
    for k in restantes:
        # Aligner avec la première séquence déjà alignée
        reference = alignees[idx_i]
        a_ref, a_new, _ = needleman_wunsch(reference, sequences[k])
        # Propager les gaps de la référence aux autres séquences déjà alignées
        for l in range(n):
            if alignees[l] is not None:
                seq_propagee = []
                pos_ref = 0
                for c in a_ref:
                    if c == "-" and pos_ref >= len(alignees[l]):
                        seq_propagee.append("-")
                    elif pos_ref < len(alignees[l]):
                        seq_propagee.append(alignees[l][pos_ref])
                        if c != "-":
                            pos_ref += 1
                    else:
                        seq_propagee.append("-")
                alignees[l] = "".join(seq_propagee)
        alignees[k] = a_new

    return [a for a in alignees if a is not None]
```

---

## 3. Le vote pondéré par la confiance

### 3.1 Principe du vote

Une fois les séquences alignées, on peut voter position par position : pour chaque colonne de l'alignement multiple, le caractère retenu est celui qui obtient le score pondéré le plus élevé, où chaque vote est pondéré par la confiance du modèle source à cette position.

Formellement, pour une colonne $c$ de l'alignement multiple avec $M$ modèles :

$$\hat{y}_c = \argmax_{v \in \mathcal{V} \cup \{-\}} \sum_{m=1}^{M} w_m \cdot \mathbf{1}[y_c^{(m)} = v]$$

où $w_m$ est le poids du modèle $m$ à la position $c$, $y_c^{(m)}$ est le caractère prédit par le modèle $m$ à la position $c$, et $\mathcal{V}$ est l'alphabet des caractères possibles (incluant le gap `-`).

```python
from collections import Counter
from typing import Optional

def voter_position_par_position(
    sequences_alignees: list[str],
    confidences: list[float],
    seuil_gap: float = 0.5
) -> tuple[str, list[float]]:
    """
    Vote pondéré par la confiance sur des séquences préalablement alignées.

    Args:
        sequences_alignees : séquences avec gaps '-' (même longueur)
        confidences        : poids de chaque séquence (confiance du modèle source)
        seuil_gap          : si le vote pondéré pour '-' dépasse ce seuil,
                             la position est ignorée (probable gap réel)

    Returns:
        sequence_finale    : séquence gagnante (sans les gaps)
        scores_position    : score de confiance par position
    """
    n_positions = len(sequences_alignees[0])
    n_modeles   = len(sequences_alignees)

    # Normaliser les poids
    poids = np.array(confidences)
    poids = poids / (poids.sum() + 1e-8)

    sequence_finale = []
    scores_position = []

    for pos in range(n_positions):
        # Récupérer les votes à cette position
        votes: dict[str, float] = {}
        for m in range(n_modeles):
            c = sequences_alignees[m][pos]
            votes[c] = votes.get(c, 0.0) + poids[m]

        # Trouver le caractère gagnant
        gagnant = max(votes, key=votes.get)
        score_gagnant = votes[gagnant]

        # Si le gap gagne avec un score élevé → position ignorée
        if gagnant == "-" and score_gagnant >= seuil_gap:
            continue   # On ne l'ajoute pas à la séquence finale

        # Si un caractère non-gap gagne, l'utiliser
        if gagnant != "-":
            sequence_finale.append(gagnant)
            scores_position.append(score_gagnant)
        else:
            # Le gap gagne mais avec un score faible → prendre le 2e meilleur
            votes_sans_gap = {k: v for k, v in votes.items() if k != "-"}
            if votes_sans_gap:
                second = max(votes_sans_gap, key=votes_sans_gap.get)
                sequence_finale.append(second)
                scores_position.append(votes_sans_gap[second])

    return "".join(sequence_finale), scores_position


def agreger_transcriptions(
    transcriptions: dict[str, tuple[str, float]],
    methode: str = "vote_pondere"
) -> tuple[str, float, dict]:
    """
    Point d'entrée principal pour l'agrégation de transcriptions.

    Args:
        transcriptions : {nom_modele: (transcription, confiance_globale)}
                         ex: {"TrOCR": ("En cel tems", 0.89),
                              "Kraken": ("En cel tem", 0.72)}
        methode        : "vote_pondere" | "meilleur" | "plus_long"

    Returns:
        transcription_finale, confiance_finale, details
    """
    noms     = list(transcriptions.keys())
    textes   = [transcriptions[n][0] for n in noms]
    poids    = [transcriptions[n][1] for n in noms]

    if len(textes) == 1:
        return textes[0], poids[0], {"methode": "unique"}

    if methode == "meilleur":
        # Retenir simplement la transcription avec la meilleure confiance
        idx_max = int(np.argmax(poids))
        return textes[idx_max], poids[idx_max], {"gagnant": noms[idx_max]}

    if methode == "vote_pondere":
        # Alignement multiple + vote
        sequences_alignees = aligner_multiple(textes)
        texte_final, scores = voter_position_par_position(
            sequences_alignees, poids
        )
        confiance_finale = float(np.mean(scores)) if scores else 0.0

        return texte_final, confiance_finale, {
            "methode"           : "vote_pondere",
            "n_modeles"         : len(noms),
            "scores_par_pos"    : scores,
            "transcriptions_src": dict(zip(noms, textes)),
        }

    raise ValueError(f"Méthode inconnue : {methode}")
```

### 3.2 Gestion des scores de confiance par modèle

TrOCR et Kraken produisent des scores de confiance, mais ces scores ne sont pas directement comparables : TrOCR produit des log-probabilités softmax normalisées sur son vocabulaire BPE ; Kraken produit des probabilités CTC par caractère sur son alphabet. Les deux sont dans [0, 1] mais leurs distributions sont différentes.

Pour les rendre commensurables, on peut utiliser la **calibration de Platt** ou simplement les normaliser par rapport à leur distribution empirique sur le corpus de validation.

```python
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

class CalibreurConfiance:
    """
    Calibre les scores de confiance d'un modèle pour qu'ils correspondent
    à des probabilités réelles (P(transcription correcte | score)).

    La calibration transforme : score_brut → P(correct)
    ce qui permet de comparer des scores de modèles différents.
    """

    def __init__(self, methode: str = "isotonic"):
        """
        methode : "isotonic" (non-paramétrique, recommandé)
                  "platt"    (régression logistique, plus rapide)
        """
        if methode == "isotonic":
            self.calibreur = IsotonicRegression(out_of_bounds="clip")
        elif methode == "platt":
            self.calibreur = LogisticRegression(C=1.0)
        self.methode = methode
        self.est_entraine = False

    def entrainer(
        self,
        scores_bruts: list[float],
        labels: list[int]   # 1 = transcription correcte, 0 = incorrecte
    ) -> None:
        """
        Entraîne le calibreur sur le corpus de validation.
        scores_bruts : confiances produites par le modèle
        labels       : 1 si CER de la ligne = 0, 0 sinon
        """
        X = np.array(scores_bruts).reshape(-1, 1)
        y = np.array(labels)

        if self.methode == "isotonic":
            self.calibreur.fit(scores_bruts, y)
        else:
            self.calibreur.fit(X, y)
        self.est_entraine = True

    def calibrer(self, scores_bruts: list[float]) -> list[float]:
        """Transforme les scores bruts en probabilités calibrées."""
        if not self.est_entraine:
            return scores_bruts   # Retour direct si pas encore entraîné

        if self.methode == "isotonic":
            return list(self.calibreur.predict(scores_bruts))
        else:
            X = np.array(scores_bruts).reshape(-1, 1)
            return list(self.calibreur.predict_proba(X)[:, 1])


def construire_calibreurs(
    dataset_val,
    modele_trocr,
    modele_kraken,
    processor_trocr
) -> dict[str, CalibreurConfiance]:
    """
    Entraîne un calibreur de confiance pour chaque modèle
    sur le corpus de validation.
    Retourne un dict {nom_modele: calibreur}.
    """
    calibreurs = {
        "TrOCR" : CalibreurConfiance("isotonic"),
        "Kraken": CalibreurConfiance("isotonic"),
    }

    scores_trocr, scores_kraken = [], []
    labels_trocr, labels_kraken = [], []

    for item in dataset_val:
        ref = item["text"]
        img = item["image"]

        # TrOCR
        pred_trocr, conf_trocr = transcrire_avec_trocr(
            modele_trocr, processor_trocr, img
        )
        scores_trocr.append(conf_trocr)
        labels_trocr.append(1 if editdistance.eval(pred_trocr, ref) == 0 else 0)

        # Kraken (si disponible)
        # pred_kraken, conf_kraken = transcrire_avec_kraken(...)
        # scores_kraken.append(conf_kraken)
        # labels_kraken.append(...)

    calibreurs["TrOCR"].entrainer(scores_trocr, labels_trocr)
    return calibreurs
```

### 3.3 Intégration des transcriptions humaines

Les transcriptions manuelles produites lors du TP de la section 1.3 sont le signal de supervision le plus fiable disponible. Quand une ligne a été transcrite manuellement et vérifiée, cette transcription doit prendre la priorité absolue dans l'agrégation — avec un poids très élevé.

```python
def agreger_avec_humain(
    transcriptions_auto: dict[str, tuple[str, float]],
    transcription_humaine: Optional[str] = None,
    poids_humain: float = 0.95
) -> tuple[str, float, dict]:
    """
    Agrégation avec priorité forte sur la transcription humaine si disponible.

    En pratique : si une ligne a été corrigée dans eScriptorium,
    la correction humaine est utilisée directement sans vote.
    Les transcriptions automatiques sont utilisées uniquement
    pour les lignes non encore corrigées.
    """
    if transcription_humaine is not None:
        # La transcription humaine prime : on la retourne directement
        # mais on note la divergence éventuelle avec les modèles auto
        divergences = {
            nom: editdistance.eval(transcription_humaine, pred) / max(len(transcription_humaine), 1)
            for nom, (pred, _) in transcriptions_auto.items()
        }
        return transcription_humaine, poids_humain, {
            "source"      : "humain",
            "divergences" : divergences,
            "needs_review": False
        }

    # Pas de transcription humaine → vote pondéré
    return agreger_transcriptions(transcriptions_auto, methode="vote_pondere")
```

---

## 4. Gestion des cas limites

### 4.1 Cartographie des cas limites

Un pipeline de production robuste doit identifier et traiter explicitement les situations où la transcription automatique est peu fiable. Voici les cas limites les plus fréquents dans notre contexte.

**Cas 1 — Ligne trop courte**

Une ligne avec moins de 3 à 5 caractères est souvent :
- Une réclame (mot en bas de page qui annonce la page suivante).
- Un numéro de folio.
- Une ligne partiellement tronquée par la segmentation.
- Un artefact de segmentation (non-texte segmenté comme texte).

Ces lignes ne méritent pas nécessairement d'être transcrites avec autant de soin que les lignes normales — et leur CER calculé sur des références courtes est peu significatif.

**Cas 2 — Mélange de langues**

Les manuscrits médiévaux mélangent fréquemment le latin et le vieux français. Un modèle entraîné sur du moyen français produira de mauvaises transcriptions sur les passages en latin, et vice versa. La détection automatique de la langue permet d'adapter le traitement.

**Cas 3 — Confiance anormalement basse**

Une confiance < 0.5 sur une ligne indique que le modèle est très incertain. Cela peut signifier : dégradation physique sévère (tache, déchirure), écriture très différente du corpus d'entraînement, ou présence de caractères hors alphabet (symboles alchimiques, notations musicales, dessins dans le texte).

**Cas 4 — Discordance extrême entre modèles**

Si TrOCR et Kraken produisent des transcriptions dont le CER est > 50%, c'est le signe que l'une des deux (au moins) est très mauvaise — ou que la ligne est réellement illisible. Dans ce cas, le vote pondéré est peu fiable.

**Cas 5 — Image de ligne vide ou presque vide**

Une image de ligne avec moins de 2% de pixels sombres ne contient probablement pas de texte. Cela peut arriver si la segmentation BLLA a détecté une fausse ligne dans l'espace inter-colonnes ou dans une zone d'illustration.

### 4.2 Code de gestion des cas limites

```python
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 42   # Reproductibilité

def detecter_langue(texte: str) -> tuple[str, float]:
    """
    Détecte la langue d'un texte court.
    Retourne (code_langue, confiance).
    Codes pertinents : 'fr' (français moderne), 'la' (latin), 'ca' (catalan)...

    Note : langdetect n'est pas entraîné sur l'ancien français —
    il peut confondre vieux français et latin ou catalan.
    Utiliser avec prudence sur des textes < 20 caractères.
    """
    if len(texte.strip()) < 5:
        return "inconnu", 0.0
    try:
        langue = detect(texte)
        return langue, 0.8   # langdetect ne retourne pas de score de confiance
    except Exception:
        return "inconnu", 0.0


def traiter_cas_limite(
    image_ligne_np: np.ndarray,
    transcription: str,
    confiance: float,
    transcription_b: Optional[str] = None,
    confiance_b: Optional[float] = None,
    seuil_longueur_min: int = 3,
    seuil_confiance_min: float = 0.5,
    seuil_densite_pixels: float = 0.02,
    seuil_discordance: float = 0.5
) -> dict:
    """
    Analyse une transcription pour détecter les cas limites.
    Retourne un dict de flags et de recommandations.
    """
    flags = []
    recommendations = []
    needs_review = False

    # ── Cas 1 : ligne trop courte ────────────────────────────────────────────
    if len(transcription.strip()) < seuil_longueur_min:
        flags.append("LIGNE_COURTE")
        recommendations.append(f"Transcription de {len(transcription)} caractères seulement — vérifier si c'est une réclame ou un artefact de segmentation")
        needs_review = True

    # ── Cas 2 : image presque vide (peu de pixels sombres) ───────────────────
    if image_ligne_np is not None:
        densite = float(np.mean(image_ligne_np < 100))
        if densite < seuil_densite_pixels:
            flags.append("IMAGE_VIDE")
            recommendations.append("Très peu de pixels sombres — probable non-texte")
            needs_review = True

    # ── Cas 3 : confiance basse ─────────────────────────────────────────────
    if confiance < seuil_confiance_min:
        flags.append("CONFIANCE_BASSE")
        recommendations.append(f"Confiance {confiance:.2f} < seuil {seuil_confiance_min} — zone dégradée ou écriture hors distribution")
        needs_review = True

    # ── Cas 4 : discordance entre modèles ───────────────────────────────────
    if transcription_b is not None:
        discordance = editdistance.eval(transcription, transcription_b) / max(
            len(transcription), len(transcription_b), 1
        )
        if discordance > seuil_discordance:
            flags.append("DISCORDANCE_MODELES")
            recommendations.append(f"CER entre modèles = {discordance:.1%} — transcription peu fiable")
            needs_review = True

    # ── Cas 5 : détection de langue non française ────────────────────────────
    langue, _ = detecter_langue(transcription)
    if langue not in ("fr", "ca", "pt", "es", "it", "inconnu"):
        # Probablement du latin ou un artefact de reconnaissance
        if langue == "la" or langue not in ["fr", "de", "en", "it", "es"]:
            flags.append(f"LANGUE_DETECTEE:{langue}")
            recommendations.append(f"Langue détectée : {langue} — possible latin ou artefact")

    return {
        "needs_review"    : needs_review,
        "flags"           : flags,
        "recommendations" : recommendations,
        "langue_detectee" : langue,
        "confiance"       : confiance,
    }
```

---

## 5. Pipeline d'agrégation complet

### 5.1 La fonction principale

```python
def pipeline_agregation_complet(
    resultats_trocr: list[dict],
    resultats_kraken: list[dict],
    transcriptions_humaines: Optional[dict[str, str]] = None,
    calibreurs: Optional[dict[str, CalibreurConfiance]] = None,
    images_lignes: Optional[dict[str, np.ndarray]] = None
) -> list[dict]:
    """
    Agrège les transcriptions de TrOCR et Kraken en un résultat consolidé.

    Args:
        resultats_trocr      : [{id, transcription, confiance, ...}, ...]
        resultats_kraken     : [{id, transcription, confiance, ...}, ...]
        transcriptions_humaines : {id_ligne: transcription} — prioritaires
        calibreurs           : calibreurs de confiance par modèle
        images_lignes        : {id_ligne: image_numpy} — pour le diagnostic

    Returns:
        Liste de dicts consolidés, triés par ordre de lecture
    """
    # Indexer par id de ligne
    trocr_par_id  = {r["id"]: r for r in resultats_trocr}
    kraken_par_id = {r["id"]: r for r in resultats_kraken}
    humain_par_id = transcriptions_humaines or {}
    images_par_id = images_lignes or {}

    # Union de tous les ids de lignes
    tous_ids = sorted(
        set(trocr_par_id.keys()) | set(kraken_par_id.keys()),
        key=lambda x: int(x[1:]) if x[1:].isdigit() else 0
    )

    resultats_consolides = []

    for ligne_id in tous_ids:
        trocr  = trocr_par_id.get(ligne_id)
        kraken = kraken_par_id.get(ligne_id)
        humain = humain_par_id.get(ligne_id)
        image  = images_par_id.get(ligne_id)

        # ── Constituer le dict des sources disponibles ───────────────────
        sources = {}
        if trocr:
            conf_t = trocr["confiance"]
            if calibreurs and "TrOCR" in calibreurs:
                conf_t = calibreurs["TrOCR"].calibrer([conf_t])[0]
            sources["TrOCR"] = (trocr["transcription"], conf_t)

        if kraken:
            conf_k = kraken["confiance"]
            if calibreurs and "Kraken" in calibreurs:
                conf_k = calibreurs["Kraken"].calibrer([conf_k])[0]
            sources["Kraken"] = (kraken["transcription"], conf_k)

        if not sources:
            continue   # Aucune source disponible

        # ── Agrégation ────────────────────────────────────────────────────
        transcription_finale, confiance_finale, details = agreger_avec_humain(
            sources,
            transcription_humaine=humain,
            poids_humain=0.95
        )

        # ── Détection des cas limites ─────────────────────────────────────
        t_b = kraken["transcription"] if kraken else None
        c_b = sources.get("Kraken", (None, 0.0))[1]

        cas_limite = traiter_cas_limite(
            image_ligne_np=image,
            transcription=transcription_finale,
            confiance=confiance_finale,
            transcription_b=t_b,
            confiance_b=c_b,
        )

        # ── Résultat consolidé ────────────────────────────────────────────
        resultats_consolides.append({
            "line_id"          : ligne_id,
            "transcription"    : transcription_finale,
            "confidence"       : round(confiance_finale, 4),
            "needs_review"     : cas_limite["needs_review"],
            "flags"            : cas_limite["flags"],
            "langue_detectee"  : cas_limite["langue_detectee"],
            "sources"          : {
                nom: {"transcription": pred, "confiance": round(conf, 4)}
                for nom, (pred, conf) in sources.items()
            },
            "source_humaine"   : humain is not None,
            "details"          : details,
        })

    return resultats_consolides
```

### 5.2 Le CER estimé en production

En production, on ne dispose pas de la vérité terrain pour calculer le CER. Cependant, on peut l'*estimer* à partir des scores de confiance, si le calibreur a bien été ajusté.

```python
def estimer_cer_production(
    resultats_consolides: list[dict],
    calibreurs: Optional[dict] = None
) -> dict:
    """
    Estime le CER global en production sans vérité terrain,
    à partir des scores de confiance calibrés.

    Hypothèse : P(ligne correcte) ≈ confiance calibrée.
    CER estimé ≈ 1 - moyenne des confidences.

    Note : cette estimation est optimiste (sous-estime le CER)
    si le calibreur n'est pas parfait. À utiliser comme borne inférieure.
    """
    confidences = [r["confidence"] for r in resultats_consolides]
    n_needs_review = sum(1 for r in resultats_consolides if r["needs_review"])
    n_total = len(resultats_consolides)

    cer_estime = 1.0 - np.mean(confidences) if confidences else 1.0
    taux_review = n_needs_review / n_total if n_total > 0 else 0.0

    stats = {
        "n_lignes"        : n_total,
        "cer_estime"      : round(cer_estime, 4),
        "confiance_moy"   : round(float(np.mean(confidences)), 4),
        "confiance_med"   : round(float(np.median(confidences)), 4),
        "confiance_p10"   : round(float(np.percentile(confidences, 10)), 4),
        "taux_needs_review": round(taux_review, 4),
        "n_needs_review"  : n_needs_review,
        "flags_frequents" : Counter(
            f for r in resultats_consolides for f in r["flags"]
        ).most_common(10),
    }

    print("\n=== Statistiques d'agrégation ===")
    print(f"  Lignes traitées   : {stats['n_lignes']}")
    print(f"  CER estimé        : {stats['cer_estime']:.1%}")
    print(f"  Confiance moyenne : {stats['confiance_moy']:.3f}")
    print(f"  Lignes à réviser  : {stats['n_needs_review']} ({stats['taux_needs_review']:.1%})")
    if stats["flags_frequents"]:
        print("  Flags les plus fréquents :")
        for flag, n in stats["flags_frequents"]:
            print(f"    {flag:30s} : {n}")

    return stats
```

---

## 6. Export vers le data contract du module NLP

### 6.1 Le format JSON final

Le résultat de l'agrégation est exporté dans le format JSON défini dans le data contract. Ce format est le *seul* format de sortie du pipeline CV — il sera lu par le module NLP pour la correction, la normalisation et l'annotation linguistique.

```python
import json
from pathlib import Path
from datetime import datetime

def exporter_dataset_nlp(
    resultats_par_page: dict[str, list[dict]],
    dossier_sortie: str = "dataset_nlp",
    version: str = "1.0",
    pipeline_utilise: str = "SAM+TrOCR+Kraken_agrege"
) -> None:
    """
    Exporte l'ensemble du dataset au format JSON du data contract.
    Crée un fichier JSON par page + un fichier de métadonnées global.

    Structure du data contract (définie en section 4.4) :
    {
        "page_id"        : str,
        "pipeline"       : str,
        "n_lignes"       : int,
        "cer_estime"     : float,
        "taux_review"    : float,
        "lignes" : [
            {
                "line_id"       : str,
                "transcription" : str,
                "confidence"    : float,
                "needs_review"  : bool,
                "flags"         : [str],
                "langue"        : str,
                "sources"       : {modele: {transcription, confiance}},
            },
            ...
        ]
    }
    """
    Path(dossier_sortie).mkdir(exist_ok=True)
    metadonnees_globales = {
        "version"         : version,
        "pipeline"        : pipeline_utilise,
        "date_creation"   : datetime.now().isoformat(),
        "n_pages"         : len(resultats_par_page),
        "pages"           : [],
    }

    for page_id, resultats in resultats_par_page.items():
        stats_page = estimer_cer_production(resultats)

        document_page = {
            "page_id"   : page_id,
            "pipeline"  : pipeline_utilise,
            "n_lignes"  : len(resultats),
            "cer_estime": stats_page["cer_estime"],
            "taux_review": stats_page["taux_needs_review"],
            "lignes"    : [
                {
                    "line_id"      : r["line_id"],
                    "transcription": r["transcription"],
                    "confidence"   : r["confidence"],
                    "needs_review" : r["needs_review"],
                    "flags"        : r["flags"],
                    "langue"       : r["langue_detectee"],
                    "sources"      : r.get("sources", {}),
                    "source_humaine": r.get("source_humaine", False),
                }
                for r in resultats
            ]
        }

        # Sauvegarde page par page
        chemin_page = Path(dossier_sortie) / f"{page_id}.json"
        with open(chemin_page, "w", encoding="utf-8") as f:
            json.dump(document_page, f, ensure_ascii=False, indent=2)

        # Métadonnées globales
        metadonnees_globales["pages"].append({
            "page_id"    : page_id,
            "n_lignes"   : len(resultats),
            "cer_estime" : stats_page["cer_estime"],
            "taux_review": stats_page["taux_needs_review"],
        })

    # Fichier de métadonnées global
    chemin_meta = Path(dossier_sortie) / "_metadata.json"
    with open(chemin_meta, "w", encoding="utf-8") as f:
        json.dump(metadonnees_globales, f, ensure_ascii=False, indent=2)

    # Rapport final
    cer_moyen = np.mean([p["cer_estime"] for p in metadonnees_globales["pages"]])
    taux_review_moyen = np.mean([p["taux_review"] for p in metadonnees_globales["pages"]])
    print(f"\n=== Dataset NLP exporté dans {dossier_sortie}/ ===")
    print(f"  {len(resultats_par_page)} pages")
    print(f"  CER estimé moyen  : {cer_moyen:.1%}")
    print(f"  Taux needs_review : {taux_review_moyen:.1%}")


def exporter_huggingface_dataset(
    dossier_json: str,
    dossier_sortie: str = "dataset_hf"
) -> None:
    """
    Convertit les fichiers JSON en dataset HuggingFace (format Arrow).
    Facilite le chargement par le module NLP.
    """
    from datasets import Dataset, DatasetDict

    tous_exemples = []
    for json_file in sorted(Path(dossier_json).glob("*.json")):
        if json_file.name.startswith("_"):
            continue
        with open(json_file, encoding="utf-8") as f:
            page = json.load(f)
        for ligne in page["lignes"]:
            tous_exemples.append({
                "page_id"      : page["page_id"],
                "line_id"      : ligne["line_id"],
                "text"         : ligne["transcription"],
                "confidence"   : ligne["confidence"],
                "needs_review" : ligne["needs_review"],
                "flags"        : " | ".join(ligne.get("flags", [])),
                "langue"       : ligne.get("langue", "inconnu"),
                "source_humaine": ligne.get("source_humaine", False),
            })

    dataset = Dataset.from_list(tous_exemples)

    # Split selon needs_review : les lignes fiables en train, les autres en review
    dataset_dict = DatasetDict({
        "all"    : dataset,
        "fiable" : dataset.filter(lambda x: not x["needs_review"]),
        "review" : dataset.filter(lambda x: x["needs_review"]),
    })

    dataset_dict.save_to_disk(dossier_sortie)
    print(f"\n  Dataset HuggingFace exporté dans {dossier_sortie}/")
    print(f"  Total   : {len(dataset)} lignes")
    print(f"  Fiable  : {len(dataset_dict['fiable'])} lignes")
    print(f"  Review  : {len(dataset_dict['review'])} lignes")
```

---

## 7. Connexions avec le pipeline global

La section 4.3 est le **point de convergence** des deux branches du pipeline :

```
Pipeline modulaire (SAM + TrOCR)
    resultats_trocr ──────────────┐
                                  ↓
                        [4.3] Agrégation ──→ Dataset consolidé
                                  ↑           (JSON + HuggingFace)
    resultats_kraken ─────────────┘               ↓
Pipeline Kraken intégré                    [Module NLP]

                    +
    transcriptions_humaines (section 1.3)
    calibreurs (section 3.5 validation)
```

**Ce que l'agrégation garantit au module NLP :**
- Chaque ligne a une transcription unique, consolidée depuis les meilleures sources disponibles.
- Chaque ligne a un score de confiance calibré — le NLP peut prioriser les corrections sur les lignes `needs_review`.
- Les cas limites sont explicitement signalés — le NLP ne traitera pas un symbole alchimique comme du vieux français.
- Les langues détectées orientent le choix du modèle de langue pour la normalisation.

---

## Glossaire des termes avancés

**Active learning**
Voir glossaire section 4.2. Dans ce contexte : sélection des lignes à faire réviser par l'expert humain en priorité, selon leur flag `needs_review` et leur score de confiance.

**Algorithme de Needleman-Wunsch**
Algorithme de programmation dynamique calculant l'alignement global optimal de deux séquences. Développé en 1970 pour la bioinformatique (alignement de séquences d'ADN), directement applicable aux transcriptions textuelles. Complexité $O(mn)$ en temps et en espace.

**Alignement multiple de séquences**
Extension de l'alignement par paires à $N \geq 3$ séquences. Produit $N$ séquences de même longueur avec des gaps `-` aux positions où certaines séquences ont des insertions. Base du vote position par position.

**Calibration de confiance**
Transformation des scores de confiance bruts d'un modèle pour qu'ils correspondent à des probabilités réelles. Un modèle bien calibré a $P(\text{correct} | \text{score} = 0.8) \approx 0.8$. Permet de comparer des scores de modèles différents.

**Coefficient phi (φ)**
Mesure de corrélation entre deux variables binaires, analogue au coefficient de Pearson. $\phi \approx 0$ : erreurs indépendantes (ensemble utile) ; $\phi \approx 1$ : erreurs corrélées (ensemble peu utile).

**CER estimé en production**
Estimation du Character Error Rate sans vérité terrain, calculée à partir des scores de confiance calibrés. Fournit une borne inférieure sur la qualité réelle de la transcription.

**Corrélation des erreurs**
Mesure de la dépendance entre les patterns d'erreurs de deux modèles. Une faible corrélation (phi ≈ 0) indique que les modèles se trompent sur des lignes différentes — condition nécessaire pour que leur agrégation améliore les performances.

**Data contract**
Spécification formelle du format et du contenu du dataset livré par le pipeline CV au module NLP. Documente les champs obligatoires, leurs types, leurs plages de valeurs, et les conventions de codage (ex. : signification de `needs_review`, codes de langue).

**Flag `needs_review`**
Indicateur booléen dans le JSON de sortie signalant qu'une ligne nécessite une vérification humaine. Déclenché par : confiance < seuil, discordance inter-modèles > seuil, ligne trop courte, image vide, langue non identifiée.

**Gap (`-`)**
Caractère spécial inséré dans les séquences alignées pour marquer une insertion ou une suppression. Permet de mettre en correspondance des séquences de longueurs différentes avant le vote caractère par caractère.

**Méthode isotonique (*isotonic regression*)**
Méthode de calibration non-paramétrique qui ajuste une fonction croissante et non-décroissante pour prédire $P(\text{correct} | \text{score})$. Flexible et ne suppose aucune forme paramétrique de la relation.

**Régression de Platt (*Platt scaling*)**
Méthode de calibration par régression logistique : $P(\text{correct}) = \sigma(a \cdot \text{score} + b)$. Simple et rapide, mais suppose une relation sigmoïdale entre score et probabilité.

**Vote pondéré**
Méthode d'ensemble où chaque source vote pour un choix, avec un poids proportionnel à sa fiabilité estimée. La décision est la somme pondérée des votes. Généralise le vote majoritaire (tous les poids égaux).

---

## Bibliographie de référence

### Méthodes d'ensemble et agrégation

- **Dietterich, T. G.** (2000). *Ensemble Methods in Machine Learning*. LNCS 1857, 1–15. Springer. — Introduction aux méthodes d'ensemble, avec analyse théorique des conditions de leur efficacité.

- **Polikar, R.** (2006). *Ensemble Based Systems in Decision Making*. IEEE Circuits and Systems Magazine, 6(3), 21–45. — Revue complète des méthodes d'ensemble pour la classification et la reconnaissance.

### Alignement de séquences

- **Needleman, S. B., Wunsch, C. D.** (1970). *A general method applicable to the search for similarities in the amino acid sequence of two proteins*. Journal of Molecular Biology, 48(3), 443–453. — L'algorithme d'alignement global. Article fondateur, applicable aux transcriptions.

- **Wagner, R. A., Fischer, M. J.** (1974). *The String-to-String Correction Problem*. Journal of the ACM, 21(1), 168–173. — Généralisation de Levenshtein avec coûts variables. Utile pour pondérer différemment les substitutions de caractères visuellement proches.

### Calibration de confiance

- **Guo, C., Pleiss, G., Sun, Y., Weinberger, K. Q.** (2017). *On Calibration of Modern Neural Networks*. ICML 2017. [arXiv:1706.04599] — Analyse de la sur-confiance des réseaux profonds et méthodes de calibration (temperature scaling, Platt scaling).

- **Zadrozny, B., Elkan, C.** (2001). *Obtaining Calibrated Probability Estimates from Decision Trees and Naive Bayesian Classifiers*. ICML 2001. — Introduction à la régression isotonique pour la calibration.

### Post-traitement HTR et vote

- **Strauß, T., Leifert, G., Labahn, R.** (2018). *ISPR 2016 Transcription Evaluation: Using Character Error Rate as Metric for Post-Correction*. — Sur l'utilisation des scores de confiance pour le post-traitement et la correction.

- **Reul, C., Springmann, U., Wick, C., Puppe, F.** (2018). *Improving OCR Accuracy on Early Printed Books by Combining Outputs of Two OCR Engines*. JLCL, 33(1). — Agrégation de deux moteurs OCR par vote, directement applicable à notre contexte TrOCR + Kraken.

### Détection de langue

- **Nakatani, S.** (2010). *langdetect: Language Detection Library for Java (et Python)*. — La bibliothèque `langdetect` utilisée dans le code. Note : peu performante sur les langues médiévales — à utiliser comme indicateur, pas comme oracle.

### Formats de données et interopérabilité

- **HuggingFace Datasets** (2021). *Datasets: A Community Library for Natural Language Processing*. EMNLP 2021. — La bibliothèque `datasets` pour la gestion et le partage de datasets ML.

---

*Support de cours rédigé pour le module Computer Vision · Promotion MD5 · Mai 2026*
*Ce document accompagne la séance 4.3 du Jour 4.*
*Il suppose la lecture des sections 4.1 (prétraitement) et 4.2 (pipeline Kraken).*
*Durée estimée de lecture : 60–75 minutes.*
