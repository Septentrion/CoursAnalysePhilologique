# Cours 4.4 — Architecture du pipeline et data contract

**Module 4 · Computer Vision appliquée aux manuscrits médiévaux · MD5**

---

> *« Un pipeline de production ne se termine pas quand les modèles fonctionnent. Il se termine quand les données produites sont assez bien documentées pour que quelqu'un d'autre — le module NLP, un chercheur dans trois ans, vous-même dans six mois — puisse les utiliser sans vous demander ce qu'elles signifient. »*

---

## Introduction : de l'assemblage à la livraison

Les sections 4.1 à 4.3 ont construit les composants du pipeline et leur assemblage. La section 4.3 a produit un dataset consolidé — mais un dataset n'est pas encore une livraison. Une livraison, c'est un dataset **documenté**, **validé**, **versionné**, accompagné d'une spécification formelle que le destinataire peut lire indépendamment du code qui l'a produit.

C'est l'objet de cette section : formaliser le **data contract** qui lie le pipeline CV au module NLP, et documenter les choix éditoriaux qui ont guidé toutes les décisions de transcription tout au long du cours.

Cette section est plus courte que les précédentes — trente minutes en cours magistral — parce qu'elle synthétise plutôt qu'elle n'introduit. Mais elle est loin d'être anodine : les erreurs de livraison (format mal documenté, champ manquant, convention non spécifiée) sont parmi les causes les plus fréquentes d'échec dans les projets interdisciplinaires entre équipes d'ingénieurs et équipes de recherche en humanités.

---

## 1. Vue d'ensemble : les deux architectures du pipeline

### 1.1 Rappel de l'architecture complète

Le Jour 4 a assemblé deux pipelines complémentaires. Le diagramme ci-dessous les récapitule côte à côte, depuis le scan brut jusqu'au dataset NLP.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        ENTRÉE COMMUNE                                   │
│                 Scan TIFF/JPEG (300–400 DPI)                            │
└──────────────────────────────┬──────────────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────────────┐
│  SECTION 4.1 — PRÉTRAITEMENT (commun aux deux pipelines)                 │
│  OpenCV + scikit-image :                                                 │
│  • Deskewing (correction d'inclinaison)                                  │
│  • CLAHE (amélioration du contraste sur canal L)                         │
│  • Sauvola (binarisation adaptative)                                     │
│  • Filtre médian + nettoyage morphologique                               │
│  • Détection des zones saturées (or, enluminures)                        │
└────────────────┬────────────────────────────────────────┬────────────────┘
                 ↓                                        ↓
  ┌──────────────────────────┐            ┌───────────────────────────────┐
  │  PIPELINE MODULAIRE       │            │  PIPELINE KRAKEN INTÉGRÉ      │
  │  (section 4.2 + Jour 3)  │            │  (section 4.2)                │
  │                           │            │                               │
  │  SAM (ViT-H)              │            │  Kraken BLLA                  │
  │  → Régions layout         │            │  → Baselines + polygones      │
  │    (boîtes + classes)     │            │  → Ordre de lecture           │
  │        ↓                  │            │  → Régions typées             │
  │  Kraken BLLA              │            │        ↓                      │
  │  → Lignes (baseline)      │            │  Kraken HTR (LSTM-CTC)        │
  │        ↓                  │            │  → Transcriptions brutes      │
  │  TrOCR fine-tuné          │            │  → Confiances par ligne       │
  │  → Transcriptions         │            │        ↓                      │
  │  → Scores softmax         │            │  PAGE XML / ALTO              │
  │        ↓                  │            │  (eScriptorium / Gallica)     │
  │  CLIP / LLaVA             │            │        ↓                      │
  │  → Descriptions images    │            │  [Correction humaine opt.]    │
  └──────────────┬────────────┘            └──────────────┬────────────────┘
                 ↓                                        ↓
┌────────────────────────────────────────────────────────────────────────────┐
│  SECTION 4.3 — AGRÉGATION                                                  │
│  • Alignement Needleman-Wunsch + vote pondéré (TrOCR × Kraken)            │
│  • Intégration des transcriptions humaines (poids prioritaire)             │
│  • Calibration des scores de confiance                                     │
│  • Détection et flagging des cas limites                                   │
│  • Estimation du CER en production                                         │
└────────────────────────────────────────────────────────────────────────────┘
                               ↓
┌────────────────────────────────────────────────────────────────────────────┐
│  SECTION 4.4 — DATA CONTRACT ET EXPORT (cette section)                     │
│  • JSON validé par JSON Schema                                             │
│  • HuggingFace DatasetDict versionné                                       │
│  • Documentation des choix éditoriaux                                      │
└──────────────────────────────┬─────────────────────────────────────────────┘
                               ↓
┌────────────────────────────────────────────────────────────────────────────┐
│                        MODULE NLP (cours suivant)                          │
│  Normalisation orthographique · Développement des abréviations             │
│  Annotation linguistique · Correction par modèle de langue                 │
└────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Quand utiliser quel pipeline

Ce tableau synthétise la décision d'architecture pour chaque situation rencontrée dans le projet (complémentaire du tableau de la section 4.2) :

| Situation | Pipeline recommandé | Raison |
|-----------|--------------------|---------| 
| Page simple (1 colonne, peu dégradée) | Kraken seul | Plus rapide, PAGE XML natif |
| Page à 2 colonnes + marginalia | Kraken seul | Ordre de lecture intégré |
| Page très dégradée (CER Kraken > 20%) | TrOCR + vote | Meilleure modélisation linguistique |
| Page avec enluminures à décrire | Pipeline modulaire + CLIP | CLIP pour les descriptions |
| Corpus homogène (1 main, 1 siècle) | Kraken seul | Fine-tuning ciblé suffisant |
| Corpus très diversifié | Les deux + vote | Complémentarité maximale |
| Contrainte CPU (pas de GPU) | Kraken seul | TrOCR impraticable sans GPU |
| Livraison eScriptorium | Kraken seul | PAGE XML natif |
| Livraison module NLP | Les deux + vote | Meilleur CER final |

---

## 2. Le data contract : spécification formelle

### 2.1 Qu'est-ce qu'un data contract ?

Un **data contract** est un accord formel entre producteur et consommateur de données sur :

- La **structure** des données (quels champs, quels types, quelles relations).
- La **sémantique** de chaque champ (que signifie exactement `confidence = 0.73` ?).
- Les **garanties** offertes (toutes les pages ont un `page_id`, toutes les lignes ont un `transcription` non vide).
- Les **conventions** editoriales (comment les abréviations sont-elles encodées ? les lacunes ?).
- La **version** du contrat (pour gérer les évolutions futures).

Sans data contract explicite, chaque consommateur de données doit deviner ce que les champs signifient — source inépuisable de bugs et de malentendus dans les projets interdisciplinaires.

### 2.2 Le schéma JSON

Le data contract est formalisé en **JSON Schema** — un format standard de description et de validation de documents JSON.

```python
DATA_CONTRACT_SCHEMA = {
    "$schema"    : "https://json-schema.org/draft/2020-12/schema",
    "$id"        : "https://projet-manuscrits-md5/data-contract/v1.0",
    "title"      : "Dataset de transcription HTR — Pipeline CV MD5",
    "description": (
        "Format de livraison du pipeline Computer Vision au module NLP. "
        "Chaque fichier JSON correspond à une page de manuscrit transcrite."
    ),
    "type": "object",

    # ── Champs obligatoires ────────────────────────────────────────────────
    "required": [
        "page_id", "schema_version", "pipeline",
        "date_creation", "n_lignes", "lignes"
    ],

    "properties": {

        "page_id": {
            "type"       : "string",
            "description": "Identifiant unique de la page, dérivé du nom du fichier image.",
            "examples"   : ["bnf_fr768_f001r", "gallica_btv1b_f045"]
        },

        "schema_version": {
            "type"       : "string",
            "pattern"    : r"^\d+\.\d+$",
            "description": "Version du data contract (SemVer majeur.mineur).",
            "examples"   : ["1.0", "1.1"]
        },

        "pipeline": {
            "type"       : "string",
            "enum"       : [
                "modulaire_SAM_TrOCR",
                "kraken_integre",
                "hybride_vote",
                "humain_seul"
            ],
            "description": "Pipeline utilisé pour produire cette page."
        },

        "date_creation": {
            "type"       : "string",
            "format"     : "date-time",
            "description": "Horodatage ISO 8601 de la création du fichier."
        },

        "n_lignes": {
            "type"       : "integer",
            "minimum"    : 0,
            "description": "Nombre de lignes dans le tableau 'lignes'."
        },

        # ── Métadonnées optionnelles de la page ───────────────────────────
        "cer_estime": {
            "type"       : ["number", "null"],
            "minimum"    : 0.0,
            "maximum"    : 1.0,
            "description": (
                "CER estimé en production (sans vérité terrain), "
                "calculé depuis les scores de confiance calibrés. "
                "Null si non calculé."
            )
        },

        "taux_needs_review": {
            "type"       : ["number", "null"],
            "minimum"    : 0.0,
            "maximum"    : 1.0,
            "description": "Fraction des lignes flaggées needs_review."
        },

        "conventions_transcription": {
            "type"       : "string",
            "description": "Référence au fichier de conventions éditoriales utilisé.",
            "examples"   : ["conventions_v1.0.md", "CREMMA_guide_transcription"]
        },

        # ── Tableau des lignes (obligatoire, au moins vide) ───────────────
        "lignes": {
            "type"       : "array",
            "description": "Transcriptions ligne par ligne, dans l'ordre de lecture.",
            "items"      : {
                "type"    : "object",
                "required": [
                    "line_id", "transcription",
                    "confidence", "needs_review"
                ],
                "properties": {

                    "line_id": {
                        "type"       : "string",
                        "pattern"    : r"^l\d{4}$",
                        "description": "Identifiant de ligne unique dans la page (l0001, l0002…)"
                    },

                    "transcription": {
                        "type"       : "string",
                        "description": (
                            "Transcription semi-diplomatique de la ligne. "
                            "Abréviations développées selon les conventions. "
                            "Lacunes encodées [...]."
                        )
                    },

                    "confidence": {
                        "type"       : "number",
                        "minimum"    : 0.0,
                        "maximum"    : 1.0,
                        "description": (
                            "Score de confiance calibré [0, 1]. "
                            "Interprétation approximative : "
                            "P(transcription_correcte | score)."
                        )
                    },

                    "needs_review": {
                        "type"       : "boolean",
                        "description": (
                            "True si la ligne nécessite une vérification humaine. "
                            "Déclenché par : confidence < 0.5, "
                            "discordance inter-modèles > 50%, "
                            "ligne trop courte, image vide, "
                            "langue non identifiée."
                        )
                    },

                    # ── Champs optionnels de ligne ────────────────────────
                    "flags": {
                        "type"       : "array",
                        "items"      : {"type": "string"},
                        "description": (
                            "Liste des drapeaux actifs. Valeurs possibles : "
                            "LIGNE_COURTE, CONFIANCE_BASSE, DISCORDANCE_MODELES, "
                            "IMAGE_VIDE, LANGUE_DETECTEE:<code>."
                        )
                    },

                    "langue": {
                        "type"       : "string",
                        "description": (
                            "Code de langue détectée (ISO 639-1). "
                            "Valeurs attendues : 'fr' (français), 'la' (latin), "
                            "'inconnu' si non déterminable."
                        ),
                        "examples"   : ["fr", "la", "inconnu"]
                    },

                    "sources": {
                        "type"       : "object",
                        "description": "Transcriptions brutes par modèle source.",
                        "additionalProperties": {
                            "type"    : "object",
                            "required": ["transcription", "confiance"],
                            "properties": {
                                "transcription": {"type": "string"},
                                "confiance"    : {
                                    "type": "number",
                                    "minimum": 0.0,
                                    "maximum": 1.0
                                }
                            }
                        }
                    },

                    "source_humaine": {
                        "type"       : "boolean",
                        "description": "True si la transcription a été validée ou saisie par un humain."
                    },

                    "baseline": {
                        "type"       : ["array", "null"],
                        "description": "Coordonnées de la baseline [(x1,y1),(x2,y2),...] en pixels.",
                        "items"      : {
                            "type"    : "array",
                            "items"   : {"type": "integer"},
                            "minItems": 2,
                            "maxItems": 2
                        }
                    }
                },

                "additionalProperties": False
            }
        }
    },

    "additionalProperties": False
}
```

### 2.3 Validation par JSON Schema

```python
import json
import jsonschema
from pathlib import Path

def valider_fichier_json(
    chemin_json: str,
    schema: dict = DATA_CONTRACT_SCHEMA,
    verbose: bool = True
) -> tuple[bool, list[str]]:
    """
    Valide un fichier JSON contre le data contract.

    Returns:
        (valide, liste_erreurs)
    """
    with open(chemin_json, encoding="utf-8") as f:
        document = json.load(f)

    validator = jsonschema.Draft202012Validator(schema)
    erreurs = list(validator.iter_errors(document))

    if verbose:
        if not erreurs:
            print(f"  ✓ {Path(chemin_json).name} — valide")
        else:
            print(f"  ✗ {Path(chemin_json).name} — {len(erreurs)} erreur(s) :")
            for err in erreurs[:5]:
                path = " → ".join(str(p) for p in err.absolute_path)
                print(f"    [{path}] {err.message}")

    return len(erreurs) == 0, [e.message for e in erreurs]


def valider_dataset_complet(
    dossier: str,
    schema: dict = DATA_CONTRACT_SCHEMA
) -> dict:
    """
    Valide tous les fichiers JSON d'un dossier et produit un rapport.
    """
    fichiers = sorted(Path(dossier).glob("*.json"))
    fichiers = [f for f in fichiers if not f.name.startswith("_")]

    resultats = {"valides": 0, "invalides": 0, "erreurs_par_fichier": {}}

    for f in fichiers:
        valide, erreurs = valider_fichier_json(str(f), schema, verbose=True)
        if valide:
            resultats["valides"] += 1
        else:
            resultats["invalides"] += 1
            resultats["erreurs_par_fichier"][f.name] = erreurs

    print(f"\n  Bilan validation : {resultats['valides']} valides, "
          f"{resultats['invalides']} invalides sur {len(fichiers)} fichiers")
    return resultats
```

### 2.4 Vérifications sémantiques complémentaires

Le JSON Schema valide la structure et les types, mais pas la cohérence sémantique. Des vérifications supplémentaires sont nécessaires.

```python
def verifier_coherence_semantique(
    document: dict
) -> list[str]:
    """
    Vérifie la cohérence interne d'un document JSON au-delà du schéma.
    Retourne la liste des incohérences détectées.
    """
    avertissements = []

    # Vérifier que n_lignes correspond à la longueur du tableau lignes
    if document["n_lignes"] != len(document["lignes"]):
        avertissements.append(
            f"n_lignes={document['n_lignes']} mais len(lignes)={len(document['lignes'])}"
        )

    # Vérifier l'unicité des line_id
    ids = [l["line_id"] for l in document["lignes"]]
    if len(ids) != len(set(ids)):
        from collections import Counter
        doublons = [k for k, v in Counter(ids).items() if v > 1]
        avertissements.append(f"line_id en double : {doublons}")

    # Vérifier que needs_review est cohérent avec les flags
    for ligne in document["lignes"]:
        if ligne.get("flags") and not ligne["needs_review"]:
            avertissements.append(
                f"Ligne {ligne['line_id']} : flags présents mais needs_review=False"
            )
        if ligne["confidence"] < 0.5 and not ligne["needs_review"]:
            avertissements.append(
                f"Ligne {ligne['line_id']} : confidence={ligne['confidence']:.2f} "
                f"< 0.5 mais needs_review=False"
            )

    # Vérifier que les transcriptions ne sont pas vides pour les lignes non-flaggées
    for ligne in document["lignes"]:
        if not ligne["needs_review"] and not ligne["transcription"].strip():
            avertissements.append(
                f"Ligne {ligne['line_id']} : transcription vide sans needs_review"
            )

    return avertissements
```

---

## 3. Versioning du dataset

### 3.1 Pourquoi versionner

Le dataset de transcription évoluera au fil du projet :

- **Itération 1** : transcriptions brutes du pipeline initial (CER ~20%).
- **Itération 2** : après correction dans eScriptorium sur 500 pages + réentraînement (CER ~12%).
- **Itération 3** : après correction complète et validation académique (CER ~5%).

Chaque itération est une version différente du dataset. Sans versioning, il est impossible de savoir quelle version du dataset a produit quels résultats NLP — ce qui rend la reproductibilité impossible et la comparaison entre expériences sans signification.

### 3.2 Versioning avec HuggingFace Datasets

```python
from datasets import Dataset, DatasetDict, load_from_disk
import json
from pathlib import Path
from datetime import datetime

def creer_dataset_versionne(
    dossier_json: str,
    version: str,
    description: str,
    chemin_sortie: str
) -> DatasetDict:
    """
    Crée un DatasetDict HuggingFace versionné depuis les fichiers JSON.

    La version est encodée dans les métadonnées du dataset.
    Les splits sont définis selon le statut des lignes :
    - 'train'      : lignes fiables, source humaine ou confiance élevée
    - 'validation' : lignes fiables, confiance intermédiaire
    - 'needs_review': lignes à réviser (ne pas utiliser en entraînement NLP)
    """
    tous_exemples = []

    for json_file in sorted(Path(dossier_json).glob("*.json")):
        if json_file.name.startswith("_"):
            continue
        with open(json_file, encoding="utf-8") as f:
            page = json.load(f)

        for ligne in page["lignes"]:
            tous_exemples.append({
                # Champs principaux
                "page_id"       : page["page_id"],
                "line_id"       : ligne["line_id"],
                "text"          : ligne["transcription"],
                "confidence"    : ligne["confidence"],
                "needs_review"  : ligne["needs_review"],
                # Métadonnées
                "langue"        : ligne.get("langue", "inconnu"),
                "flags"         : " | ".join(ligne.get("flags", [])),
                "source_humaine": ligne.get("source_humaine", False),
                "pipeline"      : page["pipeline"],
                # Version du dataset
                "dataset_version": version,
            })

    dataset_complet = Dataset.from_list(tous_exemples)

    # Définir les splits
    fiables = dataset_complet.filter(
        lambda x: not x["needs_review"] and x["confidence"] >= 0.7
    )
    validation = dataset_complet.filter(
        lambda x: not x["needs_review"] and 0.5 <= x["confidence"] < 0.7
    )
    review = dataset_complet.filter(lambda x: x["needs_review"])

    # Séparer fiables en train (80%) et test (20%)
    split_train_test = fiables.train_test_split(test_size=0.2, seed=42)

    dataset_dict = DatasetDict({
        "train"       : split_train_test["train"],
        "test"        : split_train_test["test"],
        "validation"  : validation,
        "needs_review": review,
    })

    # Ajouter les métadonnées de version
    dataset_dict.info = {
        "version"         : version,
        "description"     : description,
        "date_creation"   : datetime.now().isoformat(),
        "n_total"         : len(dataset_complet),
        "n_train"         : len(dataset_dict["train"]),
        "n_test"          : len(dataset_dict["test"]),
        "n_validation"    : len(dataset_dict["validation"]),
        "n_needs_review"  : len(dataset_dict["needs_review"]),
    }

    # Sauvegarde
    dataset_dict.save_to_disk(chemin_sortie)

    print(f"\n=== Dataset v{version} sauvegardé dans {chemin_sortie}/ ===")
    for nom, ds in dataset_dict.items():
        print(f"  {nom:15s} : {len(ds):5d} lignes")

    return dataset_dict


def charger_dataset_versionne(chemin: str) -> DatasetDict:
    """Charge un dataset versionné depuis le disque."""
    dataset_dict = load_from_disk(chemin)
    print(f"Dataset chargé depuis {chemin}")
    for nom, ds in dataset_dict.items():
        print(f"  {nom:15s} : {len(ds):5d} lignes")
    return dataset_dict
```

### 3.3 Changelog du dataset

Chaque nouvelle version du dataset doit être accompagnée d'un changelog — une description des modifications apportées.

```python
CHANGELOG = {
    "1.0": {
        "date"       : "2026-05-01",
        "description": "Version initiale — pipeline brut sans correction humaine",
        "n_pages"    : 45,
        "cer_estime" : 0.19,
        "notes"      : [
            "TrOCR-base-handwritten sans fine-tuning",
            "Kraken avec modèle base_medieval.mlmodel",
            "Aucune correction humaine"
        ]
    },
    "1.1": {
        "date"       : "2026-05-15",
        "description": "Fine-tuning TrOCR sur 1000 lignes CREMMA + correction eScriptorium",
        "n_pages"    : 45,
        "cer_estime" : 0.11,
        "notes"      : [
            "TrOCR fine-tuné LoRA r=16 sur CREMMA Médiéval",
            "200 pages corrigées dans eScriptorium",
            "Réentraînement Kraken iter1"
        ]
    },
    "2.0": {
        "date"       : "2026-06-01",
        "description": "Dataset de livraison finale — corpus complet corrigé",
        "n_pages"    : 120,
        "cer_estime" : 0.07,
        "notes"      : [
            "Corpus complet (120 pages)",
            "Toutes les pages validées par un expert paléographe",
            "Breaking change : champ 'baseline' ajouté aux lignes"
        ]
    }
}

def afficher_changelog() -> None:
    print("=== Changelog du dataset ===\n")
    for version, info in sorted(CHANGELOG.items()):
        print(f"v{version} ({info['date']}) — {info['description']}")
        print(f"  Pages : {info['n_pages']} | CER estimé : {info['cer_estime']:.1%}")
        for note in info["notes"]:
            print(f"  • {note}")
        print()
```

---

## 4. Documentation des choix éditoriaux

### 4.1 Pourquoi documenter les choix

Tout au long du cours, nous avons pris des décisions implicites sur la façon de transcrire les manuscrits. Ces décisions semblent évidentes aux personnes qui les ont prises — elles sont totalement opaques pour le module NLP qui recevra les données, et pour les chercheurs qui voudront réutiliser le corpus dans deux ans.

Exemples de décisions non documentées qui peuvent causer des problèmes en aval :

- Un abréviation de *dominus* est-elle développée en *dominus* ou en *dns* dans le dataset ?
- Une lacune de 3 caractères est-elle encodée `[...]` ou `[???]` ou `<gap/>` ?
- Les rubriques sont-elles incluses dans le flux du texte ou dans un champ séparé ?
- L'espace entre deux colonnes est-il encodé par un saut de ligne ou un séparateur spécial ?
- La casse est-elle préservée telle quelle, ou normalisée ?

Ces questions n'ont pas de bonne ou mauvaise réponse — ce qui compte est que **la réponse soit unique et documentée**.

### 4.2 Le fichier de conventions éditoriales

```markdown
# Conventions de transcription — Pipeline CV MD5
**Version : 1.0 | Date : Mai 2026 | Référence : DATA_CONTRACT_SCHEMA v1.0**

---

## 1. Niveau de fidélité

Le dataset adopte une **transcription semi-diplomatique** :

- L'orthographe du scribe est préservée telle quelle (pas de normalisation).
- Les abréviations sont développées selon les conventions ci-dessous.
- La casse est préservée telle qu'elle apparaît dans le manuscrit.
- La ponctuation du manuscrit est reproduite sans modification.

**Référence :** Guyotjeannin & Vielliard (2001), *Conseils pour l'édition des textes médiévaux*.

---

## 2. Abréviations

| Situation | Convention | Exemple |
|-----------|-----------|---------|
| Abréviation développée de façon certaine | Développement direct | `dns` → `dominus` |
| Développement incertain | Développement entre parenthèses | `(dominus)` |
| Signe non identifié | `[abr]` | `q[abr]` |
| Tilde au-dessus → nasale omise | Développée sans marquage | `ho~e` → `home` |

**Note pour le module NLP :** les parenthèses `(...)` dans les transcriptions indiquent
une incertitude éditoriale, pas une parenthèse dans le texte original.

---

## 3. Lacunes et zones illisibles

| Situation | Encodage |
|-----------|---------|
| Caractères illisibles (nombre connu) | `[?]` par caractère illisible |
| Zone illisible (nombre inconnu) | `[...]` |
| Lacune physique (parchemin détruit) | `[†]` |
| Mot proposé avec incertitude | `[mot?]` |

---

## 4. Éléments structurels

| Élément | Convention dans `transcription` |
|---------|--------------------------------|
| Rubrique (texte rouge) | Incluse dans le flux, précédée de `<R>` et suivie de `</R>` |
| Lettrine | Premier caractère de la première ligne de la section |
| Note marginale | Champ séparé `"note_marginale"` si présente, non intégré au flux |
| Fin de colonne | `||` (double barre verticale) dans la transcription |
| Fin de page | Non encodée (une page = un fichier JSON) |

---

## 5. Langues mélangées

Quand une ligne mélange le vieux français et le latin (ou une autre langue),
la totalité de la ligne est transcrite dans son état brut. Le champ `langue`
indique la langue dominante détectée. Le module NLP est responsable
de la désambiguïsation linguistique interne à la ligne.

---

## 6. Encodage des caractères spéciaux

| Caractère | Unicode | Signification |
|-----------|---------|---------------|
| ȝ (yogh) | U+021D | Lettre médiévale pour /j/ ou /ʒ/ |
| ꝑ (p barré) | U+A751 | Abréviation de *per-* ou *par-* |
| ⁊ (tironien et) | U+204A | Abréviation de *et* |
| æ | U+00E6 | Ligature ae |
| œ | U+0153 | Ligature oe |

**Note :** tous les caractères sont encodés en UTF-8.
Les caractères médiévaux sans équivalent Unicode sont translittérés
selon les conventions MUFI (*Medieval Unicode Font Initiative*).

---

## 7. Ce que ce dataset N'encode PAS

Les éléments suivants ne sont **pas** encodés dans le champ `transcription`
et ne doivent pas être attendus par le module NLP :

- La position spatiale des mots dans la ligne (voir champ `baseline`).
- Les ratures et corrections du scribe (transcription de l'état final).
- Les annotations marginales postérieures (champ séparé).
- Les illustrations et enluminures (champ séparé `illustrations`).
- L'accentuation (non systématique dans les manuscrits médiévaux).
```

```python
def generer_fichier_conventions(
    chemin_sortie: str = "dataset_nlp/CONVENTIONS_TRANSCRIPTION.md"
) -> None:
    """Génère le fichier de conventions éditoriales."""
    contenu = """# Conventions de transcription — Pipeline CV MD5
[... voir le contenu ci-dessus ...]
"""
    with open(chemin_sortie, "w", encoding="utf-8") as f:
        f.write(contenu)
    print(f"  Conventions exportées : {chemin_sortie}")
```

---

## 5. Monitoring en production

### 5.1 Métriques à surveiller

Un pipeline de production doit être surveillé pour détecter les dérives de performance au fil du temps — par exemple, si le corpus s'étend à des siècles ou des régions non représentés dans les données d'entraînement.

```python
from dataclasses import dataclass, field
from datetime import datetime
import json

@dataclass
class MetriquesProduction:
    """Métriques de suivi du pipeline en production."""
    timestamp          : str     = field(default_factory=lambda: datetime.now().isoformat())
    n_pages_traitees   : int     = 0
    n_lignes_totales   : int     = 0
    cer_estime_moyen   : float   = 0.0
    taux_needs_review  : float   = 0.0
    confiance_mediane  : float   = 0.0
    # Distribution des flags
    freq_confiance_basse: float  = 0.0
    freq_discordance   : float   = 0.0
    freq_ligne_courte  : float   = 0.0
    freq_image_vide    : float   = 0.0
    # Performance par modèle
    cer_estime_trocr   : float   = 0.0
    cer_estime_kraken  : float   = 0.0


def calculer_metriques_production(
    dossier_json: str
) -> MetriquesProduction:
    """Calcule les métriques de production depuis un dossier de JSON."""
    metriques = MetriquesProduction()
    tous_cer, toutes_conf = [], []
    tous_flags = []

    for json_file in Path(dossier_json).glob("*.json"):
        if json_file.name.startswith("_"):
            continue
        with open(json_file, encoding="utf-8") as f:
            page = json.load(f)

        metriques.n_pages_traitees += 1
        if page.get("cer_estime") is not None:
            tous_cer.append(page["cer_estime"])

        for ligne in page.get("lignes", []):
            metriques.n_lignes_totales += 1
            toutes_conf.append(ligne["confidence"])
            tous_flags.extend(ligne.get("flags", []))

    if tous_cer:
        metriques.cer_estime_moyen = float(np.mean(tous_cer))
    if toutes_conf:
        metriques.confiance_mediane  = float(np.median(toutes_conf))
        metriques.taux_needs_review  = float(np.mean(
            [c < 0.5 for c in toutes_conf]
        ))

    # Fréquence des flags
    n = metriques.n_lignes_totales or 1
    from collections import Counter
    freq_flags = Counter(tous_flags)
    metriques.freq_confiance_basse = freq_flags.get("CONFIANCE_BASSE", 0) / n
    metriques.freq_discordance     = freq_flags.get("DISCORDANCE_MODELES", 0) / n
    metriques.freq_ligne_courte    = freq_flags.get("LIGNE_COURTE", 0) / n
    metriques.freq_image_vide      = freq_flags.get("IMAGE_VIDE", 0) / n

    return metriques


def detecter_derive(
    metriques_actuelles: MetriquesProduction,
    metriques_reference: MetriquesProduction,
    seuil_cer_absolu: float = 0.05,
    seuil_review_relatif: float = 0.50
) -> list[str]:
    """
    Détecte les dérives de performance par rapport à une référence.

    Alertes générées si :
    - CER estimé a augmenté de plus de seuil_cer_absolu
    - Taux needs_review a augmenté de plus de seuil_review_relatif (relatif)
    """
    alertes = []

    delta_cer = metriques_actuelles.cer_estime_moyen - metriques_reference.cer_estime_moyen
    if delta_cer > seuil_cer_absolu:
        alertes.append(
            f"⚠ Dérive CER : +{delta_cer:.1%} "
            f"({metriques_reference.cer_estime_moyen:.1%} → "
            f"{metriques_actuelles.cer_estime_moyen:.1%})"
        )

    if metriques_reference.taux_needs_review > 0:
        delta_review_rel = (
            (metriques_actuelles.taux_needs_review
             - metriques_reference.taux_needs_review)
            / metriques_reference.taux_needs_review
        )
        if delta_review_rel > seuil_review_relatif:
            alertes.append(
                f"⚠ Dérive taux_review : +{delta_review_rel:.0%} "
                f"({metriques_reference.taux_needs_review:.1%} → "
                f"{metriques_actuelles.taux_needs_review:.1%})"
            )

    if not alertes:
        print("  ✓ Aucune dérive détectée.")
    else:
        print("  Dérives détectées :")
        for alerte in alertes:
            print(f"    {alerte}")

    return alertes
```

---

## 6. Le README du dataset : guide du destinataire

Le fichier `README.md` accompagnant le dataset est le premier document que le module NLP lira. Il doit être autosuffisant.

```markdown
# Dataset de transcription HTR — Manuscrits médiévaux MD5

**Version :** 1.0 | **Date :** Mai 2026 | **Pipeline :** hybride_vote

---

## Résumé

Ce dataset contient les transcriptions automatiques de [N] pages de manuscrits
en vieux et moyen français (IXe–XVe siècle), issues du pipeline Computer Vision
développé dans le module MD5. Il est destiné au module de Traitement du Langage
Naturel pour la normalisation orthographique et l'annotation linguistique.

## Utilisation rapide

```python
from datasets import load_from_disk
ds = load_from_disk("dataset_hf/")
print(ds["train"][0])
# {'page_id': 'bnf_fr768_f001r', 'line_id': 'l0001',
#  'text': 'En cel tems que li rois Artus regnoit',
#  'confidence': 0.84, 'needs_review': False, ...}
```

## Structure du dataset

| Split | Lignes | Description |
|-------|--------|-------------|
| train | N | Lignes fiables (confidence ≥ 0.7) |
| test | N | Lignes fiables, réservées à l'évaluation |
| validation | N | Lignes à confiance intermédiaire (0.5–0.7) |
| needs_review | N | Lignes nécessitant une vérification humaine |

**Recommandation :** n'utiliser que `train` et `test` pour l'entraînement
et l'évaluation NLP. Le split `needs_review` ne doit pas être utilisé
sans vérification humaine préalable.

## Conventions de transcription

Voir `CONVENTIONS_TRANSCRIPTION.md` pour le détail complet.

Points essentiels :
- **Abréviations** : développées (incertitudes entre parenthèses)
- **Lacunes** : encodées `[...]`
- **Rubriques** : balisées `<R>...</R>` dans le flux
- **Encodage** : UTF-8, caractères MUFI pour les symboles médiévaux

## Performances du pipeline

| Métrique | Valeur |
|----------|--------|
| CER estimé (moyen) | ~11% |
| Taux needs_review | ~18% |
| Lignes avec source humaine | ~5% |

## Limitations connues

- Les pages avec fort show-through ont un CER plus élevé.
- Le latin est moins bien transcrit que le français (modèle entraîné sur CREMMA).
- Les écritures du IXe–XIe siècle (carolingienne) sont sous-représentées.

## Licence et attribution

Les images sont la propriété des bibliothèques sources (BnF, etc.).
Les transcriptions sont sous licence CC-BY 4.0.
Citation : [référence du projet MD5]
```

---

## 7. Checklist de livraison

Avant de passer le dataset au module NLP, vérifier chaque point.

```python
def checklist_livraison(dossier_dataset: str) -> bool:
    """
    Vérifie que le dataset est prêt pour la livraison au module NLP.
    Retourne True si tous les points sont validés.
    """
    checks = []
    p = Path(dossier_dataset)

    def check(condition: bool, message: str) -> bool:
        symbole = "✓" if condition else "✗"
        print(f"  {symbole} {message}")
        checks.append(condition)
        return condition

    print("\n=== Checklist de livraison ===\n")

    # Structure
    check((p / "CONVENTIONS_TRANSCRIPTION.md").exists(),
          "Fichier de conventions présent")
    check((p / "README.md").exists(),
          "README présent")
    check((p / "_metadata.json").exists(),
          "Fichier de métadonnées présent")

    # Validation JSON Schema
    fichiers_json = [f for f in p.glob("*.json") if not f.name.startswith("_")]
    check(len(fichiers_json) > 0, f"{len(fichiers_json)} fichiers JSON trouvés")

    n_invalides = sum(
        1 for f in fichiers_json
        if not valider_fichier_json(str(f), verbose=False)[0]
    )
    check(n_invalides == 0,
          f"Tous les JSON valident le schéma ({n_invalides} invalides)")

    # Cohérence
    n_incoherents = 0
    for f in fichiers_json:
        with open(f) as fp:
            doc = json.load(fp)
        avertissements = verifier_coherence_semantique(doc)
        if avertissements:
            n_incoherents += 1
    check(n_incoherents == 0,
          f"Cohérence sémantique OK ({n_incoherents} fichiers incohérents)")

    # Dataset HuggingFace
    hf_path = p.parent / "dataset_hf"
    check(hf_path.exists(),
          "Dataset HuggingFace créé et sauvegardé")

    # Métriques
    with open(p / "_metadata.json") as f:
        meta = json.load(f)
    cer_moyen = np.mean([pg["cer_estime"] for pg in meta.get("pages", []) if "cer_estime" in pg])
    check(cer_moyen < 0.20,
          f"CER estimé moyen < 20% (actuel : {cer_moyen:.1%})")

    # Résultat
    n_ok = sum(checks)
    n_total = len(checks)
    print(f"\n  {n_ok}/{n_total} points validés", end="")
    if n_ok == n_total:
        print(" — ✓ Dataset prêt pour la livraison")
        return True
    else:
        print(f" — ✗ {n_total - n_ok} point(s) à corriger avant livraison")
        return False
```

---

## 8. Connexions avec le pipeline global

La section 4.4 est le **point final du pipeline CV** et le **point de départ du module NLP** :

```
[Pipeline CV — Jours 1 à 4]
        ↓
Checklist de livraison
        ↓
dataset_nlp/          ← Fichiers JSON validés par le schéma
dataset_hf/           ← DatasetDict HuggingFace versionné
CONVENTIONS.md        ← Référence éditoriale
README.md             ← Guide d'utilisation
_metadata.json        ← Métriques de qualité
        ↓
[Module NLP]
• Normalisation orthographique
• Développement des abréviations restantes
• Correction contextuelle par modèle de langue
• Annotation morpho-syntaxique
```

**Ce que le module NLP peut attendre :**
- Des transcriptions cohérentes avec les conventions documentées.
- Des scores de confiance calibrés pour prioriser les corrections.
- Des flags explicites sur les lignes incertaines.
- Un dataset versionné et reproductible.

**Ce que le module NLP ne doit pas attendre :**
- Des transcriptions parfaites — le CER estimé est ~11%.
- Une orthographe normalisée — c'est son rôle.
- Des annotations linguistiques — c'est son rôle.

---

## Glossaire des termes avancés

**Breaking change**
Modification d'un format de données incompatible avec les versions précédentes. Un changement de majeur de version (1.x → 2.0) signale un breaking change dans la convention de versionnage sémantique.

**Changelog**
Document listant les modifications apportées à chaque version d'un logiciel ou d'un dataset. Indispensable pour la traçabilité et la reproductibilité des expériences.

**Data contract**
Accord formel entre producteur et consommateur de données spécifiant la structure, la sémantique, les garanties et les conventions des données échangées.

**JSON Schema**
Standard (IETF) pour la description et la validation de documents JSON. Permet de vérifier automatiquement qu'un document respecte la structure et les types attendus. Versionné (Draft 4, 7, 2020-12).

**MUFI (*Medieval Unicode Font Initiative*)**
Initiative académique qui définit des points de code Unicode pour les caractères spéciaux utilisés dans les manuscrits médiévaux non couverts par Unicode standard. Référence pour l'encodage de caractères comme ȝ, ꝑ, ⁊.

**Monitoring de dérive**
Surveillance continue des métriques de performance d'un pipeline en production pour détecter les dégradations progressives dues à un changement de distribution des données d'entrée (distribution shift).

**Semver (*Semantic Versioning*)**
Convention de versionnage : `MAJEUR.MINEUR.PATCH`. Majeur : breaking change. Mineur : nouvelle fonctionnalité rétrocompatible. Patch : correction de bug. Adoptée pour le versioning du dataset.

**Split (train/test/validation)**
Partition d'un dataset en sous-ensembles aux rôles différents. Train : entraînement du modèle. Validation : ajustement des hyperparamètres. Test : évaluation finale (ne jamais l'utiliser pendant l'entraînement).

**Traceback (retrace)**
Phase de l'algorithme de programmation dynamique (Needleman-Wunsch, Viterbi…) qui reconstruit la solution optimale depuis la matrice remplie. Produit l'alignement explicite, pas seulement le score.

---

## Bibliographie de référence

### Data contracts et ingénierie des données

- **Majors, A.** (2023). *Deciphering Data Architectures*. O'Reilly Media. Chapitre 7 : *Data Contracts*. — Introduction aux data contracts dans les architectures de données modernes.

- **Sato, C., Bhatt, A.** (2022). *Designing Machine Learning Systems*. O'Reilly Media. Chapitre 6 : *Feature Engineering*, section *Data Contracts*. — Les data contracts dans le contexte des systèmes ML en production.

### JSON Schema

- **Droettboom, M. et al.** (2020). *JSON Schema: A Media Type for Describing JSON Documents*. IETF Internet-Draft. [json-schema.org](https://json-schema.org) — La spécification de référence JSON Schema Draft 2020-12.

### Versioning de datasets

- **Hutchinson, B. et al.** (2021). *Towards Accountability for Machine Learning Datasets*. FAccT 2021. [arXiv:2010.13561] — Sur la documentation et la traçabilité des datasets ML.

- **HuggingFace** (2021). *datasets: A Community Library for Natural Language Processing*. EMNLP 2021. — La bibliothèque `datasets` pour la gestion de datasets ML.

### Conventions éditoriales pour les manuscrits

- **Guyotjeannin, O., Vielliard, F.** (dir.) (2001). *Conseils pour l'édition des textes médiévaux*, 3 vol. CTHS/École nationale des chartes, Paris. — La référence philologique pour les conventions de transcription.

- **Burnard, L., Sperberg-McQueen, C. M.** (TEI Consortium). *TEI P5 Guidelines*. [tei-c.org/guidelines/](https://tei-c.org/guidelines/) — Référence pour les conventions TEI sous-jacentes à nos choix éditoriaux.

- **MUFI** (Medieval Unicode Font Initiative). *Character Recommendation*. [skaldic.abdn.ac.uk/mufi/](https://skaldic.abdn.ac.uk/mufi/) — Encodage Unicode des caractères médiévaux.

---

*Support de cours rédigé pour le module Computer Vision · Promotion MD5 · Mai 2026*
*Ce document accompagne la séance 4.4 du Jour 4.*
*Il suppose la lecture des sections 4.1 à 4.3.*
*Durée estimée de lecture : 45 minutes (section dense malgré sa courte durée en cours).*
