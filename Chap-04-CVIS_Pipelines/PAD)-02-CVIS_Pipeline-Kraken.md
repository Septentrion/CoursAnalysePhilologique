# Cours 4.2 — Le pipeline Kraken intégré : baselines, ordre de lecture et sérialisation

**Module 4 · Computer Vision appliquée aux manuscrits médiévaux · MD5**

---

> *« Kraken ne demande pas comment segmenter une page — il demande où sont les lignes de base, dans quel ordre les lire, et comment enregistrer le résultat pour que d'autres outils puissent en hériter. Ces trois questions définissent un système complet, pas seulement un modèle. »*

---

## Introduction : deux pipelines pour le même problème

Le Jour 3 a construit un pipeline modulaire : SAM segmente le layout, Kraken détecte les lignes, TrOCR transcrit. Chaque composant est optimisable indépendamment, et l'ensemble atteint d'excellentes performances sur les manuscrits difficiles.

Mais ce pipeline a un coût : il nécessite plusieurs modèles, un GPU pour TrOCR, une gestion manuelle de l'ordre de lecture, et un format de sortie JSON qui n'est pas nativement compatible avec les outils des humanités numériques. Un chercheur en histoire médiévale qui veut corriger les transcriptions dans eScriptorium ne peut pas ouvrir un fichier JSON.

Kraken propose une réponse différente — non pas meilleure en absolu, mais plus adaptée à certains contextes. Son pipeline ATR (*Automatic Text Recognition*) est **intégré** : un seul outil gère la segmentation par baselines, l'ordre de lecture, la transcription, et la sérialisation dans des formats standard. Il est entraînable et ré-entraînable dans une boucle de rétroaction qui est le workflow quotidien des projets patrimoniaux professionnels.

Cette section couvre ce pipeline de A à Z — depuis la notion de baseline, souvent incomprise, jusqu'à la boucle de rétroaction qui permet d'améliorer progressivement un modèle sur un corpus donné.

---

## 1. Pourquoi pas des boîtes ? La géométrie réelle d'une ligne manuscrite

### 1.1 Ce qu'est une ligne de texte — du point de vue du scribe

Quand un copiste médiéval écrit, il suit une ligne de réglure tracée au préalable à la pointe sèche ou à la mine de plomb. Cette ligne imaginaire s'appelle la **ligne de base** (*baseline*) : c'est le trait sur lequel reposent les lettres, en dessous duquel descendent les **jambages descendants** (*descenders*) du *p*, du *q*, du *g*, et au-dessus duquel s'élèvent les **jambages ascendants** (*ascenders*) du *l*, du *h*, du *d*.

La ligne de base n'est pas une ligne horizontale parfaite sur le parchemin vieilli :

- Le parchemin se déforme avec l'humidité et le temps. Une feuille plate lors de sa fabrication peut présenter des ondulations millénaires qui font légèrement varier la hauteur de la ligne.
- La reliure courbe les pages. Sur un codex ouvert, les lignes situées près de la couture suivent la courbure de la reliure — elles ne sont plus horizontales dans l'image plane.
- Le scribe lui-même n'est pas parfaitement régulier. Sur les documents cursifs rapides (registres, chartes), les lignes montent ou descendent légèrement au fil de l'écriture.

### 1.2 La limite des boîtes englobantes rectangulaires

Une boîte englobante (*bounding box*) est le plus petit rectangle horizontal qui contient tous les pixels d'une ligne. C'est l'approche la plus simple — et celle que SAM, la plupart des détecteurs d'objets, et les premières générations de systèmes HTR utilisent.

Pour une ligne de texte parfaitement horizontale, la boîte fonctionne bien. Pour une ligne courbée ou inclinée sur un manuscrit, trois problèmes se posent :

**Problème 1 : les débordements verticaux.** Pour une ligne courbée (en arc), la boîte rectangulaire doit englober les extrémités de l'arc, ce qui inclut une partie du fond au-dessus et en dessous du texte réel — et parfois des parties des lignes adjacentes. Le modèle HTR reçoit alors une image contenant des morceaux de lettres étrangères à la ligne.

**Problème 2 : les décalages horizontaux.** Si la ligne est inclinée à 5 degrés, la boîte inclut du fond vide dans les coins supérieur-gauche et inférieur-droit. Sur 300 pixels de large, un angle de 5 degrés déplace le bord inférieur de 26 pixels par rapport au bord supérieur — ce qui représente environ une demi-hauteur de lettre. L'image de ligne est déformée.

**Problème 3 : les lettrines multi-lignes.** Une lettrine ornée qui s'étend sur 4 lignes de texte crée une boîte qui chevauche toutes ces lignes. Le découpage par boîte inclut la lettrine dans chacune des 4 lignes, ou tente de l'exclure de toutes — dans les deux cas, le résultat est incorrect.

### 1.3 Le modèle des baselines et des polygones

La solution de Kraken est de représenter chaque ligne non pas par une boîte rectangulaire, mais par deux éléments complémentaires :

**La baseline** : une **polyligne** — une série de points $(x_1, y_1), (x_2, y_2), \ldots, (x_n, y_n)$ — qui suit la ligne de base réelle du texte. La baseline est tracée au niveau du bas des lettres sans jambage descendant (le bas de *a*, *e*, *m*, *n*, *o*, *r*…).

**Le polygone englobant** (*boundary polygon*) : un polygone convexe ou non convexe qui délimite exactement la zone de pixels appartenant à cette ligne — en suivant le haut des ascendants et le bas des descendants, et en évitant d'empiéter sur les lignes adjacentes.

```
Ligne de texte réelle (courbée) :

  ┌─────────────────────────────────────────────┐  ← Haut du polygone
  │   l   i   r   e   u   n   e   l   i   g   n   e  │
  │_._._._._._._._._._._._._._._._._._._._._._._│  ← baseline (polyligne)
  │   p   q   y   g                             │
  └─────────────────────────────────────────────┘  ← Bas du polygone

Avec une boîte rectangulaire :
  ╔═════════════════════════════════════════════╗
  ║  Zone vide en haut-gauche   l  i  r  ...   ║
  ║                       p  q  y  g  Zone vide║
  ╚═════════════════════════════════════════════╝

Avec un polygone :
  ╱‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾╲
  │ l  i  r  e  u  n  e  l  i  g  n  e          │
  │..................................baseline....│
  │ p  q  y  g                                  │
  ╲____________________________________________╱
```

Le polygone permet d'extraire exactement les pixels appartenant à la ligne, sans inclure les pixels des lignes adjacentes ni le fond vide des coins d'une boîte rectangulaire.

### 1.4 BLLA : le modèle de segmentation de Kraken

**BLLA** (*Baseline Layout Analysis*) est le module de segmentation de Kraken. C'est un réseau de neurones entièrement convolutif qui prend en entrée une image de page (en niveaux de gris ou couleur) et produit :

- Une **carte de baselines** : une image où les pixels appartenant à une baseline ont une valeur élevée.
- Une **carte de régions** : une image sémantique indiquant le type de chaque région (texte, image, marge, etc.)

À partir de ces cartes, un algorithme de post-traitement extrait les polylignes individuelles et leurs polygones associés.

```python
from kraken import blla
from kraken.lib import models as kraken_models
from PIL import Image
import numpy as np
import json

# ── Chargement du modèle de segmentation ───────────────────────────────────────
# Les modèles BLLA sont distribués via HTR-United et le dépôt Kraken
# Téléchargement d'un modèle pré-entraîné pour les manuscrits médiévaux :
# kraken get blla.mlmodel
SEG_MODEL_PATH = "modeles/blla.mlmodel"

def charger_modele_segmentation(chemin: str):
    """Charge un modèle BLLA depuis le disque."""
    try:
        modele = kraken_models.load_any(chemin)
        print(f"  Modèle BLLA chargé : {chemin}")
        return modele
    except FileNotFoundError:
        print(f"  Modèle non trouvé : {chemin}")
        print("  Téléchargement : kraken get blla.mlmodel")
        return None


def segmenter_page_blla(
    image_pil: Image.Image,
    modele_seg,
    texte_seulement: bool = False
) -> object:
    """
    Segmente une page de manuscrit par BLLA.

    Args:
        image_pil      : image PIL de la page prétraitée (section 4.1)
        modele_seg     : modèle BLLA chargé
        texte_seulement: si True, ignore les régions non-texte

    Returns:
        Objet de segmentation Kraken contenant :
        - .lines   : liste des lignes avec baseline et polygone
        - .regions : zones détectées par type
        - .text_direction : direction du texte (ltr, rtl, ttb)
    """
    # BLLA attend une image en mode RGB ou L (niveaux de gris)
    if image_pil.mode not in ("RGB", "L"):
        image_pil = image_pil.convert("RGB")

    resultats = blla.segment(
        image_pil,
        model=modele_seg,
        text_direction="horizontal-lr",   # Gauche à droite pour le français
    )

    print(f"  Lignes détectées     : {len(resultats.lines)}")
    print(f"  Régions détectées    : {len(resultats.regions)}")
    print(f"  Types de régions     : {list(set(r for r in resultats.regions.keys()))}")

    return resultats


def inspecter_segmentation(resultats, n_lignes: int = 5) -> None:
    """Affiche les données de segmentation des premières lignes."""
    print(f"\nInspection des {min(n_lignes, len(resultats.lines))} premières lignes :")
    for i, ligne in enumerate(resultats.lines[:n_lignes]):
        print(f"\n  Ligne {i+1} :")
        print(f"    Baseline   : {ligne.baseline[:3]}...")   # Premiers points
        print(f"    Polygone   : {len(ligne.boundary)} sommets")
        print(f"    Tags       : {ligne.tags}")
```

### 1.5 Extraction et redressement des lignes par la baseline

Une fois la baseline détectée, l'extraction de l'image de ligne se fait en deux étapes :

1. **Découpe du polygone** : masquer tous les pixels hors du polygone englobant.
2. **Redressement** (*dewarping* local) : appliquer une transformation qui « déplie » la ligne courbée pour produire une image horizontale de hauteur fixe — c'est ce que le modèle HTR recevra.

```python
from kraken.lib.segmentation import extract_polygons
import cv2

def extraire_images_lignes(
    image_pil: Image.Image,
    resultats_blla,
    hauteur_cible: int = 48
) -> list[dict]:
    """
    Extrait et redresse les images de lignes depuis les polygones BLLA.

    Le redressement aligne la ligne sur la baseline, produisant une image
    horizontale même si la ligne originale était courbée ou inclinée.
    """
    image_np = np.array(image_pil.convert("L"))
    lignes_extraites = []

    for i, ligne in enumerate(resultats_blla.lines):
        try:
            # Kraken fournit une fonction d'extraction qui :
            # 1. Découpe le polygone
            # 2. Redresse la ligne sur la baseline
            # 3. Normalise à hauteur_cible pixels
            img_ligne = extract_polygons(
                image_pil,
                ligne.baseline,
                ligne.boundary,
                target_height=hauteur_cible
            )

            lignes_extraites.append({
                "id"      : f"l{i:04d}",
                "image"   : img_ligne,
                "baseline": ligne.baseline,
                "polygone": ligne.boundary,
                "tags"    : ligne.tags,
            })
        except Exception as e:
            print(f"  ⚠ Erreur extraction ligne {i} : {e}")

    print(f"  {len(lignes_extraites)} lignes extraites et redressées")
    return lignes_extraites
```

---

## 2. L'ordre de lecture : un problème non trivial

### 2.1 Pourquoi l'ordre de lecture est un vrai problème

Considérons une page typique d'un manuscrit médiéval en vieux français du XIVe siècle :

- **Deux colonnes de texte** séparées par un espace blanc vertical.
- **Une rubrique** en rouge en haut de la colonne de gauche, sur deux lignes.
- **Une lettrine ornée** de grande taille (4 lignes de haut) au début de la colonne de gauche, avec le texte qui contourne.
- **Une note marginale** dans la marge de droite, annotant la ligne 7 de la colonne de droite.
- **Un colophon** centré en bas de page, après les deux colonnes.

La question est simple à formuler mais complexe à résoudre algorithmiquement : dans quel ordre faut-il lire ces éléments pour obtenir un texte cohérent ?

La réponse humaine est immédiate (rubrique → lettrine + début de colonne gauche → suite de colonne gauche → colonne droite → colophon, avec la note marginale associée à sa ligne) parce qu'elle s'appuie sur des conventions visuelles et typographiques que nous avons intériorisées. Ces conventions ne sont pas universelles — elles varient selon le type de document, le siècle, et la région de production du manuscrit.

**Les cas ambigus les plus fréquents dans les manuscrits médiévaux :**

| Situation | Ambiguïté d'ordre de lecture |
|-----------|------------------------------|
| Deux colonnes | Faut-il lire toute la colonne gauche avant de commencer la droite ? (oui, en général) |
| Rubrique en tête de section | La rubrique précède-t-elle ou suit-elle la dernière ligne de la section précédente ? |
| Note marginale | Quand insérer la note dans le flux : à sa position verticale exacte, ou à la fin de la page ? |
| Lettrines multi-lignes | La lettrine est-elle le premier caractère de la première ligne de texte, ou un élément séparé ? |
| Colophon centré | S'agit-il de la continuation du texte ou d'un élément éditorial distinct ? |
| Colonnes de registre | L'en-tête horizontal précède-t-il toutes les lignes de données, ou chaque colonne est-elle lue indépendamment ? |

### 2.2 L'approche de Kraken : graphe topologique

Kraken représente l'ordre de lecture comme un problème de **tri topologique sur un graphe de lignes**. Chaque ligne est un nœud ; les arêtes encodent une relation de précédence (« cette ligne se lit avant celle-là »).

Les arêtes sont construites à partir de règles heuristiques basées sur la géométrie des lignes et leur appartenance à des régions :

**Règle 1 — Appartenance à une région.** Les lignes appartenant à la même région (même colonne de texte, même zone) sont groupées et ordonnées ensemble avant les lignes d'une autre région.

**Règle 2 — Ordre vertical au sein d'une région.** Au sein d'une région, les lignes sont triées par position verticale croissante (de haut en bas). La position de référence est le point médian de la baseline.

**Règle 3 — Ordre des régions entre elles.** Les régions sont elles-mêmes ordonnées : d'abord de gauche à droite (pour un texte à lecture horizontal gauche-droite), puis de haut en bas pour les régions de même position horizontale.

**Règle 4 — Cas spéciaux déclarés.** Certains types de régions ont un ordre prédéfini : les en-têtes (*header*) toujours en premier, les pieds-de-page (*footer*) toujours en dernier, les annotations marginales (*marginalia*) intercalées à leur position verticale.

```python
def inspecter_ordre_lecture(resultats_blla) -> None:
    """
    Affiche l'ordre de lecture calculé par BLLA.
    Montre la position de chaque ligne dans la séquence finale.
    """
    print("\nOrdre de lecture calculé par BLLA :")
    print(f"{'Rang':>5} | {'Région':20} | {'Y (baseline)':>12} | Tags")
    print("-" * 65)

    # Kraken encode l'ordre de lecture dans l'ordre des lignes retournées
    for rang, ligne in enumerate(resultats_blla.lines):
        # Récupérer la position verticale médiane de la baseline
        if ligne.baseline:
            y_median = int(np.median([pt[1] for pt in ligne.baseline]))
        else:
            y_median = -1

        # Région d'appartenance
        region = "inconnue"
        for type_region, lignes_region in resultats_blla.regions.items():
            if any(ligne in lignes_region for _ in [True]):
                region = type_region
                break

        tags_str = str(ligne.tags)[:30] if ligne.tags else ""
        print(f"{rang+1:>5} | {region:20} | {y_median:>12} | {tags_str}")
```

### 2.3 Les limites de l'ordre de lecture automatique

Le tri topologique de Kraken gère bien les cas standards (une ou deux colonnes de texte continu) mais échoue sur les mises en page complexes :

**Textes en U ou en L.** Certains manuscrits entourent une illustration centrale avec du texte. L'ordre de lecture suit le pourtour de l'illustration, mais l'algorithme topologique lit les lignes par position verticale, ce qui produit un ordre incorrect.

**Textes croisés.** Des tables où les colonnes et les lignes portent des informations complémentaires — on lit d'abord les en-têtes, puis chaque cellule dans un ordre matriciel — ne sont pas gérées nativement.

**Annotations marginales avec ancre textuelle précise.** Si une note marginale annote un mot précis d'une ligne (et non la ligne entière), l'ordre optimal dépend de la position du mot ancre — une information que le système de segmentation ne connaît pas.

**La solution pragmatique** pour ces cas est de laisser l'ordre de lecture automatique, puis de permettre à l'annotateur humain de le corriger dans eScriptorium avant le réentraînement.

### 2.4 Annotation de l'ordre de lecture dans eScriptorium

eScriptorium offre une interface graphique pour corriger l'ordre de lecture : les lignes sont numérotées dans leur ordre courant, et l'annotateur peut les réordonner par glisser-déposer ou en assignant manuellement un numéro de rang.

```bash
# Lancer eScriptorium localement (via Docker)
docker run -p 8080:8080 registry.gitlab.com/scripta.psl.eu/escriptorium/escriptorium

# Ou via l'instance publique : escriptorium.fr (compte nécessaire)
```

Une fois l'ordre corrigé dans eScriptorium, l'export en PAGE XML préserve cet ordre dans les attributs `readingOrder` des éléments `TextLine`.

---

## 3. La reconnaissance HTR avec Kraken

### 3.1 Le modèle de reconnaissance

Le module de reconnaissance de Kraken (*Kraken OCR*) est un réseau LSTM-CTC entraîné à transcrire des images de lignes extraites par BLLA. Contrairement à TrOCR (encodeur-décodeur autorégressif), Kraken utilise la CTC — ce qui le rend plus rapide mais sans modèle de langue implicite.

```python
from kraken import rpred
from kraken.lib import models as kraken_models

def charger_modele_htr_kraken(chemin: str):
    """Charge un modèle HTR Kraken depuis le disque."""
    modele = kraken_models.load_any(chemin)
    print(f"  Modèle HTR chargé  : {modele.name if hasattr(modele, 'name') else chemin}")
    print(f"  Alphabet           : {len(modele.codec)} caractères")
    return modele


def reconnaitre_lignes_kraken(
    image_pil: Image.Image,
    resultats_blla,
    modele_htr,
    pad: int = 16
) -> list[dict]:
    """
    Transcrit les lignes segmentées par BLLA avec le modèle HTR Kraken.

    Args:
        image_pil     : image de la page originale
        resultats_blla: résultats de segmentation BLLA
        modele_htr    : modèle HTR Kraken chargé
        pad           : pixels de marge autour de chaque ligne

    Returns:
        Liste de dicts {id, transcription, confiance, baseline, polygone}
    """
    # rpred.rpred : reconnaissance sur des lignes avec baseline
    predictions = rpred.rpred(
        network=modele_htr,
        im=image_pil,
        bounds=resultats_blla,
        pad=pad,
        bidi_reorder=True    # Réorganisation bidi pour le texte hébreu/arabe (optionnel)
    )

    resultats = []
    for i, pred in enumerate(predictions):
        resultats.append({
            "id"            : f"l{i:04d}",
            "transcription" : pred.prediction,
            "confiance"     : float(np.mean([c for _, c in pred.cuts]))
                              if pred.cuts else 0.0,
            "baseline"      : pred.line.baseline,
            "polygone"      : pred.line.boundary,
            "cuts"          : pred.cuts,   # Alignement caractère-position
        })

    return resultats
```

### 3.2 Le pipeline complet en une passe

```python
def pipeline_kraken_complet(
    chemin_image: str,
    modele_seg,
    modele_htr,
    chemin_sortie_xml: str = None
) -> dict:
    """
    Pipeline ATR Kraken complet : image → transcription + PAGE XML.

    Étapes :
    1. Chargement et prétraitement (section 4.1)
    2. Segmentation BLLA (baselines + ordre de lecture)
    3. Reconnaissance HTR (LSTM-CTC)
    4. Sérialisation PAGE XML (section 4.2.4)

    Returns:
        dict {page_id, n_lignes, lignes, page_xml}
    """
    from pathlib import Path

    page_id = Path(chemin_image).stem
    print(f"\n=== Pipeline Kraken — {page_id} ===")

    # ── Étape 1 : chargement et prétraitement ───────────────────────────────
    image_pil = Image.open(chemin_image).convert("RGB")
    # (le prétraitement complet est appliqué en section 4.1)
    print(f"  Image chargée : {image_pil.size[0]} × {image_pil.size[1]} px")

    # ── Étape 2 : segmentation BLLA ─────────────────────────────────────────
    resultats_seg = segmenter_page_blla(image_pil, modele_seg)

    # ── Étape 3 : reconnaissance HTR ────────────────────────────────────────
    resultats_htr = reconnaitre_lignes_kraken(image_pil, resultats_seg, modele_htr)
    print(f"  Transcription  : {len(resultats_htr)} lignes")

    # Statistiques de confiance
    confidences = [r["confiance"] for r in resultats_htr if r["confiance"] > 0]
    if confidences:
        print(f"  Confiance moy. : {np.mean(confidences):.3f}")
        print(f"  Lignes < 0.7   : {sum(c < 0.7 for c in confidences)}")

    # ── Étape 4 : sérialisation ──────────────────────────────────────────────
    page_xml = serialiser_page_xml(
        page_id, image_pil.size, resultats_seg, resultats_htr
    )

    if chemin_sortie_xml:
        with open(chemin_sortie_xml, "w", encoding="utf-8") as f:
            f.write(page_xml)
        print(f"  PAGE XML exporté : {chemin_sortie_xml}")

    return {
        "page_id"       : page_id,
        "n_lignes"      : len(resultats_htr),
        "lignes"        : resultats_htr,
        "segmentation"  : resultats_seg,
        "page_xml"      : page_xml,
    }
```

---

## 4. La sérialisation : PAGE XML et ALTO

### 4.1 Pourquoi les formats XML spécialisés existent

Le résultat d'un pipeline HTR contient deux types d'information indissociables :

1. **Le contenu textuel** : les chaînes de caractères transcrites.
2. **L'information spatiale** : les coordonnées de chaque région, ligne et mot dans l'image d'origine.

Un fichier texte brut (`.txt`) ne conserve que le contenu. Un fichier JSON personnalisé comme celui produit par notre pipeline modulaire conserve les deux, mais dans un format que seuls nos propres scripts savent lire.

Les formats XML spécialisés pour les documents — **PAGE XML** et **ALTO** — sont des standards internationaux qui encodent ces deux types d'information dans un format lisible par des dizaines d'outils : eScriptorium, Transkribus, Gamera, FineReader, les portails Gallica et Europeana. Ce sont eux que les bibliothèques, archives et instituts de recherche utilisent pour déposer et partager leurs données transcrites.

Comprendre ces formats, c'est comprendre le langage commun des humanités numériques.

### 4.2 PAGE XML : structure et éléments essentiels

**PAGE XML** (*Page Analysis and Ground Truth Elements*) a été développé dans le cadre du projet européen PRIMA (Pattern Recognition and Image analysis) à partir de 2010. Il est maintenu par le consortium PRImA Research Ltd. et est devenu le standard de facto pour l'annotation de documents dans les projets HTR.

Un fichier PAGE XML est un document XML dont l'élément racine est `PcGts`. Sa structure reflète directement la hiérarchie physique d'une page de document :

```
PcGts (racine)
└── Page (une page du document)
    ├── TextRegion (une zone de texte)
    │   ├── Coords (coordonnées du polygone de la zone)
    │   └── TextLine (une ligne de texte)
    │       ├── Coords (coordonnées du polygone de la ligne)
    │       ├── Baseline (coordonnées de la ligne de base)
    │       └── TextEquiv (transcription)
    │           └── Unicode (la chaîne de caractères)
    ├── ImageRegion (une illustration)
    │   └── Coords
    └── MarginaliaRegion (une note marginale)
        └── Coords
```

```python
def serialiser_page_xml(
    page_id: str,
    taille_image: tuple,              # (largeur, hauteur) en pixels
    resultats_seg,                    # Résultat BLLA
    resultats_htr: list[dict],
    creator: str = "Pipeline Kraken MD5",
    version: str = "2019-07-15"
) -> str:
    """
    Génère un fichier PAGE XML conforme au schéma PRImA.
    Compatible eScriptorium, Transkribus, HTR-United.
    """
    from datetime import datetime
    import xml.etree.ElementTree as ET

    # Espace de noms PAGE XML
    NS   = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
    XSI  = "http://www.w3.org/2001/XMLSchema-instance"
    SLOC = (f"http://schema.primaresearch.org/PAGE/gts/pagecontent/{version} "
            f"http://schema.primaresearch.org/PAGE/gts/pagecontent/{version}"
            f"/pagecontent.xsd")

    ET.register_namespace("",    NS)
    ET.register_namespace("xsi", XSI)

    # ── Élément racine ──────────────────────────────────────────────────────
    root = ET.Element(f"{{{NS}}}PcGts")
    root.set(f"{{{XSI}}}schemaLocation", SLOC)

    # ── Métadonnées ─────────────────────────────────────────────────────────
    metadata = ET.SubElement(root, f"{{{NS}}}Metadata")
    ET.SubElement(metadata, f"{{{NS}}}Creator").text = creator
    ET.SubElement(metadata, f"{{{NS}}}Created").text = datetime.now().isoformat()
    ET.SubElement(metadata, f"{{{NS}}}LastChange").text = datetime.now().isoformat()

    # ── Page ─────────────────────────────────────────────────────────────────
    largeur, hauteur = taille_image
    page = ET.SubElement(root, f"{{{NS}}}Page")
    page.set("imageFilename", f"{page_id}.tif")
    page.set("imageWidth",    str(largeur))
    page.set("imageHeight",   str(hauteur))

    # Ordre de lecture global
    reading_order = ET.SubElement(page, f"{{{NS}}}ReadingOrder")
    ro_group = ET.SubElement(reading_order, f"{{{NS}}}OrderedGroup")
    ro_group.set("id", "ro_main")

    # ── Régions ──────────────────────────────────────────────────────────────
    # Construire un index {ligne_id → résultat_htr}
    htr_par_ligne = {r["id"]: r for r in resultats_htr}

    # Grouper les lignes par région
    for region_type, lignes_region in resultats_seg.regions.items():

        # Déterminer le type de région PAGE XML
        if "text" in region_type.lower():
            tag_region = f"{{{NS}}}TextRegion"
        elif "image" in region_type.lower() or "illustration" in region_type.lower():
            tag_region = f"{{{NS}}}ImageRegion"
        elif "margin" in region_type.lower():
            tag_region = f"{{{NS}}}MarginaliaRegion"
        else:
            tag_region = f"{{{NS}}}TextRegion"

        region_elem = ET.SubElement(page, tag_region)
        region_id   = f"region_{region_type}_{id(lignes_region)}"
        region_elem.set("id", region_id)
        region_elem.set("type", region_type)

        # Ajout dans l'ordre de lecture
        roi_ref = ET.SubElement(ro_group, f"{{{NS}}}RegionRefIndexed")
        roi_ref.set("regionRef", region_id)
        roi_ref.set("index",     str(len(list(ro_group))))

        # Coordonnées de la région (boîte englobante des lignes)
        if lignes_region:
            coords_region = _calculer_coords_region(lignes_region)
            coords_elem   = ET.SubElement(region_elem, f"{{{NS}}}Coords")
            coords_elem.set("points", _points_to_string(coords_region))

        # ── Lignes de texte dans la région ───────────────────────────────────
        for idx_ligne, ligne in enumerate(resultats_seg.lines):
            ligne_id = f"l{resultats_seg.lines.index(ligne):04d}"

            line_elem = ET.SubElement(region_elem, f"{{{NS}}}TextLine")
            line_elem.set("id", ligne_id)
            line_elem.set("readingOrder", str(idx_ligne))

            # Polygone (Coords)
            if ligne.boundary:
                coords = ET.SubElement(line_elem, f"{{{NS}}}Coords")
                coords.set("points", _points_to_string(ligne.boundary))

            # Baseline
            if ligne.baseline:
                baseline_elem = ET.SubElement(line_elem, f"{{{NS}}}Baseline")
                baseline_elem.set("points", _points_to_string(ligne.baseline))

            # Transcription (TextEquiv)
            if ligne_id in htr_par_ligne:
                htr = htr_par_ligne[ligne_id]
                text_equiv = ET.SubElement(line_elem, f"{{{NS}}}TextEquiv")
                if htr["confiance"] > 0:
                    text_equiv.set("conf", f"{htr['confiance']:.4f}")
                ET.SubElement(text_equiv, f"{{{NS}}}Unicode").text = htr["transcription"]

    # Sérialisation en XML
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")   # Python 3.9+

    import io
    buffer = io.BytesIO()
    tree.write(buffer, encoding="utf-8", xml_declaration=True)
    return buffer.getvalue().decode("utf-8")


def _points_to_string(points: list) -> str:
    """Convertit une liste de tuples [(x,y),...] en chaîne 'x1,y1 x2,y2 ...'"""
    return " ".join(f"{int(x)},{int(y)}" for x, y in points)


def _calculer_coords_region(lignes: list) -> list:
    """Calcule le polygone englobant d'un ensemble de lignes."""
    tous_points = []
    for ligne in lignes:
        if hasattr(ligne, "boundary") and ligne.boundary:
            tous_points.extend(ligne.boundary)
    if not tous_points:
        return [(0, 0), (100, 0), (100, 100), (0, 100)]
    xs = [p[0] for p in tous_points]
    ys = [p[1] for p in tous_points]
    return [(min(xs), min(ys)), (max(xs), min(ys)),
            (max(xs), max(ys)), (min(xs), max(ys))]
```

### 4.3 ALTO : l'alternative de la Library of Congress

**ALTO** (*Analyzed Layout and Text Object*) est un format XML développé par la Bibliothèque nationale d'Autriche et maintenu par la Library of Congress. Il est utilisé par Gallica (BnF), la British Library, et la plupart des grandes bibliothèques nationales pour leurs données de numérisation.

La structure ALTO est légèrement différente de PAGE XML mais encode les mêmes informations :

```xml
<?xml version="1.0" encoding="UTF-8"?>
<alto xmlns="http://www.loc.gov/standards/alto/v3#"
      xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
      xsi:schemaLocation="http://www.loc.gov/standards/alto/v3# ...">

  <Description>
    <MeasurementUnit>pixel</MeasurementUnit>
    <sourceImageInformation>
      <fileName>scan_manuscrit.tif</fileName>
    </sourceImageInformation>
  </Description>

  <Layout>
    <Page WIDTH="3200" HEIGHT="4500" ID="page_1">
      <PrintSpace HPOS="120" VPOS="80" WIDTH="2960" HEIGHT="4340">

        <!-- Bloc de texte (zone de texte) -->
        <TextBlock ID="block_1" HPOS="120" VPOS="80" WIDTH="1400" HEIGHT="4200">

          <!-- Ligne de texte -->
          <TextLine ID="line_1"
                    HPOS="120" VPOS="120" WIDTH="1380" HEIGHT="60"
                    BASELINE="50 170 1380 172">
            <!-- Mots avec confiance -->
            <String ID="word_1" CONTENT="En" HPOS="120" VPOS="120"
                    WIDTH="80" HEIGHT="50" WC="0.95"/>
            <SP WIDTH="20" HPOS="200" VPOS="120"/>
            <String ID="word_2" CONTENT="cel" HPOS="220" VPOS="120"
                    WIDTH="90" HEIGHT="50" WC="0.88"/>
            <!-- ... -->
          </TextLine>

        </TextBlock>
      </PrintSpace>
    </Page>
  </Layout>

</alto>
```

**Différences clés PAGE XML vs ALTO :**

| Critère | PAGE XML | ALTO |
|---------|----------|------|
| Granularité native | Ligne (baseline + polygone) | Mot (boîte rectangulaire) |
| Support des baselines | Natif (balise `<Baseline>`) | Attribut `BASELINE` (points) |
| Polygones non rectangulaires | Natif (`<Coords>` quelconques) | Non standard |
| Ordre de lecture | Natif (`<ReadingOrder>`) | Limité (ordre des éléments XML) |
| Utilisé par | eScriptorium, Transkribus | Gallica, British Library, Europeana |
| Standard maintenu par | PRImA Research Ltd. | Library of Congress |

**Conversion entre formats**

Kraken produit nativement PAGE XML. La conversion vers ALTO (pour Gallica par exemple) peut être effectuée avec des outils dédiés :

```bash
# Conversion PAGE XML → ALTO via Transkribus tools (ligne de commande)
# Ou via le module Python page2alto

pip install page2alto
page2alto input.xml output.alto.xml
```

### 4.4 Lire un fichier PAGE XML existant

Pour analyser les fichiers produits par eScriptorium ou importés depuis HTR-United :

```python
def lire_page_xml(chemin_xml: str) -> dict:
    """
    Parse un fichier PAGE XML et retourne les transcriptions
    avec leurs coordonnées spatiales.
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(chemin_xml)
    root = tree.getroot()

    # Détecter le namespace
    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"

    # Trouver la page
    page = root.find(f"{ns}Page")
    if page is None:
        raise ValueError("Pas d'élément Page dans le fichier")

    img_width  = int(page.get("imageWidth",  0))
    img_height = int(page.get("imageHeight", 0))
    img_file   = page.get("imageFilename", "")

    # Extraire toutes les TextLines
    lignes = []
    for line in root.iter(f"{ns}TextLine"):
        # Baseline
        baseline_elem = line.find(f"{ns}Baseline")
        baseline = None
        if baseline_elem is not None:
            pts_str = baseline_elem.get("points", "")
            baseline = [
                tuple(int(v) for v in pt.split(","))
                for pt in pts_str.split()
                if "," in pt
            ]

        # Polygone
        coords_elem = line.find(f"{ns}Coords")
        polygone = None
        if coords_elem is not None:
            pts_str = coords_elem.get("points", "")
            polygone = [
                tuple(int(v) for v in pt.split(","))
                for pt in pts_str.split()
                if "," in pt
            ]

        # Transcription
        text_equiv = line.find(f"{ns}TextEquiv")
        transcription = ""
        confiance = 0.0
        if text_equiv is not None:
            unicode_elem = text_equiv.find(f"{ns}Unicode")
            if unicode_elem is not None:
                transcription = unicode_elem.text or ""
            confiance_str = text_equiv.get("conf", "0")
            try:
                confiance = float(confiance_str)
            except ValueError:
                confiance = 0.0

        lignes.append({
            "id"           : line.get("id", ""),
            "transcription": transcription,
            "confiance"    : confiance,
            "baseline"     : baseline,
            "polygone"     : polygone,
            "reading_order": int(line.get("readingOrder", 0)),
        })

    # Trier par ordre de lecture
    lignes.sort(key=lambda l: l["reading_order"])

    return {
        "image_file"  : img_file,
        "image_width" : img_width,
        "image_height": img_height,
        "n_lignes"    : len(lignes),
        "lignes"      : lignes,
    }
```

---

## 5. La boucle de rétroaction : produire → corriger → réentraîner

### 5.1 Le principe

La vraie force du pipeline Kraken n'est pas dans la qualité d'un modèle initial — elle est dans sa capacité à **s'améliorer de façon incrémentale** à chaque correction apportée par un expert humain. C'est ce que les praticiens appellent le workflow *human-in-the-loop* ou, dans le domaine de l'apprentissage automatique, l'*active learning* supervisé.

La boucle de rétroaction fonctionne ainsi :

```
┌─────────────────────────────────────────────────────────────┐
│  Itération N                                                │
│                                                             │
│  1. PRODUCTION                                              │
│     Kraken traite les pages non encore annotées             │
│     → Fichiers PAGE XML avec transcriptions automatiques    │
│                                                             │
│  2. CORRECTION (eScriptorium)                               │
│     Un expert corrige les transcriptions ligne par ligne    │
│     → Fichiers PAGE XML avec transcriptions corrigées       │
│     → Vérité terrain pour cette itération                   │
│                                                             │
│  3. RÉENTRAÎNEMENT (ketos)                                  │
│     ketos train sur l'accumulation de vérité terrain        │
│     → Nouveau modèle amélioré                               │
│                                                             │
│  4. ÉVALUATION (ketos test)                                 │
│     CER sur les pages de test réservées                     │
│     → Décision : continuer ou arrêter                       │
│                                                             │
│  → Itération N+1 avec le nouveau modèle                     │
└─────────────────────────────────────────────────────────────┘
```

**L'effet cumulatif :** à chaque itération, le modèle voit plus de données représentatives du corpus cible. Le CER diminue progressivement. La vitesse de correction augmente (moins d'erreurs à corriger), ce qui permet de traiter plus de pages par itération.

### 5.2 Le réentraînement avec ketos

`ketos` est l'outil en ligne de commande de Kraken pour l'entraînement et l'évaluation des modèles.

```bash
# ── Structure des données d'entraînement ──────────────────────────────────────
# data/train/
#   ├── page_001.tif      (image originale)
#   ├── page_001.xml      (PAGE XML avec transcriptions corrigées)
#   ├── page_002.tif
#   ├── page_002.xml
#   └── ...
# data/val/
#   ├── page_050.tif
#   ├── page_050.xml
#   └── ...
# data/test/             (JAMAIS utilisé en entraînement)
#   ├── page_090.tif
#   └── ...

# ── Entraînement depuis un modèle pré-existant (fine-tuning) ─────────────────
ketos train \
  --ground-truth "data/train/*.xml" \
  --evaluation-files "data/val/*.xml" \
  --load "modeles/base_medieval.mlmodel" \   # Modèle de départ
  --output "modeles/cremma_iter1.mlmodel" \  # Modèle produit
  --format xml \                              # Format des données (PAGE XML)
  --epochs 100 \
  --batch-size 16 \
  --lag 20 \                                  # Early stopping : arrêt si pas d'amélioration pendant 20 epochs
  --report-every 10                           # Rapport CER tous les 10 epochs

# ── Évaluation ────────────────────────────────────────────────────────────────
ketos test \
  --model "modeles/cremma_iter1.mlmodel" \
  --evaluation-files "data/test/*.xml" \
  --format xml

# Sortie attendue :
# Successfully loaded model cremma_iter1.mlmodel
# Evaluating on 12 lines from 3 document(s)
# CER: 0.0834 (8.3%)
# WER: 0.2156 (21.6%)

# ── Entraînement depuis zéro (si aucun modèle pré-existant adapté) ────────────
ketos train \
  --ground-truth "data/train/*.xml" \
  --evaluation-files "data/val/*.xml" \
  --output "modeles/cremma_scratch.mlmodel" \
  --format xml \
  --epochs 200 \
  --batch-size 8 \
  --report-every 20
```

**Via Python (pour plus de contrôle) :**

```python
from kraken.ketos import cli as ketos_cli
import subprocess

def reentrainer_modele_kraken(
    dossier_train: str,
    dossier_val: str,
    modele_base: str,
    modele_sortie: str,
    n_epochs: int = 100,
    batch_size: int = 16,
    lag: int = 20
) -> dict:
    """
    Lance un réentraînement Kraken et retourne les métriques finales.
    Wrapper autour de la commande `ketos train`.
    """
    cmd = [
        "ketos", "train",
        "--ground-truth", f"{dossier_train}/*.xml",
        "--evaluation-files", f"{dossier_val}/*.xml",
        "--load", modele_base,
        "--output", modele_sortie,
        "--format", "xml",
        "--epochs", str(n_epochs),
        "--batch-size", str(batch_size),
        "--lag", str(lag),
        "--report-every", "10",
    ]
    print(f"  Commande : {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout[-2000:])   # Dernières lignes du log

    if result.returncode != 0:
        print(f"  ⚠ Erreur ketos : {result.stderr[-500:]}")

    # Parser le CER final depuis la sortie
    cer_final = None
    for ligne in result.stdout.split("\n"):
        if "CER:" in ligne:
            try:
                cer_final = float(ligne.split("CER:")[1].split()[0])
            except (IndexError, ValueError):
                pass

    return {
        "modele_sortie": modele_sortie,
        "cer_val"      : cer_final,
        "log"          : result.stdout,
    }
```

### 5.3 L'active learning : quelles pages annoter en priorité

Toutes les pages ne sont pas également utiles pour améliorer le modèle. L'*active learning* vise à sélectionner les pages à annoter qui maximiseront le gain de performance à chaque itération.

**Stratégie 1 — Incertitude maximale**

Annoter les pages sur lesquelles le modèle est le moins confiant. Ces pages correspondent aux zones de l'espace des données où le modèle manque d'information.

```python
def selectionner_pages_actives(
    dossier_pages: str,
    modele_seg,
    modele_htr,
    n_a_selectionner: int = 20,
    methode: str = "incertitude"   # "incertitude" ou "diversite"
) -> list[str]:
    """
    Sélectionne les pages les plus utiles pour l'annotation humaine.

    methode="incertitude" : pages avec la confiance HTR la plus basse
    methode="diversite"   : pages les plus différentes des données déjà annotées
    """
    from pathlib import Path

    images = sorted(Path(dossier_pages).glob("*.tif"))
    scores = []

    for img_path in images:
        # Ignorer les pages déjà annotées
        xml_path = img_path.with_suffix(".xml")
        if xml_path.exists():
            continue

        image_pil = Image.open(img_path)

        # Segmenter et transcrire
        try:
            seg = segmenter_page_blla(image_pil, modele_seg)
            htr = reconnaitre_lignes_kraken(image_pil, seg, modele_htr)

            # Score d'incertitude : confiance moyenne inversée
            confidences = [r["confiance"] for r in htr if r["confiance"] > 0]
            score = 1.0 - np.mean(confidences) if confidences else 1.0

        except Exception:
            score = 1.0   # Erreur → priorité maximale

        scores.append((str(img_path), score))

    # Trier par score décroissant (plus incertain = priorité plus haute)
    scores.sort(key=lambda x: x[1], reverse=True)

    pages_selectionnees = [chemin for chemin, _ in scores[:n_a_selectionner]]
    print(f"\n  {len(pages_selectionnees)} pages sélectionnées pour annotation :")
    for chemin, score in scores[:n_a_selectionner]:
        print(f"    {Path(chemin).name} — incertitude : {score:.3f}")

    return pages_selectionnees
```

**Stratégie 2 — Diversité maximale**

Annoter des pages représentatives de la diversité du corpus (différentes mains, différents types de mise en page, différents états de conservation). Cette stratégie est complémentaire de l'incertitude — une page peut être incertaine *et* peu diverse (similaire à ce qui a déjà été annoté).

En pratique, on combine les deux : sélectionner d'abord les pages à forte incertitude, puis filtrer pour conserver la diversité maximale par clustering des features DINOv2 (vues au Jour 3, section 3.1).

### 5.4 Contribution à HTR-United

Une fois le corpus annoté et le modèle réentraîné, la communauté bénéficie d'un partage. HTR-United est le dépôt de référence pour les datasets HTR de documents patrimoniaux en français.

```bash
# Structure attendue pour un dépôt HTR-United
mon-corpus-medieval/
├── README.md              # Description du corpus, dates, sources, licence
├── data/
│   ├── train/
│   │   ├── page_001.tif
│   │   ├── page_001.xml   # PAGE XML avec transcriptions validées
│   │   └── ...
│   ├── val/
│   └── test/
├── catalog.json           # Métadonnées machine-readable (HTR-United format)
└── LICENSE                # CC-BY ou CC-BY-SA recommandé

# Dépôt sur GitHub + signalement à HTR-United
# → Indexation automatique dans le catalogue
# → Disponible pour tous les projets futurs via HTR-United
```

---

## 6. Comparaison opérationnelle : pipeline modulaire vs pipeline Kraken

### 6.1 Le tableau de décision complet

| Critère | Pipeline modulaire (SAM + TrOCR) | Pipeline Kraken intégré |
|---------|----------------------------------|------------------------|
| **Qualité HTR absolue** | Supérieure (LM implicite GPT-2) | Bonne (LSTM-CTC) |
| **Lignes courbées (parchemin)** | Limitée (boîtes rectangulaires) | Native (polygones baseline) |
| **Ordre de lecture** | Manuel ou absent | Natif (graphe topologique) |
| **Format de sortie** | JSON propriétaire | PAGE XML / ALTO (standard) |
| **Compatibilité eScriptorium** | Import manuel possible | Native |
| **Compatibilité Transkribus** | Non | Partielle |
| **Compatibilité Gallica/BnF** | Non | ALTO natif |
| **Boucle de rétroaction** | À construire manuellement | Intégrée (`ketos train`) |
| **GPU requis** | Oui (TrOCR) | Non (CPU suffisant) |
| **Vitesse (lignes/seconde, GPU)** | ~50 | ~200 |
| **Vitesse (lignes/seconde, CPU)** | ~2 | ~20 |
| **Modèles médiévaux disponibles** | Non (à fine-tuner) | Oui (HTR-United) |
| **Courbe d'apprentissage** | Élevée | Modérée |
| **Déploiement en production** | Complexe | Simple |
| **Contexte recommandé** | Recherche, cas difficiles | Production, équipes mixtes |

### 6.2 La décision en pratique

**Choisir le pipeline Kraken intégré quand :**

- L'équipe inclut des humanistes ou des philologues qui corrigeront les transcriptions dans eScriptorium.
- Le corpus est relativement homogène (même main, même siècle, même type de document).
- Les pages contiennent des lignes courbées ou des mises en page à plusieurs colonnes.
- Les résultats doivent être compatibles avec des outils de bibliothèques (Gallica, Europeana).
- Les ressources GPU sont limitées.
- On vise une amélioration progressive sur plusieurs semaines ou mois.

**Choisir le pipeline modulaire (SAM + TrOCR) quand :**

- La qualité maximale est requise sur des textes très difficiles (encre pâlie, abréviations denses, dégradations sévères).
- Le pipeline doit être hautement personnalisable et expérimentable.
- Le corpus est très diversifié (plusieurs siècles, plusieurs scripts) et chaque composant peut être adapté indépendamment.
- Les résultats sont directement transmis à un modèle NLP et n'ont pas besoin d'être visualisés dans un outil dédié.

**La combinaison des deux (recommandée pour ce projet) :**

Dans notre projet, les deux pipelines sont complémentaires :

1. **Kraken** traite la majorité du corpus en production : rapide, autonome, compatible eScriptorium, boucle de rétroaction intégrée.
2. **TrOCR fine-tuné** traite les pages difficiles identifiées par Kraken comme à faible confiance : meilleur CER, mais plus lent et plus lourd.
3. Les transcriptions des deux modèles sont **fusionnées par vote pondéré** (section 4.3) pour les pages difficiles.

---

## 7. Connexions avec le pipeline global

La section 4.2 apporte deux contributions au pipeline du Jour 4 :

**Contribution 1 — Un pipeline alternatif complet**

```
OpenCV (prétraitement, section 4.1)
    ↓
Kraken BLLA            → baselines + polygones + ordre de lecture
    ↓
Kraken HTR (LSTM-CTC)  → transcriptions + confidences
    ↓
Sérialisation PAGE XML → compatible eScriptorium, Gallica, HTR-United
    ↓
[Correction humaine dans eScriptorium]
    ↓
ketos train            → modèle amélioré → retour à l'étape 2
```

**Contribution 2 — Des composants réutilisables dans le pipeline modulaire**

- La segmentation BLLA peut remplacer la combinaison SAM + Kraken segment du pipeline modulaire — elle est plus adaptée aux lignes courbées et intègre l'ordre de lecture.
- Les fichiers PAGE XML peuvent servir de format d'échange avec le module NLP, en complément ou en remplacement du JSON.
- La boucle de rétroaction `ketos train` peut être adaptée pour réentraîner TrOCR sur les corrections faites dans eScriptorium.

---

## Glossaire des termes avancés

**Active learning**
Stratégie d'apprentissage automatique où le modèle sélectionne activement les exemples les plus utiles à annoter, plutôt que d'utiliser tous les exemples disponibles de façon indifférenciée. Réduit le coût d'annotation en maximisant le gain de performance par exemple annoté.

**ALTO (*Analyzed Layout and Text Object*)**
Format XML standard pour l'encodage des résultats de reconnaissance de documents, développé par la Bibliothèque nationale d'Autriche et maintenu par la Library of Congress. Utilisé par Gallica, British Library, Europeana.

**Ascendant (*ascender*)**
Partie d'une lettre qui s'élève au-dessus de la ligne de base. Concerne les lettres *b*, *d*, *f*, *h*, *k*, *l*, *t*. En écriture gothique, les ascendants peuvent être très allongés et décorés.

**BLLA (*Baseline Layout Analysis*)**
Module de segmentation de Kraken basé sur un réseau de neurones. Détecte les baselines et les polygones englobants de toutes les lignes d'une page, ainsi que les régions (colonnes, marges, illustrations). Entraîné sur des données de documents patrimoniaux.

**Baseline (ligne de base)**
Ligne imaginaire sur laquelle reposent les lettres d'une écriture. En typographie et en paléographie, c'est la référence géométrique principale d'une ligne de texte. Dans Kraken, elle est représentée par une polyligne (série de points) qui suit la courbure réelle de la ligne.

**Descendant (*descender*)**
Partie d'une lettre qui descend en dessous de la ligne de base. Concerne les lettres *g*, *j*, *p*, *q*, *y*, et certaines formes de *f* et *z* médiévaux.

**eScriptorium**
Plateforme web open source développée par l'Université Paris Sciences et Lettres pour l'annotation HTR collaborative. Nativement compatible avec Kraken (segmentation BLLA et reconnaissance). Permet la correction des transcriptions, la gestion de l'ordre de lecture, et l'export en PAGE XML.

**Graphe topologique**
Graphe orienté sans cycle (*DAG — Directed Acyclic Graph*) représentant un ordre partiel entre éléments. Dans Kraken, utilisé pour représenter l'ordre de lecture : chaque nœud est une ligne, chaque arête représente une relation de précédence. Le tri topologique produit une séquence de lecture valide.

**Human-in-the-loop**
Paradigme de traitement où un humain intervient à des étapes clés du processus automatique pour corriger les erreurs ou valider les résultats. Dans le contexte HTR, les corrections humaines alimentent le réentraînement du modèle.

**ketos**
Outil en ligne de commande de Kraken pour l'entraînement (`ketos train`), le test (`ketos test`), et la gestion des modèles HTR. Prend en entrée des fichiers PAGE XML ou ALTO avec images associées.

**Lettrine (initiale ornée)**
Grande lettre initiale placée en début de chapitre ou de section, souvent ornée ou enluminée. Elle peut s'étendre sur plusieurs lignes de texte. Pose un problème particulier pour la segmentation de layout et l'ordre de lecture.

**PAGE XML**
Format XML standard pour l'encodage des résultats d'analyse de documents, développé dans le projet PRIMA (PRImA Research Ltd.). Supporte nativement les baselines, les polygones non rectangulaires, et l'ordre de lecture explicite. Utilisé par eScriptorium, Transkribus, HTR-United.

**Polyligne**
Courbe géométrique définie par une séquence ordonnée de points reliés par des segments droits. La baseline d'une ligne de texte dans Kraken est une polyligne — plus précise qu'une droite pour les lignes courbées, moins coûteuse qu'une courbe paramétrique.

**Polygone englobant (*boundary polygon*)**
Polygone (possiblement non convexe) qui délimite exactement la zone de pixels appartenant à une ligne de texte dans une page. Plus précis qu'une boîte rectangulaire pour les lignes courbées ou inclinées.

**Réclame (*catchword*)**
Mot ou syllabe écrit en bas de la dernière page d'un cahier, répétant le premier mot de la page suivante. Convention médiévale pour faciliter la reliure dans le bon ordre. Constitue une ligne courte en fin de page qui doit être correctement classée dans l'ordre de lecture.

**Rubrique**
Élément textuel écrit à l'encre rouge (*rubrum*) dans un manuscrit médiéval : titre de chapitre, indication liturgique, sous-titre. La couleur distingue la rubrique du corps du texte et lui confère un statut structurel différent dans l'ordre de lecture.

**rpred**
Module Kraken pour la reconnaissance de texte sur des images de lignes avec baseline (`rpred.rpred`). Prend en entrée une image de page complète et les résultats BLLA, et produit les transcriptions ligne par ligne en utilisant les informations de baseline pour l'extraction.

**Tri topologique**
Algorithme de graphe qui produit un ordonnancement linéaire des nœuds d'un graphe orienté sans cycle (DAG) respectant toutes les relations de précédence. Utilisé par Kraken pour calculer l'ordre de lecture à partir du graphe de lignes.

---

## Bibliographie de référence

### Kraken et BLLA

- **Kiessling, B.** (2019). *Kraken: An Universal Text Recognizer for the Humanities*. Digital Humanities Conference (DH2019). — Présentation synthétique de Kraken, de BLLA et de ses applications aux manuscrits patrimoniaux.

- **Kiessling, B., Stökl Ben Ezra, D., Miller, M. T.** (2019). *BADAM: A Public Dataset for Baseline Detection in Arabic-script Manuscripts*. Document Analysis and Image Analysis Workshop (GREC/ICDAR). — L'article technique fondateur de BLLA. Présente le modèle neuronal de détection de baselines.

- **Kiessling, B.** (2022). *Kraken — Documentation officielle*. [kraken.re](https://kraken.re) — La référence complète pour l'utilisation de Kraken, ketos, et BLLA.

### Formats de sérialisation

- **Pletschacher, S., Antonacopoulos, A.** (2010). *The PAGE (Page Analysis and Ground-truth Elements) Format Framework*. ICPR 2010. — L'article fondateur de PAGE XML.

- **Library of Congress** (2004, mis à jour régulièrement). *ALTO XML Schema*. [loc.gov/standards/alto](https://loc.gov/standards/alto). — Spécification officielle du format ALTO.

- **Clausner, C., Pletschacher, S., Antonacopoulos, A.** (2011). *Aletheia — An Advanced Document Layout and Text Ground-Truthing System for Production Environments*. ICDAR 2011. — Outil de référence pour l'annotation PAGE XML, complémentaire d'eScriptorium.

### eScriptorium et workflows de production

- **Kiessling, B., Tissot, R., Stokes, P., Stökl Ben Ezra, D.** (2019). *eScriptorium: An Open Source Platform for Historical Document Analysis*. ICDAR Workshop on Open Services and Tools for Document Analysis. — Présentation d'eScriptorium, son intégration avec Kraken, et les workflows de correction HTR.

- **Stokes, P., Kiessling, B., Tissot, R., Gargon, E., Stökl Ben Ezra, D.** (2021). *The eScriptorium VRE for Manuscript Cultures*. Classics@ Journal, 18. — Description des cas d'usage académiques d'eScriptorium pour les manuscrits médiévaux.

### Ordre de lecture et analyse de layout

- **Barman, R., Ares Oliveira, S., Kaplan, F.** (2020). *A Versatile Information Extraction Framework for Historical Document Analysis*. Frontiers in Research Metrics and Analytics. — Sur l'analyse de layout et l'ordre de lecture dans les documents patrimoniaux.

- **Studer, L., Alberti, M., Pondenkandath, V., Goktepe, P., Kolonko, T., Fischer, A., Liwicki, M., Savoy, J.** (2019). *A Comprehensive Study of Imagenet Pre-Training for Historical Document Image Analysis*. ICDAR 2019. — Comparaison des approches pour la segmentation de layout des documents historiques.

### Active learning et boucle de rétroaction

- **Settles, B.** (2012). *Active Learning*. Synthesis Lectures on Artificial Intelligence and Machine Learning. Morgan & Claypool. — La référence de référence sur l'active learning. Chapitres 3 et 4 sur les stratégies d'incertitude et de diversité.

- **Diem, M. et al.** (2019). *Text Block Segmentation Using a Document Specific Model*. ICDAR 2019. — Sur l'adaptation incrémentale des modèles de segmentation à des corpus spécifiques.

### HTR-United et données ouvertes

- **Pinche, A., Camps, J.-B., Clérice, T.** (2022). *HTR-United: Mutualisons la vérité de terrain !* DHd 2022. [hal-03693079] — Présentation du projet HTR-United et de ses conventions.

- **Constum, T., Chagué, A., Gille-Levenson, M., Muller, A., Pinche, A., Romary, L., Schmitt, M., Terriel, L.** (2022). *Toward a Transcription Norm for Training Historical Handwritten Text Recognition Models*. [inria.hal.science/hal-05429033v1] — Les conventions de transcription recommandées pour créer des datasets HTR compatibles HTR-United.

---

*Support de cours rédigé pour le module Computer Vision · Promotion MD5 · Mai 2026*
*Ce document accompagne la séance 4.2 du Jour 4. Il suppose la lecture de la section 4.1 (prétraitement).*
*Durée estimée de lecture : 80–90 minutes.*
