# Cours 3.3 — SAM : Segment Anything Model

**Module 3 · Computer Vision appliquée aux manuscrits médiévaux · MD5**

---

> *« Segmenter une image, c'est répondre à la question : quelle région appartient à quel objet ? SAM répond à cette question pour n'importe quel objet, sur n'importe quelle image, avec n'importe quel type d'indication — et sans avoir jamais vu le type d'objet en question. »*

---

## Introduction : le problème de la segmentation de layout

Avant de transcrire une ligne de texte, un système HTR doit savoir *où* elle se trouve. Cette étape — la **segmentation de layout** — est souvent la principale source d'erreurs en cascade dans un pipeline HTR.

Pour les manuscrits médiévaux, la difficulté est aiguë. La mise en page n'est pas standardisée : un codex du XIIIe siècle peut avoir deux colonnes de texte, une enluminure pleine largeur, des rubriques intercalées et des notes marginales de plusieurs mains — sur la même page. Les frontières entre régions sont visuellement ambiguës. La dégradation physique (taches, transparence du parchemin, déchirures) brouille les transitions.

Les approches classiques (analyse de projections, modèles supervisés de détection) nécessitent des données annotées spécifiques au type de document. Pour la diversité des mises en page médiévales, un modèle entraîné sur des chartes du XIIIe siècle ne fonctionnera pas sur des romans enluminés du XVe.

**SAM** (*Segment Anything Model*, Kirillov et al., 2023) propose une approche radicalement différente : un modèle de segmentation universel, entraîné sur un milliard de masques, capable de segmenter n'importe quelle région cohérente d'une image à partir d'une indication minimale — sans réentraînement.

---

## 1. Le paradigme de la segmentation promptable

### 1.1 Des modèles fermés aux modèles fondationnels

Les modèles de segmentation classiques (U-Net, Mask R-CNN, DeepLab) définissent à l'entraînement un ensemble fixe de classes sémantiques. Toute nouvelle classe nécessite de nouvelles données annotées et un réentraînement.

SAM opère selon le paradigme des **modèles fondationnels** : un seul modèle pré-entraîné à très grande échelle, capable de généraliser à des tâches non vues via des prompts. La capacité clé est la **segmentation promptable** : l'utilisateur fournit une indication — un point, une boîte, un masque partiel — et SAM produit le masque correspondant.

SAM ne nomme pas les régions. Il dit : « voici le contour cohérent de la région que vous me désignez » — pas « ceci est du texte ». L'étiquetage sémantique est une tâche distincte, confiée à CLIP ou à des heuristiques en aval.

### 1.2 Définition formelle de la tâche

SAM est entraîné à résoudre la tâche suivante :

> Étant donné une image $I$ et un prompt $p$ (point, boîte, masque, ou combinaison), produire un masque de segmentation valide $M \subseteq \{0,1\}^{H \times W}$ englobant la région désignée par $p$.

La notion de « masque valide » est délibérément sous-spécifiée : pour un point au centre d'une enluminure, trois masques sont tous valides — la figure centrale, le personnage entier, ou l'enluminure entière. Pour gérer cette ambiguïté, SAM produit **trois masques candidats** à des niveaux de granularité différents, accompagnés de scores de confiance estimés.

---

## 2. Le dataset SA-1B : un milliard de masques

### 2.1 Le moteur de données en boucle fermée

Annoter manuellement un milliard de masques est impossible — un annotateur expert produit 30 à 60 masques complexes par heure. La solution de Kirillov et al. est un **moteur de données** (*data engine*) : un processus itératif où le modèle en cours d'entraînement assiste les annotateurs, réduisant le coût d'annotation jusqu'à permettre la génération entièrement automatique.

**Phase 1 — Annotation assistée (4,3 M masques)**

Un modèle SAM préliminaire suggère des masques candidats. Les annotateurs les corrigent ou les rejettent. Vitesse : 30 masques/heure contre 6 en annotation manuelle pure — accélération ×5.

**Phase 2 — Annotation semi-automatique (5,9 M masques supplémentaires)**

Le modèle réentraîné couvre certaines régions automatiquement. Les annotateurs se concentrent sur les régions non encore traitées.

**Phase 3 — Génération entièrement automatique (~1,1 Md de masques)**

Le modèle mature génère des masques sans intervention humaine via une grille dense de points prompts, filtrés par qualité et dédupliqués.

| Propriété | Valeur |
|-----------|--------|
| Images | 11 millions |
| Masques totaux | ~1,1 milliard |
| Masques/image (moy.) | ~100 |
| Résolution moyenne | 3 300 × 4 950 pixels |
| Licence | Recherche uniquement |

---

## 3. Architecture de SAM

SAM est composé de trois modules distincts, conçus pour des rôles et des cadences d'exécution différentes.

```
Image I ──→ [ Encodeur d'image ] ──→ E_I ∈ ℝ^{64×64×256}  (une fois par image)
                                              ↓
Prompt p ──→ [ Encodeur de prompt ] ──→ E_p               (une fois par prompt)
                                              ↓
                               [ Décodeur de masque ] ──→ 3 Masques + 3 Scores IoU
```

### 3.1 L'encodeur d'image : ViT-H pré-entraîné par MAE

L'encodeur est un **ViT-H/16** (632 M paramètres) pré-entraîné par **MAE** (*Masked Autoencoders*).

- Patches 16 × 16 pixels.
- Résolution d'entrée fixe : 1024 × 1024 pixels.
- Attention fenêtrée (14 × 14 patches) + 4 couches d'attention globale.
- Sortie : feature map $E_I \in \mathbb{R}^{64 \times 64 \times 256}$ après une tête de réduction de canal.

**Propriété clé :** l'encodeur s'exécute **une seule fois par image** et sa sortie est mise en cache. Les requêtes interactives successives (différents prompts sur la même image) ne ré-exécutent que l'encodeur de prompt et le décodeur — d'où la latence interactive < 50 ms par masque.

Le pré-entraînement MAE confère des représentations robustes aux régions manquantes ou dégradées — utile pour les scans de parchemins abîmés.

```python
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
from PIL import Image
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Trois variantes selon les ressources disponibles
# vit_h : 632 M params — meilleure qualité (recommandé)
# vit_l : 308 M params — bon équilibre
# vit_b :  91 M params — plus rapide, légèrement moins précis
sam = sam_model_registry["vit_h"](
    checkpoint="checkpoints/sam_vit_h_4b8939.pth"
).to(DEVICE)

predictor = SamPredictor(sam)

# Encodage de l'image — exécuté une seule fois, résultat mis en cache
image_np = np.array(Image.open("scan_manuscrit.tif").convert("RGB"))
predictor.set_image(image_np)

print(f"Features image : {predictor.features.shape}")
# → torch.Size([1, 256, 64, 64])
```

### 3.2 L'encodeur de prompt

SAM supporte deux familles de prompts.

**Prompts épars : points et boîtes**

Les points $(x, y)$ de type $t$ (positif, négatif, coin) sont représentés comme :

$$e_{\text{point}} = \text{PE}_{2D}(x, y) + e_t^{\text{type}}$$

où $\text{PE}_{2D}$ est un encodage sinusoïdal 2D et $e_t^{\text{type}} \in \mathbb{R}^{256}$ un embedding appris par type.

Une boîte $(x_1, y_1, x_2, y_2)$ est encodée comme deux points : coin haut-gauche et coin bas-droite, chacun avec leur type spécifique.

**Prompts denses : masques**

Un masque en entrée (résolution $256 \times 256$) est encodé par un petit réseau convolutif (4 couches) vers $\mathbb{R}^{64 \times 64 \times 256}$ — même résolution spatiale que $E_I$. Les features image et le masque encodé sont additionnés avant le décodeur.

### 3.3 Le décodeur de masque

Le décodeur est **léger** (< 4 M paramètres) et **rapide** (< 50 ms). Il opère sur deux ensembles de tokens mis à jour alternativement par self-attention et cross-attention :

- **Self-attention** entre les 4 tokens de sortie et les tokens de prompt.
- **Cross-attention** des tokens de sortie vers les features image $E_I$ (les tokens « lisent » l'image).
- **Cross-attention** des features image vers les tokens de sortie (l'image se conditionne sur le prompt).

Après deux itérations, les tokens de sortie génèrent :

- **3 masques** : chaque token de masque est multiplié avec des features image haute résolution ($256 \times 256$, obtenues par upsampling ×4) puis upsamplé à la résolution d'entrée.
- **3 scores IoU** : le token dédié prédit la qualité estimée de chaque masque candidat.

**Pourquoi trois masques ?** Pour gérer l'ambiguïté de granularité : fin (petite région cohérente), médium (région intermédiaire), grossier (région entière). L'utilisateur ou l'algorithme choisit selon le score IoU ou une heuristique de taille.

---

## 4. Les types de prompts en pratique

### 4.1 Point prompt

```python
def segmenter_par_point(
    predictor: SamPredictor,
    x: int, y: int,
    label: int = 1       # 1=positif, 0=négatif
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Segmente la région désignée par un point.
    Retourne (masks, scores, logits) — 3 masques candidats.
    """
    masks, scores, logits = predictor.predict(
        point_coords=np.array([[x, y]]),
        point_labels=np.array([label]),
        multimask_output=True
    )
    return masks, scores, logits


def segmenter_avec_points_positifs_negatifs(
    predictor: SamPredictor,
    points_pos: list[tuple[int, int]],
    points_neg: list[tuple[int, int]] = None
) -> tuple[np.ndarray, float]:
    """
    Combine points positifs (inclure) et négatifs (exclure).

    Cas d'usage : point positif sur le corps du texte +
    points négatifs sur les lettrines adjacentes
    → masque du texte principal sans les ornements.
    """
    coords = list(points_pos)
    labels = [1] * len(points_pos)
    if points_neg:
        coords += list(points_neg)
        labels += [0] * len(points_neg)

    masks, scores, _ = predictor.predict(
        point_coords=np.array(coords),
        point_labels=np.array(labels),
        multimask_output=True
    )
    best = np.argmax(scores)
    return masks[best], scores[best]
```

### 4.2 Box prompt

```python
def segmenter_par_boite(
    predictor: SamPredictor,
    x1: int, y1: int, x2: int, y2: int
) -> tuple[np.ndarray, float]:
    """
    Segmente la région contenue dans la boîte englobante.
    La boîte lève l'ambiguïté — SAM produit un seul masque précis.
    """
    masks, scores, _ = predictor.predict(
        box=np.array([x1, y1, x2, y2]),
        multimask_output=False
    )
    return masks[0], scores[0]
```

La boîte est le prompt le plus utile quand on dispose déjà d'une détection approximative — par exemple, les boîtes d'un détecteur de layout. SAM affine ces boîtes en masques précis, y compris pour des formes non rectangulaires (enluminures irrégulières, notes en forme de main pointante).

### 4.3 Raffinement itératif par masque prompt

```python
def raffiner_masque_iteratif(
    predictor: SamPredictor,
    logits_precedents: np.ndarray,
    scores_precedents: np.ndarray,
    point_correction_x: int,
    point_correction_y: int,
    label_correction: int = 1
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Affine un masque précédent avec un point de correction.
    Les logits du masque précédent servent de prompt dense — contexte fort.

    Cas d'usage : le premier masque inclut une zone à exclure
    → ajouter un point négatif avec le masque précédent comme contexte.
    """
    best_idx = np.argmax(scores_precedents)
    masks, scores, logits = predictor.predict(
        point_coords=np.array([[point_correction_x, point_correction_y]]),
        point_labels=np.array([label_correction]),
        mask_input=logits_precedents[best_idx:best_idx+1],
        multimask_output=True
    )
    best = np.argmax(scores)
    return masks[best], scores[best], logits
```

Le raffinement itératif converge en 2 à 3 points de correction pour la majorité des régions complexes.

---

## 5. Génération automatique de masques

### 5.1 Principe

Pour une page dont on ne connaît pas la structure, SAM peut générer tous les masques de régions cohérentes sans prompt humain, via une **grille dense de points** comme prompts simultanés.

```python
from segment_anything import SamAutomaticMaskGenerator

def generer_masques_automatiques(
    image: np.ndarray,
    sam_model,
    points_par_cote: int = 32,
    pred_iou_thresh: float = 0.88,
    stability_score_thresh: float = 0.95,
    min_mask_region_area: int = 500
) -> list[dict]:
    """
    Génère automatiquement tous les masques de régions cohérentes.

    Chaque masque est un dict contenant :
    - 'segmentation'     : (H, W) booléen
    - 'area'             : surface en pixels
    - 'bbox'             : [x, y, w, h]
    - 'predicted_iou'    : score IoU estimé
    - 'stability_score'  : robustesse au seuil
    - 'point_coords'     : point prompt générateur
    """
    generator = SamAutomaticMaskGenerator(
        model=sam_model,
        points_per_side=points_par_cote,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        crop_n_layers=1,
        min_mask_region_area=min_mask_region_area,
    )
    masks = generator.generate(image)
    masks = sorted(masks, key=lambda m: m["area"], reverse=True)
    print(f"  {len(masks)} régions détectées")
    return masks
```

### 5.2 Visualisation des masques générés

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def visualiser_masques(
    image: np.ndarray,
    masques: list[dict],
    titre: str = "Segmentation automatique SAM",
    nom_figure: str = "sam_auto"
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle(titre, fontsize=13, fontweight="bold")

    axes[0].imshow(image)
    axes[0].set_title("Image originale", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(image)
    np.random.seed(42)
    for masque in masques:
        c = np.random.rand(3)
        overlay = np.zeros((*image.shape[:2], 4))
        overlay[masque["segmentation"]] = [*c, 0.45]
        axes[1].imshow(overlay)
        x, y, w, h = masque["bbox"]
        axes[1].add_patch(mpatches.Rectangle(
            (x, y), w, h, linewidth=1,
            edgecolor=c, facecolor="none", alpha=0.8
        ))

    axes[1].set_title(
        f"{len(masques)} régions — couleur = région, contour = boîte", fontsize=11
    )
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(f"figures/{nom_figure}_segmentation.png", dpi=150, bbox_inches="tight")
    plt.show()
```

### 5.3 Post-traitement : filtrage et classification heuristique

SAM génère souvent des masques redondants ou non pertinents. Un post-traitement est nécessaire.

```python
def filtrer_et_classifier_masques(
    masques: list[dict],
    image: np.ndarray,
    surface_min_ratio: float = 0.001,
    surface_max_ratio: float = 0.85,
    score_iou_min: float = 0.85
) -> dict[str, list[dict]]:
    """
    Filtre par taille/qualité et classe heuristiquement en 4 catégories.

    Note : ces heuristiques sont approximatives.
    Une classification précise requiert CLIP ou un classifieur dédié.
    """
    H, W = image.shape[:2]
    surface_page = H * W

    filtres = [
        m for m in masques
        if (surface_min_ratio * surface_page
            <= m["area"]
            <= surface_max_ratio * surface_page)
        and m["predicted_iou"] >= score_iou_min
    ]
    print(f"  Après filtrage : {len(filtres)}/{len(masques)} masques conservés")

    categories = {"texte": [], "illustration": [], "marge": [], "autre": []}

    for m in filtres:
        x, y, w, h = m["bbox"]
        region_gris = np.mean(image[y:y+h, x:x+w], axis=2)
        ratio_aspect  = w / (h + 1e-8)
        densite_sombre = np.mean(region_gris < 100)
        en_marge = (x < 0.1*W or x+w > 0.9*W
                    or y < 0.05*H or y+h > 0.95*H)

        if en_marge and m["area"] < 0.05 * surface_page:
            categories["marge"].append(m)
        elif densite_sombre > 0.15 and 0.3 < ratio_aspect < 10:
            categories["texte"].append(m)
        elif densite_sombre < 0.08 and m["area"] > 0.02 * surface_page:
            categories["illustration"].append(m)
        else:
            categories["autre"].append(m)

    for cat, ms in categories.items():
        print(f"  {cat:15s} : {len(ms)} région(s)")

    return categories
```

---

## 6. Pipeline intégré SAM + CLIP

Combiner SAM (segmente sans classifier) et CLIP (classifie sans segmenter) donne un pipeline de layout zero-shot complet.

```python
from transformers import CLIPProcessor, CLIPModel
import torch, json
from pathlib import Path

CLASSES_LAYOUT = [
    "a block of handwritten text in a medieval manuscript",
    "an illuminated miniature or decorative illustration",
    "a large decorative initial letter or lettrine",
    "a rubric or title written in red ink",
    "a marginal annotation or note",
    "a blank or damaged area of the page",
]

def pipeline_layout_sam_clip(
    chemin_scan: str,
    sam_predictor: SamPredictor,
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    device: str = "cuda"
) -> list[dict]:
    """
    SAM segmente, CLIP classifie chaque région.
    Retourne la liste des régions enrichies de leur catégorie CLIP.
    """
    image_pil = Image.open(chemin_scan).convert("RGB")
    image_np  = np.array(image_pil)
    H, W = image_np.shape[:2]
    surface_page = H * W

    # ── SAM : génération automatique ──────────────────────────────────────────
    sam_predictor.set_image(image_np)
    generator = SamAutomaticMaskGenerator(
        sam_predictor.model,
        points_per_side=32, pred_iou_thresh=0.86,
        stability_score_thresh=0.92, min_mask_region_area=800
    )
    masques = generator.generate(image_np)
    masques = [
        m for m in masques
        if 0.002 * surface_page <= m["area"] <= 0.80 * surface_page
        and m["predicted_iou"] >= 0.85
    ]

    # ── CLIP : classification de chaque région ────────────────────────────────
    clip_model.eval()
    resultats = []

    for masque in masques:
        x, y, w, h = [int(v) for v in masque["bbox"]]
        pad = 5
        region = image_pil.crop((
            max(0, x-pad), max(0, y-pad),
            min(W, x+w+pad), min(H, y+h+pad)
        ))
        inputs = clip_processor(
            text=CLASSES_LAYOUT, images=region,
            return_tensors="pt", padding=True
        ).to(device)
        with torch.no_grad():
            logits = clip_model(**inputs).logits_per_image[0]
            probas = torch.softmax(logits * 100, dim=0).cpu().numpy()

        idx_max = int(probas.argmax())
        resultats.append({
            **masque,
            "categorie_clip" : CLASSES_LAYOUT[idx_max],
            "confiance_clip" : float(probas[idx_max]),
        })

    resultats.sort(key=lambda r: r["area"], reverse=True)
    return resultats


def exporter_layout_json(
    resultats: list[dict],
    chemin_scan: str,
    chemin_sortie: str
) -> None:
    """Export au format JSON du data contract (Jour 4)."""
    TYPE_COURT = {
        "handwritten text": "text",
        "illustration": "image", "miniature": "image",
        "initial": "lettrine", "lettrine": "lettrine",
        "rubric": "rubric", "red": "rubric",
        "marginal": "margin",
    }

    regions = []
    for i, r in enumerate(resultats):
        cat = r["categorie_clip"]
        type_court = next(
            (v for k, v in TYPE_COURT.items() if k in cat), "other"
        )
        regions.append({
            "region_id"     : f"r{i:03d}",
            "type"          : type_court,
            "type_detail"   : cat,
            "confiance_type": round(r["confiance_clip"], 3),
            "bbox"          : {
                "x": int(r["bbox"][0]), "y": int(r["bbox"][1]),
                "w": int(r["bbox"][2]), "h": int(r["bbox"][3])
            },
            "area"          : int(r["area"]),
            "iou_estime"    : round(r["predicted_iou"], 3),
        })

    sortie = {
        "page_id" : Path(chemin_scan).stem,
        "pipeline": "SAM-ViT-H + CLIP-ViT-L/14",
        "regions" : regions,
    }
    with open(chemin_sortie, "w", encoding="utf-8") as f:
        json.dump(sortie, f, ensure_ascii=False, indent=2)
    print(f"  Layout exporté : {chemin_sortie} ({len(regions)} régions)")
```

---

## 7. SAM 2 : améliorations et extension vidéo

### 7.1 Motivations

SAM 2 (Ravi et al., 2024) étend SAM à la **segmentation vidéo** : propager un masque initial à travers une séquence de frames. Ce qui nous importe davantage : les améliorations architecturales bénéficient aussi à la segmentation d'images statiques.

### 7.2 L'encodeur Hiera

SAM 2 remplace le ViT-H par **Hiera** (*Hierarchical Vision Encoder*), une architecture hiérarchique masquable beaucoup plus rapide :

| Modèle | Paramètres | Images/s (A100) | AP COCO |
|--------|-----------|-----------------|---------|
| SAM ViT-H | 632 M | ~6 | 49,1 |
| SAM 2 Hiera-L | 224 M | ~24 | 51,5 |
| SAM 2 Hiera-B+ | 80 M | ~76 | 49,2 |

SAM 2 est 4× plus rapide avec une qualité supérieure — ce qui change concrètement la praticabilité pour traiter des corpus de milliers de pages.

```python
# pip install git+https://github.com/facebookresearch/segment-anything-2.git
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

sam2 = build_sam2(
    "sam2_hiera_large.yaml",
    "checkpoints/sam2_hiera_large.pt",
    device=DEVICE
)
predictor_sam2 = SAM2ImagePredictor(sam2)

# L'API est identique à SAM 1 pour les images statiques
predictor_sam2.set_image(image_np)
masks, scores, logits = predictor_sam2.predict(
    point_coords=np.array([[500, 300]]),
    point_labels=np.array([1]),
    multimask_output=True
)
```

### 7.3 Mise en cache des features pour les grands corpus

L'encodeur de SAM est lourd mais s'exécute une seule fois par image. Pour un corpus de milliers de pages, mettre les features en cache sur disque est essentiel.

```python
import pickle
from pathlib import Path

def encoder_et_cacher(
    predictor: SamPredictor,
    chemin_image: str,
    dossier_cache: str = "cache_sam"
) -> None:
    """
    Encode une image et sauvegarde les features.
    Les appels ultérieurs chargent depuis le cache.
    """
    Path(dossier_cache).mkdir(exist_ok=True)
    cache_path = Path(dossier_cache) / (Path(chemin_image).stem + ".pkl")

    if cache_path.exists():
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        predictor.features       = cached["features"].to(DEVICE)
        predictor.original_size  = cached["original_size"]
        predictor.input_size     = cached["input_size"]
        predictor.is_image_set   = True
    else:
        image = np.array(Image.open(chemin_image).convert("RGB"))
        predictor.set_image(image)
        with open(cache_path, "wb") as f:
            pickle.dump({
                "features"     : predictor.features.cpu(),
                "original_size": predictor.original_size,
                "input_size"   : predictor.input_size,
            }, f)
```

---

## 8. Limites de SAM pour les manuscrits médiévaux

**SAM ne classifie pas.**
C'est la limite fondamentale : SAM produit des masques géométriques sans étiquettes sémantiques. La classification en aval (CLIP ou heuristiques) introduit ses propres erreurs.

**La segmentation ligne par ligne n'est pas le point fort de SAM.**
SAM excelle sur les *objets visuellement cohérents* délimités par des contrastes. Les lignes de texte manuscrit — discontinues, de faible hauteur, avec des frontières inter-lignes peu marquées — correspondent mal à ce modèle. **Kraken reste l'outil de référence pour la segmentation de lignes** ; SAM opère à la granularité des blocs et colonnes.

**Les documents très dégradés.**
Sur un parchemin avec forte transparence ou grandes taches d'humidité, SAM segmente parfois les artefacts physiques comme des « régions » — perturbant la segmentation de layout. Un prétraitement par binarisation (Sauvola) atténue ce problème.

**Coût computationnel de l'encodeur ViT-H.**
SAM 1 : ~5 secondes par image sur GPU A100, plusieurs minutes sur CPU. SAM 2 avec Hiera : ~0,5 seconde par image. Sans GPU, le traitement d'un corpus de milliers de pages nécessite la mise en cache et un plan de batch.

---

## 9. Connexions avec le pipeline global

SAM occupe l'**étape 2** du pipeline, entre le prétraitement et la segmentation de lignes :

```
OpenCV (prétraitement)
    ↓
SAM — segmentation de layout     ← Cette section
    ↓
Kraken — segmentation de lignes  ← Section 3.4 (TrOCR)
    ↓
TrOCR — HTR ligne par ligne
    ↓
CLIP — description des illustrations
    ↓
Export JSON
```

**Apport de SAM :** localiser les blocs de texte, les illustrations et les marges zero-shot, sans entraînement sur des données médiévales. Passer à Kraken uniquement les régions textuelles améliore significativement la qualité de la segmentation de lignes.

**Ce que SAM ne fait pas :** lire le texte, classifier les régions de façon fiable sans aide, descendre au niveau des lignes. CLIP et Kraken prennent le relais.

---

## Glossaire des termes avancés

**Ambiguïté de prompt**
Propriété d'un prompt pouvant correspondre à plusieurs segmentations valides à différentes granularités. SAM la gère en produisant 3 masques candidats (fin, médium, grossier) avec scores de confiance.

**Data engine (moteur de données)**
Pipeline automatisé de collecte et d'annotation où le modèle en cours d'entraînement participe à la génération des données. Permet de bootstrapper des datasets à grande échelle. Utilisé dans les trois phases de SA-1B.

**Hiera (*Hierarchical Vision Encoder*)**
Encodeur visuel hiérarchique de SAM 2. Produit des feature maps multi-échelles (compatible FPN) et est masquable (compatible MAE). 4× plus rapide que ViT-H avec une qualité supérieure.

**IoU (*Intersection over Union*)**
Métrique de qualité des masques : $\text{IoU} = |M_{\text{pred}} \cap M_{\text{gt}}| / |M_{\text{pred}} \cup M_{\text{gt}}|$. SAM prédit un score IoU estimé pour chaque masque candidat, sans accès à la vérité terrain.

**MAE (*Masked Autoencoder*)**
Méthode de pré-entraînement auto-supervisé : masquer 75% des patches d'image et reconstruire les pixels manquants. Confère des représentations robustes aux régions dégradées. Pré-entraînement de l'encodeur ViT-H de SAM.

**NMS (*Non-Maximum Suppression*)**
Algorithme éliminant les détections redondantes : parmi des masques se chevauchant (IoU > seuil), seul le plus confiant est conservé. Utilisé dans la génération automatique de masques SAM.

**Prompt dense**
Prompt constitué d'un masque (logits d'une prédiction précédente). Permet le raffinement itératif : les logits servent de contexte fort pour affiner la segmentation.

**Prompt éparse**
Prompt constitué de points (coordonnées + polarité) ou d'une boîte englobante. Chaque élément est encodé comme embedding positionnel + embedding de type.

**RLE (*Run-Length Encoding*)**
Format de compression pour les masques binaires. Encode les séquences de 0 et de 1 consécutifs par leur longueur. Utilisé par COCO et SAM pour stocker les masques efficacement.

**SA-1B**
Dataset de 1,1 milliard de masques sur 11 millions d'images, généré par le moteur de données de SAM. Le plus grand dataset de segmentation d'instances existant. Licence de recherche uniquement.

**Segmentation d'instance**
Chaque instance individuelle d'un objet reçoit son propre masque (par opposition à la segmentation sémantique où tous les pixels d'une même classe partagent le même label). SAM produit de la segmentation d'instance.

**Segmentation promptable**
Paradigme où le modèle reçoit une indication (prompt) sur la région à segmenter. Analogue à la génération de texte conditionnée par un prompt dans les LLM.

**Stability score (score de stabilité)**
Fraction de pixels du masque qui restent dans le masque quand le seuil de binarisation des logits varie de ±0,5. Un score élevé indique un masque robuste aux petites variations de paramètre.

---

## Bibliographie de référence

### SAM et SAM 2

- **Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., … Girshick, R.** (2023). *Segment Anything*. ICCV 2023. [arXiv:2304.02643] — L'article fondateur. Sections 3 (tâche), 4 (modèle) et 5 (moteur de données) à lire impérativement.

- **Ravi, N. et al.** (2024). *SAM 2: Segment Anything in Images and Videos*. [arXiv:2408.00714] — SAM 2. Description de Hiera, de la mémoire de streaming et des améliorations sur les images statiques.

- **Ryali, C. et al.** (2023). *Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles*. ICML 2023. [arXiv:2306.00989] — L'encodeur d'image de SAM 2.

### Architectures de segmentation de référence

- **Ronneberger, O., Fischer, P., Brox, T.** (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI 2015. [arXiv:1505.04597]

- **He, K., Gkioxari, G., Dollár, P., Girshick, R.** (2017). *Mask R-CNN*. ICCV 2017. [arXiv:1703.06870] — Référence pour la segmentation d'instance supervisée.

- **Cheng, B. et al.** (2022). *Masked-attention Mask Transformer for Universal Image Segmentation (Mask2Former)*. CVPR 2022. [arXiv:2112.01527] — État de l'art supervisé au moment de SAM.

### Pré-entraînement MAE

- **He, K., Chen, X., Xie, S., Li, Y., Dollár, P., Girshick, R.** (2022). *Masked Autoencoders Are Scalable Vision Learners*. CVPR 2022. [arXiv:2111.06377]

### Applications aux documents et manuscrits

- **Shen, Z. et al.** (2021). *LayoutParser: A Unified Toolkit for Deep Learning Based Document Image Analysis*. ICDAR 2021. [arXiv:2103.15348] — Toolkit de référence pour l'analyse de layout. Compatible avec SAM.

- **Kiessling, B.** (2019). *Kraken: An Universal Text Recognizer for the Humanities*. DH 2019. — Kraken pour la segmentation de lignes : alternative spécialisée à SAM pour la granularité ligne.

- **Oliveira, S. A. et al.** (2021). *dhSegment-T: Contextualized Historical Document Segmentation*. ICDAR 2021. [arXiv:2107.08504] — Modèle supervisé spécialisé pour les documents historiques. Baseline comparatif pour SAM.

- **Clausner, C., Pletschacher, S., Antonacopoulos, A.** (2011). *Aletheia — An Advanced Document Layout and Text Ground-Truthing System*. ICDAR 2011. — Référence pour l'évaluation de la segmentation de layout sur documents patrimoniaux.

---

*Support de cours rédigé pour le module Computer Vision · Promotion MD5 · Mai 2026*
*Ce document accompagne la séance 3.3 du Jour 3. Il suppose la lecture des sections 3.1 (DINO) et 3.2 (CLIP).*
*Durée estimée de lecture : 60–75 minutes.*
