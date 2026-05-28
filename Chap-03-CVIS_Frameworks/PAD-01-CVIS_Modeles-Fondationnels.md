# Cours 3.0 — Introduction au Jour 3 : des modèles spécialisés aux modèles fondationnels

**Module 3 · Computer Vision appliquée aux manuscrits médiévaux · MD5**
**Section introductive au Jour 3**

---

> *« Pendant vingt ans, chaque problème de vision avait son modèle. Classifier → un réseau de classification. Détecter → un détecteur. Segmenter → un segmenteur. Transcrire → un modèle HTR. Puis quelque chose a changé : des modèles entraînés à une échelle massive sur des données génériques ont commencé à résoudre des dizaines de problèmes sans avoir jamais été entraînés sur aucun d'eux. C'est ce changement que ce module explore. »*

---

## Introduction : pourquoi un module entier sur quatre outils ?

Le Jour 2 a posé les fondements architecturaux : les CNN et leurs biais inductifs, les Transformers et le mécanisme d'attention, U-Net et DeepLab pour la segmentation. Ces architectures constituent le *vocabulaire* de la vision profonde.

Le Jour 3 introduit une *grammaire* différente — la façon dont ce vocabulaire s'assemble à très grande échelle pour produire des modèles d'un type nouveau : **DINO**, **CLIP**, **SAM** et **TrOCR**.

Ces quatre outils ne sont pas de simples améliorations incrémentales des architectures du Jour 2. Ils représentent un **changement de paradigme** : le passage des modèles *spécialisés* (un modèle par tâche, entraîné sur des données de cette tâche) aux modèles *fondationnels* (un modèle général, entraîné à très grande échelle, adaptable à de nombreuses tâches).

Cette section introductive répond à trois questions avant d'entrer dans le détail de chaque outil :
1. **Pourquoi** ce changement de paradigme s'est-il produit ?
2. **Qu'est-ce que** ces quatre outils ont en commun ?
3. **En quoi** chacun est-il unique, et comment s'articulent-ils dans notre pipeline ?

---

## 1. Les limites du paradigme spécialisé : pourquoi changer ?

### 1.1 Le goulot d'étranglement de l'annotation

Le paradigme de l'apprentissage supervisé a un prérequis implacable : des données annotées. Pour chaque tâche — classer, détecter, segmenter, transcrire — il faut des milliers ou des dizaines de milliers d'exemples étiquetés *spécifiquement pour cette tâche*, dans le *domaine cible*.

Pour les manuscrits médiévaux, ce prérequis est rédhibitoire :

- Annoter une page de manuscrit pour l'HTR coûte 1 à 2 heures de travail expert.
- Annoter les régions de layout (colonne, illustration, rubrique, lettrine) sur un ensemble diversifié de manuscrits requiert des compétences paléographiques que seule une poignée de spécialistes possèdent.
- Entraîner un modèle de classification d'écritures médiévales (carolingienne, gothique, bâtarde…) nécessite un corpus varié avec des étiquettes que les paléographes eux-mêmes débattraient.

Le résultat pratique : des modèles performants sur le corpus d'entraînement mais fragiles dès qu'on les applique à un nouveau scriptorium, un nouveau siècle, ou un nouveau type de document. C'est la fragilité structurelle évoquée en section 5.2.

### 1.2 La leçon du NLP : scale and self-supervision

La rupture a d'abord eu lieu dans le traitement du langage naturel. En 2018–2020, des modèles comme BERT et GPT ont montré qu'un Transformer entraîné sur des quantités massives de texte brut (sans annotation) développait des représentations linguistiques si riches qu'il suffisait d'un fine-tuning léger pour les adapter à n'importe quelle tâche de NLP.

La clé : ces modèles n'ont pas besoin de données *étiquetées* — ils apprennent à partir du texte brut lui-même, en résolvant des tâches prétextes (prédire le mot masqué, prédire le mot suivant). L'annotation humaine n'est plus le goulot d'étranglement.

**La vision profonde a tiré la même leçon à partir de 2020–2021**, en construisant des modèles de grande taille entraînés sur des données massives et hétérogènes, sans ou avec peu d'annotations humaines. DINO, CLIP, SAM et TrOCR sont quatre manifestations différentes de cette approche.

### 1.3 Qu'est-ce qu'un modèle fondationnel ?

Le terme *foundation model* (Bommasani et al., 2021) désigne un modèle entraîné à très grande échelle sur des données larges et diversifiées, capable d'être adapté (*fine-tuned* ou utilisé directement) à un large éventail de tâches en aval.

Trois propriétés caractérisent un modèle fondationnel :

**Émergence :** le modèle développe des capacités que personne n'a explicitement entraînées. DINO produit une segmentation sémantique sans avoir jamais vu une image annotée pour la segmentation. CLIP classe des images dans des catégories qui n'existaient pas lors de son entraînement. Ces capacités *émergent* de l'entraînement à grande échelle.

**Homogénéisation :** un seul modèle peut servir de point de départ pour des dizaines de tâches différentes. Au lieu de former une équipe d'experts en segmentation, une équipe en classification et une équipe en détection, on part du même backbone pré-entraîné et on l'adapte.

**Transfert efficace :** les représentations apprises par le modèle fondationnel sur des données génériques sont suffisamment riches pour être utiles sur des données très spécifiques (manuscrits médiévaux) avec peu d'exemples d'adaptation.

---

## 2. Les quatre outils du Jour 3 : une vue d'ensemble

### 2.1 Tableau synoptique

| | **DINO / DINOv2** | **CLIP** | **SAM** | **TrOCR** |
|---|---|---|---|---|
| **Tâche principale** | Représentations visuelles génériques | Alignement vision-langage | Segmentation promptable | Reconnaissance de texte manuscrit |
| **Paradigme** | Auto-supervisé (SSL) | Supervisé par le langage | Supervisé sur 1Md masques | Seq2seq supervisé |
| **Données** | Images non étiquetées | 400M paires image-texte | SA-1B : 1Md masques | Texte imprimé + HTR |
| **Encodeur image** | ViT-S/B/L/G | ViT-B/L/H + ResNet | ViT-H (MAE) | BEiT (ViT masqué) |
| **Sortie** | Vecteur de features | Score de similarité | Masque binaire | Séquence de caractères |
| **Superviseur** | Lui-même (distillation) | Texte (contrastif) | Vérité terrain de masques | Transcription cible |
| **Zero-shot** | Clustering, classification | Classification, description | Segmentation | Non (fine-tuning nécessaire) |
| **Fine-tuning** | Sonde linéaire, LoRA | CoOp, sonde linéaire | Non recommandé | Oui (LoRA, full) |
| **Usage dans le pipeline** | Clustering des mains | Description des illustrations | Layout segmentation | Transcription HTR |

### 2.2 Positionnement dans le pipeline du projet

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  SCAN DE MANUSCRIT (entrée)                                                  │
└──────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────┐
│  Prétraitement  │  OpenCV + Sauvola + CLAHE (section 4.1)
│  (Jour 4)       │
└────────┬────────┘
         │
         ├──────────────────────────────────────────────────────┐
         │                                                      │
         ▼                                                      ▼
┌─────────────────────┐                          ┌─────────────────────────┐
│  DINO v2            │                          │  SAM                    │
│  Clustering des     │                          │  Segmentation de layout │
│  pages par style    │                          │  (colonnes, marges,     │
│  d'écriture         │                          │   illustrations)        │
│  (section 3.1)      │                          │  (section 3.3)          │
└─────────────────────┘                          └──────────┬──────────────┘
                                                            │
                                          ┌─────────────────┴──────────────┐
                                          │                                │
                                          ▼                                ▼
                                 ┌────────────────┐            ┌───────────────────┐
                                 │  TrOCR         │            │  CLIP             │
                                 │  Transcription │            │  Description des  │
                                 │  HTR ligne     │            │  illustrations     │
                                 │  par ligne     │            │  (section 3.2)    │
                                 │  (section 3.4) │            └───────────────────┘
                                 └───────┬────────┘
                                         │
                                         ▼
                              ┌─────────────────────┐
                              │  Dataset NLP        │
                              │  (JSON + HF)        │
                              └─────────────────────┘
```

---

## 3. L'architecture commune : le Vision Transformer comme fondation

### 3.1 ViT comme backbone universel

Le premier point commun fondamental des quatre outils est architectural : **ils reposent tous sur des Vision Transformers** (ViT) comme encodeur d'image. Cette convergence n'est pas un hasard.

Le mécanisme de self-attention globale de ViT (section 2.2) offre trois avantages décisifs pour les modèles fondationnels :

**Scalabilité.** Contrairement aux CNN, les ViT continuent de bénéficier de l'augmentation de la taille du modèle et des données d'entraînement sans saturer rapidement. Un ViT-G (1,1 milliard de paramètres) est significativement meilleur qu'un ViT-B (86 millions), alors que l'écart de performance entre ResNet-50 et ResNet-152 est bien plus modeste.

**Flexibilité.** Les ViT n'ont pas de biais de localité comme les CNN. Ils peuvent apprendre des patterns à n'importe quelle distance spatiale, ce qui les rend adaptables à des tâches très différentes sans modification architecturale.

**Compatibilité avec le masquage.** La représentation en tokens discrets (patches) de ViT rend naturelle l'application de techniques de masquage issues du NLP (BERT, GPT) — comme BEiT pour TrOCR et MAE pour SAM. C'est le pont qui a permis le transfert des techniques de préentraînement du NLP vers la vision.

```python
# Les quatre modèles utilisent tous des ViT — avec des configurations différentes
CONFIGURATIONS_VIT = {
    "DINO/DINOv2": {
        "modele_hf": "facebook/dinov2-base",
        "backbone" : "ViT-B/14",
        "params"   : "86 M",
        "patch_size": 14,
        "d_model"  : 768,
        "n_heads"  : 12,
        "n_layers" : 12,
        "pretraining": "DINO/iBOT (auto-supervisé)"
    },
    "CLIP": {
        "modele_hf": "openai/clip-vit-large-patch14",
        "backbone" : "ViT-L/14",
        "params"   : "307 M",
        "patch_size": 14,
        "d_model"  : 1024,
        "n_heads"  : 16,
        "n_layers" : 24,
        "pretraining": "Contrastif image-texte (400M paires)"
    },
    "SAM": {
        "backbone" : "ViT-H/16",
        "params"   : "632 M",
        "patch_size": 16,
        "d_model"  : 1280,
        "n_heads"  : 16,
        "n_layers" : 32,
        "pretraining": "MAE sur SA-1B (1,1 Md masques)"
    },
    "TrOCR": {
        "modele_hf": "microsoft/trocr-base-handwritten",
        "backbone" : "BEiT-Base (encodeur)",
        "params"   : "86 M (encodeur) + 125 M (décodeur GPT-2)",
        "patch_size": 16,
        "d_model"  : 768,
        "n_heads"  : 12,
        "n_layers" : 12,
        "pretraining": "BEiT (masquage de patches visuels)"
    },
}

for nom, config in CONFIGURATIONS_VIT.items():
    print(f"\n{nom}")
    print(f"  Backbone    : {config['backbone']}")
    print(f"  Paramètres  : {config['params']}")
    print(f"  Patch size  : {config['patch_size']}px × {config['patch_size']}px")
    print(f"  Préentrainement : {config['pretraining']}")
```

### 3.2 Le token CLS et les features de patches

Tous les ViT utilisés par ces modèles produisent deux types de représentations en sortie de l'encodeur :

**Le token CLS** : un vecteur unique représentant l'image entière, obtenu par le token de classification appris qui interagit avec tous les patches via l'attention. C'est ce que DINO et CLIP utilisent comme représentation globale de l'image pour le clustering et la comparaison vision-texte.

**Les features de patches** : $N$ vecteurs (un par patch), chacun représentant une région locale de l'image tout en intégrant le contexte global via l'attention. C'est ce que SAM utilise dans sa feature map encodeur ($64 \times 64 \times 256$), et ce que TrOCR transmet au décodeur via la cross-attention.

```python
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Illustration : extraire token CLS et features de patches avec DINOv2
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model     = AutoModel.from_pretrained("facebook/dinov2-base").to(DEVICE).eval()

image = Image.open("scan_manuscrit.tif").convert("RGB")
inputs = processor(images=image, return_tensors="pt").to(DEVICE)

with torch.no_grad():
    outputs = model(**inputs)

# Token CLS — représentation globale de l'image
cls_token = outputs.last_hidden_state[:, 0, :]     # (1, 768)
print(f"Token CLS (représentation globale) : {cls_token.shape}")

# Features de patches — représentations locales
patch_tokens = outputs.last_hidden_state[:, 1:, :]  # (1, N_patches, 768)
n_patches = patch_tokens.shape[1]
side = int(n_patches ** 0.5)
print(f"Features de patches : {patch_tokens.shape}")
print(f"  → Grille de patches : {side} × {side}")

# Cette même structure est celle que SAM (sa feature map) et
# TrOCR (ses représentations encodeur) utilisent, chacun à leur façon.
```

---

## 4. Les quatre paradigmes d'entraînement

La deuxième différence fondamentale entre les outils est leur **objectif d'entraînement** — ce qu'ils ont appris à faire à partir de leurs données massives.

### 4.1 DINO : apprendre à se comprendre soi-même

DINO (*Self-**DI**stillation with **NO** labels*) apprend sans aucune annotation humaine. Son objectif : entraîner un réseau (le *student*) à reproduire les représentations d'une version de lui-même mise à jour lentement (le *teacher*), sur des vues différentes de la même image.

**Ce que DINO apprend :** des représentations qui encodent la *sémantique visuelle* — ce que représente l'image — indépendamment des détails de surface (éclairage, angle, résolution). Ces représentations sont idéales pour regrouper des images similaires (clustering) sans jamais avoir vu de labels.

**La propriété émergente remarquable :** sans aucune supervision de segmentation, DINO développe spontanément des attention maps qui correspondent à une segmentation sémantique. Les têtes d'attention de la dernière couche se spécialisent pour délimiter les objets — un phénomène impossible avec les CNN.

```
DINO — Schéma d'entraînement

Image
  ↓ Augmentation 1 (vue locale)   Augmentation 2 (vue globale)
  ↓                                     ↓
[Student ViT]                     [Teacher ViT]
  ↓                                     ↓
Représentation s                  Représentation t
  ↓                                     ↓
     Minimiser H(t, s) = −∑ t log s
     [L'étudiant apprend à prédire le maître]
     [Le maître = EMA de l'étudiant = stable]
```

### 4.2 CLIP : apprendre l'alignement vision-langage

CLIP (*Contrastive Language-Image Pre-Training*) apprend à partir de 400 millions de paires (image, texte) collectées sur le web. Son objectif : placer l'image et sa description textuelle proches dans un espace commun, et éloigner les paires non correspondantes.

**Ce que CLIP apprend :** un espace joint image-texte où la similarité cosinus encode la correspondance sémantique entre modalités. Une image de chevalier en armure et le texte « un chevalier en armure du XIVe siècle » auront des représentations proches ; une image de texte calligraphié et le texte « une recette de cuisine moderne » seront éloignés.

**La propriété émergente :** la classification zero-shot. En encodant des descriptions textuelles de classes (« une enluminure représentant une scène de bataille »), CLIP peut classer des images dans des catégories qu'il n'a jamais vues explicitement — sans aucun entraînement supplémentaire.

```
CLIP — Schéma d'entraînement

Image I ──→ [Encodeur image ViT] ──→ z_I ∈ ℝ^d
                                            \
                                             → Matrice de similarité (N×N)
                                            /
Texte T ──→ [Encodeur texte GPT] ──→ z_T ∈ ℝ^d

Loss contrastive : maximiser sim(z_I, z_T) pour les paires correctes
                   minimiser sim(z_I, z_T) pour les paires incorrectes
[Paires correctes sur la diagonale de la matrice N×N]
```

### 4.3 SAM : apprendre à segmenter n'importe quoi

SAM (*Segment Anything Model*) est entraîné de façon supervisée sur SA-1B — 1,1 milliard de masques sur 11 millions d'images, produits par un moteur de données en trois phases (annotation assistée → semi-automatique → entièrement automatique). Son objectif : étant donné un prompt (point, boîte, masque), produire un masque de segmentation correspondant.

**Ce que SAM apprend :** une notion de *cohérence visuelle* universelle — ce qui constitue une région visuellement cohérente dans une image, quelle que soit la sémantique. SAM ne sait pas si la région est « du texte » ou « une enluminure » — il sait seulement qu'elle forme un tout cohérent.

**La propriété émergente :** la généralisation zero-shot à des domaines jamais vus à l'entraînement. SAM n'a jamais vu de manuscrit médiéval, mais sa notion de cohérence visuelle s'applique immédiatement : une colonne de texte, une lettrine, une zone d'illustration sont toutes des régions visuellement cohérentes qu'il peut segmenter sur prompt.

```
SAM — Schéma d'inférence

Image ──→ [Encodeur ViT-H] ──→ Feature map (64×64×256)
                                        │
Prompt ──→ [Encodeur prompt] ──→ Tokens de prompt
                                        │
                           [Décodeur masque léger]
                                        │
                              3 Masques candidats + scores IoU
```

### 4.4 TrOCR : apprendre à lire

TrOCR (*Transformer-based OCR*) est entraîné de façon supervisée sur des paires (image de ligne, transcription) — des données synthétiques et réelles. Son objectif : générer la séquence de caractères correspondant à l'image de ligne de texte.

**Ce que TrOCR apprend :** deux choses simultanément — une représentation visuelle des traits d'écriture (via l'encodeur BEiT), et une modélisation de la langue (via le décodeur GPT-2). La cross-attention entre les deux permet au décodeur de « lire » l'image séquentiellement.

**La propriété distinctive :** c'est le seul des quatre modèles qui n'est pas zero-shot sur les manuscrits médiévaux — son modèle de langue est entraîné sur de l'anglais imprimé moderne, très différent du vieux français manuscrit. Le fine-tuning (section 3.5) est indispensable.

```
TrOCR — Schéma d'inférence

Image de ligne ──→ [Encodeur BEiT] ──→ Représentations des patches
                                               │ Cross-attention
[Début de séquence] ──→ [Décodeur GPT-2] ──→ Token 1 → Token 2 → ... → [Fin]
                           (autorégressif)           "E"    "n"          "s"
```

---

## 5. Les différences qui comptent pour notre projet

### 5.1 Supervision et généralisation

```
Degré de supervision des données d'entraînement →

    Aucune         Supervision          Supervision          Supervision
    supervision     indirecte            faible               forte
    ↓               ↓                    ↓                    ↓
   DINO            CLIP                 SAM                  TrOCR
   (images         (textes du           (masques             (transcriptions
   non étiquetées) web comme            générés              manuelles)
                   signal faible)       semi-auto)
```

La position sur cet axe détermine directement la **capacité de généralisation zero-shot** : DINO et CLIP généralisent très bien à des domaines non vus, SAM généralise bien (il n'a pas de biais sémantique), TrOCR nécessite un fine-tuning.

### 5.2 Ce que chaque modèle « voit »

Une façon intuitive de comprendre leurs différences est de se demander ce que chaque modèle « voit » dans une image de page de manuscrit :

**DINO voit la structure visuelle globale :** la distribution des traits d'encre, la texture générale de la page, le style d'écriture. Il ne sait pas *ce que ça dit*, mais il sait que deux pages écrites par le même copiste *se ressemblent*.

**CLIP voit la signification :** il comprend qu'une page avec beaucoup de texte et une enluminure représentant un chevalier correspond à la description « manuscrit médiéval illustré ». Il associe l'image à des concepts linguistiques.

**SAM voit les frontières :** il détecte ce qui est « un objet » ou « une région » par sa cohérence visuelle — les bords des lettres, les contours d'une illustration, les limites d'une colonne.

**TrOCR voit les symboles écrits :** il décode la séquence de caractères encodée dans les traits d'encre, en mobilisant simultanément la perception visuelle et le modèle de langue.

### 5.3 Complémentarité dans le pipeline

La puissance de leur combinaison vient précisément de leur complémentarité :

- SAM segmente *sans nommer* — il découpe la page en régions.
- CLIP nomme *sans segmenter* — il décrit chaque région.
- DINO regroupe *sans annoter* — il organise les pages par style.
- TrOCR transcrit *ce que les autres ont localisé* — il lit ce que SAM a isolé.

```
Utilité combinée pour une page de manuscrit enluminée à deux colonnes :

DINO  → "Cette page ressemble aux autres du même copiste (groupe C)"
         → Orienter le choix du modèle TrOCR fine-tuné

SAM   → "Voici 47 régions visuellement cohérentes"
         → Délimiter colonnes, lettrine, enluminure, marges

CLIP  → "Région 23 : 'an illuminated scene showing a knight on horseback'"
         → Décrire l'enluminure pour l'index de recherche

TrOCR → "Région 1 (colonne gauche) : 'En cel tems que li rois Artus...'"
         → Transcrire le texte ligne par ligne
```

---

## 6. La connexion avec les architectures du Jour 2

### 6.1 Ce que le Jour 2 a préparé

Chacun des quatre outils du Jour 3 s'appuie directement sur des concepts vus le Jour 2 :

**DINO → Auto-attention et token CLS (section 2.2)**
L'émergence de la segmentation dans les attention maps de DINO n'est compréhensible qu'en comprenant le mécanisme d'attention : les têtes d'attention apprennent à aligner les tokens les plus sémantiquement proches, ce qui produit spontanément un regroupement des pixels par objet.

**CLIP → Multi-Head Attention et encodeur Transformer (section 2.2)**
L'encodeur texte de CLIP est un Transformer de type GPT — exactement l'architecture encodeur étudiée en section 2.2. La projection dans un espace commun image-texte est une extension directe de l'idée que l'attention peut opérer sur des séquences hétérogènes (patches visuels ou tokens textuels).

**SAM → ViT et MAE (section 2.2)**
L'encodeur ViT-H de SAM est pré-entraîné par MAE (*Masked Autoencoder*) — une variante de BEiT (section 2.2) qui reconstruit les pixels masqués plutôt que de prédire des tokens discrets. Le décodeur de masque de SAM utilise la cross-attention déjà vue dans l'architecture Transformer.

**TrOCR → Encodeur-décodeur et cross-attention (section 2.2)**
TrOCR est l'instanciation directe de l'architecture encodeur-décodeur avec cross-attention présentée en section 2.2. L'encodeur BEiT est un ViT pré-entraîné par masquage ; le décodeur GPT-2 est un Transformer autorégressif ; leur connexion par cross-attention est exactement le mécanisme d'attention de Bahdanau généralisé.

**U-Net et DeepLab (sections 2.2b et 2.2c) → BLLA dans le pipeline**
Le module BLLA de Kraken, qui segmente les lignes dans notre pipeline, est un réseau de type U-Net — architecture vue en section 2.2b. SAM (section 3.3) et BLLA opèrent à des granularités différentes mais avec la même logique de segmentation précise que U-Net.

### 6.2 Ce que le Jour 3 apporte en plus

| Concept du Jour 2 | Ce que le Jour 3 apporte |
|-------------------|--------------------------|
| ViT : architecture | ViT : préentraîné à très grande échelle, propriétés émergentes |
| Self-attention : mécanisme | Self-attention : à l'origine de la segmentation sans supervision (DINO) |
| Encodeur-décodeur : structure | Encodeur-décodeur : appliqué au HTR avec cross-attention image→texte (TrOCR) |
| Masquage de patches (BEiT) : technique | Masquage : fondation du préentraînement de SAM (MAE) et TrOCR (BEiT) |
| Segmentation : tâche | Segmentation zero-shot et promptable (SAM) |
| Multimodalité : concept | Espace commun image-texte appris à grande échelle (CLIP) |

---

## 7. Repères chronologiques : comment ces outils se situent dans l'histoire

Comprendre l'ordre d'apparition de ces outils permet de mieux saisir pourquoi chacun a résolu un problème précis de son époque.

```
2017  ─── Transformer (Vaswani et al.)
      │   « Attention is all you need »

2018  ─── BERT (Devlin et al.)
      │   Masquage + pré-entraînement → révolution NLP

2020  ─── ViT (Dosovitskiy et al.)
      │   Les Transformers fonctionnent en vision

2021  ─── CLIP (Radford et al.)                ← Alignement vision-langage
      │   DINO (Caron et al.)                  ← SSL en vision
      │   BEiT (Bao et al.)                    ← Masquage en vision

2022  ─── TrOCR (Li et al.)                    ← HTR par Transformer
      │   MAE (He et al.)                      ← Masquage haute performance

2023  ─── SAM (Kirillov et al.)                ← Segmentation universelle
      │   DINOv2 (Oquab et al.)               ← DINO à grande échelle

2024  ─── SAM 2 (Ravi et al.)                  ← Extension à la vidéo
      │   Florence-2, LLaVA, GPT-4V            ← Vision-langage génératif
      │   CATMuS (Clérice et al.)              ← Dataset médiéval multilingue

2025  ─── Modèles vision-langage généralistes  ← Fusion de toutes ces approches
           (Gemini, Claude, GPT-4o…)
```

**La leçon de cette chronologie :** DINO et CLIP ont posé les bases (2021) — l'auto-supervision visuelle et l'alignement vision-texte. TrOCR a appliqué ces bases à l'OCR (2022). SAM a poussé l'idée jusqu'à ses limites en vision (2023). DINOv2 a montré que la mise à l'échelle restait payante (2023). Les quatre outils que nous étudions représentent l'état de l'art opérationnel pour les humanités numériques en 2024–2026 — ils ne sont pas dépassés, mais ils s'inscrivent dans une évolution continue.

---

## 8. Ce que vous pouvez (et ne pouvez pas) en attendre pour les manuscrits médiévaux

### 8.1 Capacités établies

| Outil | Ce qu'il fait bien sur les manuscrits médiévaux |
|-------|------------------------------------------------|
| **DINOv2** | Clustering de pages par style d'écriture, sans annotation |
| **DINOv2** | Détection de changements de main en cours de manuscrit |
| **CLIP** | Description générique des enluminures (scènes religieuses, chevaliers, animaux) |
| **CLIP** | Recherche cross-modale : trouver des images correspondant à une description |
| **SAM** | Isolation des colonnes de texte, des marges, des grandes enluminures |
| **SAM** | Segmentation des zones à fort contraste (texte vs fond) |
| **TrOCR (fine-tuné)** | Transcription de textes dans le style vu à l'entraînement (CER ~8–12%) |

### 8.2 Limites connues

| Outil | Limitations pour les manuscrits médiévaux |
|-------|------------------------------------------|
| **DINOv2** | Ne connaît pas les scripts médiévaux spécifiquement — clustering basé sur des features visuelles génériques |
| **CLIP** | Mauvaises performances sur l'iconographie médiévale rare (scènes hagiographiques précises, symbolisme héraldique) |
| **SAM** | Ne segmente pas les lignes de texte individuelles (granularité trop fine) |
| **SAM** | Ne classe pas les régions (texte vs image vs rubrique) — CLIP nécessaire pour ça |
| **TrOCR** | Sans fine-tuning, CER > 40% sur le vieux français manuscrit |
| **TrOCR** | Tokenizer GPT-2 sous-optimal pour les formes médiévales (abréviations, diphtongues) |

---

## Glossaire des termes clés

**Alignement vision-langage**
Apprentissage d'un espace de représentation commun où des images et leurs descriptions textuelles sont proches. CLIP en est l'exemple canonique. Permet la classification zero-shot et la recherche cross-modale.

**Auto-supervision (SSL — *Self-Supervised Learning*)**
Apprentissage sans annotation humaine, en créant des tâches prétextes à partir des données brutes. DINO utilise la distillation comme tâche prétexte : le student apprend à reproduire les représentations du teacher sur des vues différentes de la même image.

**Cross-attention**
Mécanisme d'attention où les requêtes ($Q$) viennent d'une séquence et les clés/valeurs ($K$, $V$) d'une autre. Dans TrOCR, le décodeur texte ($Q$) lit les représentations visuelles de l'encodeur ($K$, $V$) pour décoder les caractères. Dans SAM, les tokens de masque lisent la feature map de l'image.

**Distillation de connaissances (*knowledge distillation*)**
Entraîner un modèle (student) à reproduire les sorties d'un modèle plus puissant ou plus stable (teacher). Dans DINO, le teacher est une version lissée par EMA du student lui-même (*self-distillation*).

**EMA (*Exponential Moving Average*)**
Mise à jour lissée de paramètres : $\theta_{\text{teacher}} \leftarrow m \cdot \theta_{\text{teacher}} + (1-m) \cdot \theta_{\text{student}}$ avec $m$ proche de 1. Produit un teacher plus stable que le student, essentiel pour éviter l'effondrement dans DINO.

**Émergence (*emergence*)**
Apparition de capacités non explicitement entraînées dans un modèle de grande taille. Exemple : la segmentation sémantique dans les attention maps de DINO, ou la classification zero-shot de CLIP, n'ont jamais été supervisées.

**Fine-tuning**
Adaptation d'un modèle pré-entraîné à une tâche spécifique par continuation de l'entraînement sur des données de la tâche cible. Pour TrOCR sur les manuscrits médiévaux, indispensable. Pour DINO et CLIP, optionnel (une sonde linéaire suffit souvent).

**Fondation (modèle fondationnel)**
Modèle pré-entraîné à très grande échelle sur des données larges et diversifiées, servant de base pour de nombreuses tâches en aval. Caractérisé par l'émergence, l'homogénéisation et le transfert efficace.

**MAE (*Masked Autoencoder*)**
Technique de préentraînement ViT : masquer un pourcentage élevé de patches (~75%) et entraîner le modèle à reconstruire les pixels masqués. Pré-entraîne l'encodeur de SAM. Analogue à BERT en NLP (mais en pixels plutôt qu'en tokens de texte).

**Sonde linéaire (*linear probe*)**
Évaluation des features d'un encodeur pré-entraîné en entraînant uniquement une couche linéaire (régression logistique) sur ces features gelées. Mesure la qualité des représentations indépendamment du fine-tuning.

**Token CLS (*Classification Token*)**
Token spécial ajouté en début de séquence dans les ViT et BERT. Après le passage dans l'encodeur, sa représentation agrège l'information de toute la séquence via l'attention. Utilisé comme vecteur de représentation globale de l'image dans DINO et CLIP.

**Zero-shot**
Capacité d'un modèle à résoudre une tâche sans avoir été entraîné sur des exemples de cette tâche. CLIP classe des images dans des catégories décrites textuellement ; SAM segmente des types d'objets jamais vus. Rendu possible par le préentraînement à très grande échelle.

---

## Bibliographie de référence

### Modèles fondationnels : concept et panorama

- **Bommasani, R. et al.** (2021). *On the Opportunities and Risks of Foundation Models*. [arXiv:2108.07258] — Le document fondateur du concept de « foundation model ». Analyse les propriétés (émergence, homogénéisation), les opportunités et les risques de 100+ pages.

- **LeCun, Y.** (2022). *A Path Towards Autonomous Machine Intelligence*. OpenReview. — Vision de Yann LeCun sur les limitations de l'apprentissage contrastif et les alternatives (Joint Embedding Predictive Architecture).

### DINO et DINOv2

- **Caron, M. et al.** (2021). *Emerging Properties in Self-Supervised Vision Transformers (DINO)*. ICCV 2021. [arXiv:2104.14294] — Article fondateur.
- **Oquab, M. et al.** (2023). *DINOv2: Learning Robust Visual Features without Supervision*. TMLR 2024. [arXiv:2304.07193] — Version à grande échelle.

### CLIP

- **Radford, A. et al.** (2021). *Learning Transferable Visual Models From Natural Language Supervision*. ICML 2021. [arXiv:2103.00020] — Article fondateur.
- **Zhou, K. et al.** (2022). *Learning to Prompt for Vision-Language Models (CoOp)*. IJCV. [arXiv:2109.01134] — Fine-tuning efficace de CLIP.

### SAM

- **Kirillov, A. et al.** (2023). *Segment Anything*. ICCV 2023. [arXiv:2304.02643] — Article fondateur.
- **Ravi, N. et al.** (2024). *SAM 2: Segment Anything in Images and Videos*. [arXiv:2408.00714] — Extension vidéo et améliorations.

### TrOCR et HTR

- **Li, M. et al.** (2021). *TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models*. AAAI 2023. [arXiv:2109.10282] — Article fondateur.
- **Bao, H. et al.** (2022). *BEiT: BERT Pre-Training of Image Transformers*. ICLR 2022. [arXiv:2106.08254] — Préentraînement de l'encodeur de TrOCR.

### Préentraînement et architectures ViT

- **Dosovitskiy, A. et al.** (2020). *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale*. ICLR 2021. [arXiv:2010.11929] — ViT original.
- **He, K. et al.** (2022). *Masked Autoencoders Are Scalable Vision Learners*. CVPR 2022. [arXiv:2111.06377] — MAE : préentraînement de SAM.
- **Devlin, J. et al.** (2018). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL 2019. [arXiv:1810.04805] — Le modèle NLP dont s'inspirent BEiT et DINO.

### Applications aux humanités numériques

- **Constum, T. et al.** (2024). *CATMuS Medieval: A Multilingual Large-Scale Cross-Century Dataset in Latin Script for Handwritten Text Recognition and Beyond*. [arXiv:2402.11150] — Application de ces modèles aux corpus patrimoniaux.

---

*Support de cours rédigé pour le module Computer Vision · Promotion MD5 · Mai 2026*
*Ce document constitue la section introductive 3.0 du Jour 3.*
*Il précède les sections 3.1 (DINO), 3.2 (CLIP), 3.3 (SAM) et 3.4 (TrOCR).*
*Durée estimée de lecture : 50–60 minutes.*
