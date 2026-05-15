# Cours 2.0 — Architectures de vision profonde : reconnaissance, détection et segmentation

**Module 2 · Computer Vision appliquée aux manuscrits médiévaux · MD5**
**Section introductive au Jour 2**

---

> *« La vision par ordinateur s'est longtemps limitée à une question : qu'est-ce que c'est ? Il a fallu décennies pour apprendre à répondre à deux autres : où est-ce ? et exactement quels pixels lui appartiennent ? Ces trois questions définissent trois familles d'architectures profondes, radicalement différentes dans leur conception — et toutes utiles pour traiter un manuscrit médiéval. »*

--

## Introduction : trois questions, trois familles de tâches

Avant d'entrer dans le détail des architectures CNN et Transformer (sections 2.1 et 2.2), il est nécessaire de cartographier l'espace des tâches de vision par ordinateur. Ces tâches définissent des contraintes différentes sur la sortie du modèle, et ces contraintes se traduisent directement par des choix architecturaux différents.

**Classification** : *Quelle classe est présente dans cette image ?*
L'image entière reçoit une étiquette (ou un vecteur de probabilités sur $C$ classes). La localisation n'est pas requise.

**Détection d'objets** : *Où sont les objets, et quels sont-ils ?*
Pour chaque objet dans l'image, on prédit une boîte englobante (*bounding box*) et une classe.

**Segmentation sémantique** : *À quelle classe appartient chaque pixel ?*
Chaque pixel reçoit une étiquette de classe. Deux pixels appartenant à la même classe mais à des objets différents reçoivent la même étiquette.

**Segmentation d'instances** : *À quel objet individuel appartient chaque pixel ?*
Chaque pixel est associé à une instance spécifique. Deux objets de la même classe produisent des masques distincts.

**Segmentation panoptique** : *Unification complète des deux précédentes.*
Chaque pixel est associé à une classe ET, pour les objets individuels (*things*), à une instance spécifique. Les régions de fond (*stuff*) reçoivent uniquement une classe.

```
Complexité croissante →

CLASSIFICATION     DÉTECTION         SEG. SÉMANTIQUE  SEG. INSTANCES  SEG. PANOPTIQUE
┌─────────┐        ┌─────────┐       ┌─────────┐      ┌─────────┐     ┌─────────┐
│         │        │ ┌──┐    │       │▓▓▓▓▓░░░░│      │▓₁▓₁░₂░₂│     │▓₁▓₁░░░░│
│  "chat" │        │ │🐱│"🐱"│       │▓▓▓▓▓░░░░│      │▓₁▓₁░₂░₂│     │▓₁▓₁░░░░│
│         │        │ └──┘    │       │░░░░░░░░░│      │░░░░░░░░░│     │░₃░₃░₃░₃│
└─────────┘        └─────────┘       └─────────┘      └─────────┘     └─────────┘
Sortie: vecteur    Sortie: boîtes    Sortie: carte     Sortie: masques  Sortie: tout
de probabilités    + classes         de classes pixel  par instance     à la fois
```

Pour notre projet HTR, ces quatre tâches correspondent à des étapes distinctes du pipeline :
- **Classification** : distinguer une page de texte d'une page d'illustration.
- **Détection** : localiser les régions de texte, les lettrines, les enluminures.
- **Segmentation sémantique** : séparer les pixels d'encre des pixels de fond.
- **Segmentation d'instances** : isoler chaque ligne de texte individuellement.

---

## 1. Architectures de classification

### 1.1 Le paradigme CNN pour la classification

Les architectures de classification partagent toutes la même structure en deux temps : un **extracteur de caractéristiques** (backbone) et une **tête de classification** (head).

```
Image (H×W×C)
      ↓
[ Backbone : couches convolutives ]
      ↓
Feature map (h×w×D)     avec h≪H, w≪W, D grand
      ↓
[ Global Average Pooling ou Flatten ]
      ↓
Vecteur (D,)
      ↓
[ Tête FC : D → n_classes ]
      ↓
Logits (n_classes,)     → softmax → probabilités
```

La **Global Average Pooling** (Lin et al., 2014) calcule la moyenne spatiale de chaque canal de la feature map :

$$\text{GAP}(F)_d = \frac{1}{h \cdot w} \sum_{i=1}^h \sum_{j=1}^w F_{i,j,d}$$

Elle produit un vecteur de dimension $D$ indépendant de la taille d'entrée, et réduit considérablement le nombre de paramètres de la tête de classification (remplace les couches fully connected denses de AlexNet/VGG).

### 1.2 L'évolution des backbones

```python
# Comparaison des principales architectures de backbone
# (Illustratif — pas les implémentations complètes)

import torch
import torch.nn as nn

# ── LeNet-5 (LeCun et al., 1998) ──────────────────────────────────────────────
class LeNet5(nn.Module):
    """
    La première architecture CNN opérationnelle (1998).
    Conçue pour MNIST (28×28 px, chiffres écrits à la main).
    Architecture de référence qui a défini le paradigme Conv → Pool → FC.
    """
    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),    nn.Tanh(),  nn.AvgPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5),   nn.Tanh(),  nn.AvgPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16*4*4, 120), nn.Tanh(),
            nn.Linear(120, 84),     nn.Tanh(),
            nn.Linear(84, n_classes),
        )
    def forward(self, x): return self.classifier(self.features(x))

# ── Bloc résiduel ResNet (He et al., 2016) ────────────────────────────────────
class BlocResiduel(nn.Module):
    """
    Bloc résiduel avec connexion raccourcie (skip connection).
    L'innovation clé de ResNet : apprendre F(x) = H(x) - x
    plutôt que H(x) directement.
    y = F(x, W) + x
    Permet d'entraîner des réseaux très profonds (50, 101, 152 couches).
    """
    def __init__(self, d: int):
        super().__init__()
        self.bloc = nn.Sequential(
            nn.Conv2d(d, d, 3, padding=1, bias=False), nn.BatchNorm2d(d), nn.ReLU(),
            nn.Conv2d(d, d, 3, padding=1, bias=False), nn.BatchNorm2d(d),
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bloc(x) + x)   # F(x) + x = H(x)
```

**Jalons architecturaux en classification**

| Architecture | Année | Paramètres | Top-1 ImageNet | Innovation clé |
|-------------|-------|------------|----------------|----------------|
| LeNet-5 | 1998 | 60 K | — | Convolutions, pooling |
| AlexNet | 2012 | 60 M | 63.3% | ReLU, Dropout, GPU |
| VGG-16 | 2014 | 138 M | 74.4% | Petits filtres 3×3 empilés |
| GoogLeNet/Inception | 2014 | 6.8 M | 74.8% | Modules inception multi-échelle |
| ResNet-50 | 2016 | 25 M | 76.1% | Skip connections, profondeur |
| DenseNet-121 | 2017 | 8 M | 74.9% | Connexions entre toutes les couches |
| EfficientNet-B7 | 2019 | 66 M | 84.3% | Compound scaling |
| ViT-B/16 | 2020 | 86 M | 81.8% | Patches + self-attention |

---

## 2. Architectures de détection d'objets

La détection ajoute une dimension spatiale à la classification : on ne veut pas seulement savoir *quoi*, mais aussi *où*. La sortie est un ensemble de tuples (boîte, classe, score) pour chaque objet détecté.

### 2.1 Les détecteurs en deux étapes : la famille R-CNN

**R-CNN** (Girshick et al., 2014) est la première architecture profonde de détection. Son principe : séparer la proposition de régions (*region proposal*) de la classification de ces régions.

```
Image
  ↓
[Selective Search]          ← ~2000 régions candidates
  ↓
Pour chaque région :
  → Redimensionnement à taille fixe (227×227)
  → Passage dans AlexNet
  → Extraction de features (4096-d)
  → SVM de classification
  → Régression de boîte
```

**Limite majeure :** chaque région est passée indépendamment dans le CNN. Pour 2000 régions et une image, cela représente 2000 passes *forward* — extrêmement lent (~47 secondes/image).

**Fast R-CNN** (Girshick, 2015) résout cette inefficacité par l'opération **RoI Pooling** (*Region of Interest Pooling*) :

```
Image
  ↓
[ CNN backbone ] → Feature map globale (une seule passe)
  ↓                        ↓
[Selective Search] → Régions candidates → RoI Pooling
                     (coordonnées)            ↓
                                        Vecteur fixe par région
                                              ↓
                                        FC → classe + boîte
```

Le **RoI Pooling** extrait une feature map de taille fixe ($H \times W$) pour chaque région candidate en divisant la région en $H \times W$ cellules et en faisant un max-pooling dans chaque cellule. Il permet de partager le calcul du backbone entre toutes les régions.

**Faster R-CNN** (Ren et al., 2016) remplace Selective Search par un **RPN** (*Region Proposal Network*) entraînable, appris conjointement avec le reste du réseau :

```
Image
  ↓
[ Backbone CNN ] → Feature map partagée
                         ↓            ↓
                      [ RPN ]    [ RoI Pooling ]
                         ↓            ↓
                    Régions → candidates → Tête de classification
```

Le RPN glisse une fenêtre sur la feature map et prédit, pour chaque position, $k$ **anchors** (boîtes de référence de tailles et ratios prédéfinis) : est-ce un objet ? et quelle correction de boîte appliquer ?

```python
import torch
import torch.nn as nn

class RPN(nn.Module):
    """
    Region Proposal Network (Ren et al., 2016).
    Pour chaque position spatiale de la feature map, prédit k boîtes candidates.
    """
    def __init__(self, in_channels: int = 512, n_anchors: int = 9):
        """
        n_anchors = 9 : 3 échelles × 3 ratios (typique pour Faster R-CNN)
        Échelles : 128, 256, 512 pixels
        Ratios   : 1:1, 1:2, 2:1
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 512, 3, padding=1)
        self.relu = nn.ReLU()
        # Classification : pour chaque anchor, P(objet) et P(fond)
        self.cls_logits = nn.Conv2d(512, n_anchors * 2, 1)
        # Régression : pour chaque anchor, (δx, δy, δw, δh)
        self.bbox_pred  = nn.Conv2d(512, n_anchors * 4, 1)

    def forward(self, feature_map: torch.Tensor):
        x    = self.relu(self.conv(feature_map))
        cls  = self.cls_logits(x)   # (B, 2k, H, W)
        bbox = self.bbox_pred(x)    # (B, 4k, H, W)
        return cls, bbox
```

**La loss de détection** combine une term de classification et un terme de régression :

$$\mathcal{L}_{\text{det}} = \mathcal{L}_{\text{cls}}(p_i, p_i^*) + \lambda \sum_i p_i^* \mathcal{L}_{\text{reg}}(t_i, t_i^*)$$

où $p_i$ est la probabilité prédite d'objet pour l'anchor $i$, $p_i^*$ est 1 si l'anchor est positif (IoU > 0.7) et 0 sinon, $t_i$ est la correction de boîte prédite et $t_i^*$ la correction cible. La régression n'est pénalisée que pour les anchors positifs ($p_i^* = 1$).

### 2.2 Les détecteurs en une étape

Les détecteurs en deux étapes sont précis mais lents. Les détecteurs en une étape suppriment la phase de proposal et prédisent directement boîtes et classes depuis la feature map.

**YOLO** (*You Only Look Once*, Redmon et al., 2016) divise l'image en une grille $S \times S$. Chaque cellule prédit directement $B$ boîtes et $C$ probabilités de classe :

```
Image (448×448)
      ↓
[ Backbone CNN (24 couches) ]
      ↓
Grille S×S (7×7 dans YOLO v1)
      ↓
Pour chaque cellule : B boîtes × (x, y, w, h, confiance) + C classes
Sortie : tenseur (S, S, B×5 + C)
```

**RetinaNet et la Focal Loss** (Lin et al., 2017)

Les détecteurs en une étape souffrent d'un fort déséquilibre de classes : sur une image typique, il y a quelques dizaines d'objets et des milliers de cellules de fond. L'entraînement est dominé par les exemples négatifs faciles (fond évident).

La **Focal Loss** modifie la cross-entropie pour down-pondérer les exemples bien classifiés :

$$\text{FL}(p_t) = -(1 - p_t)^\gamma \log(p_t)$$

Le facteur $(1 - p_t)^\gamma$ avec $\gamma > 0$ réduit la contribution des exemples faciles ($p_t$ proche de 1) : si $\gamma = 2$ et $p_t = 0.9$, le facteur est $(0.1)^2 = 0.01$ — l'exemple ne contribue que 1% de son poids original à la loss. Les exemples difficiles (petites valeurs de $p_t$) conservent un poids proche de 1.

### 2.3 DETR : détection par Transformer

**DETR** (*Detection Transformer*, Carion et al., 2020) reformule la détection comme un problème de prédiction directe d'un ensemble fixe de $N$ boîtes, sans anchors ni NMS :

```python
class DETR(nn.Module):
    """
    Detection Transformer (Carion et al., 2020) — version simplifiée illustrative.
    Prédit N boîtes simultanément via un Transformer encodeur-décodeur.
    """
    def __init__(self, backbone, d_model=256, n_heads=8, N=100, n_classes=91):
        super().__init__()
        self.backbone    = backbone
        self.proj        = nn.Conv2d(2048, d_model, 1)   # Projection du backbone
        self.transformer = nn.Transformer(d_model, n_heads, num_encoder_layers=6,
                                          num_decoder_layers=6)
        # N queries : vecteurs apprenables qui "cherchent" des objets
        self.queries     = nn.Embedding(N, d_model)
        self.cls_head    = nn.Linear(d_model, n_classes + 1)   # +1 pour "pas d'objet"
        self.box_head    = nn.Linear(d_model, 4)

    def forward(self, images: torch.Tensor):
        # Extraction des features par le backbone
        feat = self.proj(self.backbone(images))     # (B, d_model, H, W)
        B, C, H, W = feat.shape

        # Encodage de la feature map comme séquence (positions 2D aplaties)
        feat_seq = feat.flatten(2).permute(2, 0, 1)  # (H*W, B, d_model)

        # Décodage : les N queries lisent la feature map par cross-attention
        queries = self.queries.weight.unsqueeze(1).repeat(1, B, 1)  # (N, B, d_model)
        out = self.transformer(feat_seq, queries)                     # (N, B, d_model)

        # Prédiction de classe et de boîte pour chaque query
        out = out.permute(1, 0, 2)                   # (B, N, d_model)
        return self.cls_head(out), self.box_head(out).sigmoid()
```

**L'innovation clé de DETR :** la détection est formulée comme un problème d'affectation bipartite. Les $N$ prédictions sont associées aux $M$ objets réels par l'algorithme hongrois (*Hungarian matching*) qui minimise le coût global de l'affectation. Cela supprime le besoin d'anchors et de NMS.

**FPN — Feature Pyramid Network** (Lin et al., 2017)

La plupart des détecteurs modernes utilisent une FPN pour détecter des objets à différentes échelles. La FPN construit une pyramide de feature maps en combinant des features de bas niveau (haute résolution, faible sémantique) avec des features de haut niveau (basse résolution, haute sémantique) :

```
Backbone                           FPN
(ascendant)                     (descendant)
C2 (stride 4)  ─────────────────→ P2 (haute résolution)
C3 (stride 8)  ───────+──────────→ P3
C4 (stride 16) ───────+──────────→ P4
C5 (stride 32) ────────────────────→ P5 (basse résolution)

Chaque niveau Pi = Conv(Ci) + Upsample(Pi+1)
```

Les petits objets sont détectés à partir des niveaux haute résolution (P2, P3), les grands objets depuis les niveaux basse résolution (P4, P5).

---

## 3. Architectures de segmentation sémantique

La segmentation sémantique produit une carte de classe de même résolution que l'image d'entrée : $H \times W \times C$ où $C$ est le nombre de classes. Chaque pixel $(i, j)$ est classifié indépendamment.

### 3.1 Fully Convolutional Networks (FCN)

**FCN** (Long et al., 2015) est la première architecture de segmentation sémantique entièrement convolutive. L'idée centrale : remplacer les couches fully connected d'un classificateur (AlexNet, VGG) par des convolutions $1 \times 1$, qui produisent une carte de classes à basse résolution, puis upsampler cette carte à la résolution originale.

```
Image (H×W×3)
      ↓
[ Backbone VGG-16 ]        → feature maps à stride 32 (H/32 × W/32 × D)
      ↓
[ Conv 1×1 → n_classes ]   → carte de classes basse résolution
      ↓
[ Upsampling ×32 ]         → carte de classes haute résolution (H×W×n_classes)
```

L'upsampling direct ×32 produit des prédictions peu précises aux bords des objets. FCN-16s et FCN-8s améliorent cela en ajoutant des **skip connections** depuis des niveaux intermédiaires du backbone (stride 16 et stride 8) avant l'upsampling final.

### 3.2 U-Net : l'encodeur-décodeur avec skip connections

**U-Net** (Ronneberger et al., 2015), développé pour la segmentation d'images biomédicales, est devenu l'architecture de référence pour la segmentation sémantique en général — et particulièrement pour les documents.

```
Encodeur (contraction)           Décodeur (expansion)
──────────────────               ──────────────────
[Conv + Conv + MaxPool]      →   [Upsample + Concat + Conv + Conv]
[Conv + Conv + MaxPool]      →   [Upsample + Concat + Conv + Conv]
[Conv + Conv + MaxPool]      →   [Upsample + Concat + Conv + Conv]
[Conv + Conv + MaxPool]      →   [Upsample + Concat + Conv + Conv]
         ↓                               ↑
    [Bottleneck]          ───────────────
```

La caractéristique distinctive de U-Net est le **Concat** dans le décodeur : à chaque niveau, les features de l'encodeur (haute résolution, basse sémantique) sont concaténées avec les features du décodeur (basse résolution, haute sémantique). Ces skip connections transmettent l'information de localisation fine que les couches d'encodage successives ont dégradée.

```python
class BloquetUNet(nn.Module):
    """Double convolution 3×3 — brique de base de U-Net."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.bloc = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.bloc(x)

class UNet(nn.Module):
    """
    U-Net (Ronneberger et al., 2015).
    Architecture encodeur-décodeur avec skip connections.
    Conçue initialement pour les images biomédicales ;
    standard de facto pour la segmentation de documents.
    """
    def __init__(self, in_ch: int = 1, n_classes: int = 2,
                 features: list = [64, 128, 256, 512]):
        super().__init__()
        # Encodeur
        self.encodeurs = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)
        ch = in_ch
        for f in features:
            self.encodeurs.append(BloquetUNet(ch, f))
            ch = f
        # Bottleneck
        self.bottleneck = BloquetUNet(ch, ch * 2)
        # Décodeur
        self.decodeurs_up   = nn.ModuleList()
        self.decodeurs_bloc = nn.ModuleList()
        for f in reversed(features):
            self.decodeurs_up.append(
                nn.ConvTranspose2d(ch * 2, f, kernel_size=2, stride=2)
            )
            self.decodeurs_bloc.append(BloquetUNet(f * 2, f))   # ×2 car concat
            ch = f
        # Tête de classification
        self.tete = nn.Conv2d(features[0], n_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encodage + mise en cache des skip connections
        skips = []
        for enc in self.encodeurs:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        # Décodage avec skip connections
        for up, bloc, skip in zip(
            self.decodeurs_up, self.decodeurs_bloc, reversed(skips)
        ):
            x = up(x)
            x = torch.cat([x, skip], dim=1)   # Skip connection : concaténation
            x = bloc(x)
        return self.tete(x)
```

### 3.3 DeepLab : convolutions dilatées et champ réceptif

**DeepLab** (Chen et al., 2017) introduit les **convolutions dilatées** (*atrous convolutions*) pour augmenter le champ réceptif sans réduire la résolution spatiale.

Une convolution dilatée de taux $r$ applique le filtre avec un espacement de $r-1$ pixels entre les éléments du noyau :

$$y[i] = \sum_k x[i + r \cdot k] \cdot w[k]$$

Avec $r=1$ : convolution standard (champ réceptif $3 \times 3$).
Avec $r=2$ : le filtre $3 \times 3$ couvre un champ $5 \times 5$ sans paramètres supplémentaires.
Avec $r=4$ : champ $9 \times 9$ avec toujours 9 paramètres.

**ASPP** (*Atrous Spatial Pyramid Pooling*) applique des convolutions dilatées avec plusieurs taux $r$ en parallèle pour capturer des informations à multiple échelles :

```
Feature map
      ↓
┌─────────────────────────────────────────────────────┐
│ Conv 1×1    │ Conv dilatée r=6 │ Conv dilatée r=12  │
│ (contexte   │ (champ réceptif  │ (champ réceptif    │
│  local)     │  13×13)          │  25×25)            │
└─────────────────────────────────────────────────────┘
      ↓             ↓                   ↓
         [ Concaténation + Conv 1×1 ]
                    ↓
            Prédiction finale
```

### 3.4 SegFormer : segmentation par Transformer

**SegFormer** (Xie et al., 2021) remplace le backbone CNN par un **Mix Transformer** (MiT) hiérarchique qui produit des features multi-échelles, et utilise un décodeur MLP léger.

Le décodeur MLP de SegFormer est remarquablement simple : il agrège les features de quatre niveaux de résolution différente par simple projection linéaire et upsampling bilinéaire — aucune convolution dans le décodeur. Sa légèreté contraste avec les décodeurs lourds de DeepLab ou même U-Net.

---

## 4. Architectures de segmentation d'instances

La segmentation d'instances associe un masque binaire à chaque instance individuelle d'objet. Elle est plus difficile que la segmentation sémantique car deux objets de la même classe doivent être distingués.

### 4.1 Mask R-CNN

**Mask R-CNN** (He et al., 2017) étend Faster R-CNN en ajoutant une troisième tête parallèle aux têtes de classification et régression : la **tête de masque** (*mask head*).

```
Image
  ↓
[ Backbone + FPN ]
  ↓
[ RPN ] → Régions candidates
  ↓
[ RoI Align ]         ← Correction de Mask R-CNN par rapport au RoI Pooling
  ↓
┌─────────────────────────────────────────────────┐
│ Tête de classe   │ Tête de boîte │ Tête de masque│
│ (classification) │ (régression)  │ (FCN 28×28)   │
└─────────────────────────────────────────────────┘
```

**RoI Align vs RoI Pooling**

Le RoI Pooling introduit des erreurs d'arrondi lors de l'alignement des régions candidates sur la feature map (les coordonnées en virgule flottante sont arrondies au pixel entier le plus proche). Pour la détection, cette imprécision est généralement acceptable. Pour la segmentation de masques, elle est catastrophique — les masques prédits sont décalés de quelques pixels par rapport aux objets réels.

**RoI Align** résout ce problème par interpolation bilinéaire : les valeurs aux positions fractionnaires sont interpolées depuis les pixels voisins, sans arrondi.

La tête de masque est un petit FCN qui prédit un masque binaire $28 \times 28$ pour chaque instance et chaque classe. Pendant l'entraînement, la loss de masque est calculée uniquement sur la classe correcte — ce qui évite la compétition entre classes dans la prédiction du masque.

$$\mathcal{L}_{\text{masque}} = -\frac{1}{m^2} \sum_{i,j} \left[ y_{ij} \log \hat{y}_{ij}^{k^*} + (1-y_{ij}) \log(1-\hat{y}_{ij}^{k^*}) \right]$$

où $k^*$ est la classe correcte, $y_{ij}$ le masque de vérité terrain, et $\hat{y}_{ij}^{k^*}$ la prédiction du masque pour la classe $k^*$.

### 4.2 Méthodes directes : SOLO et SOLOv2

**SOLO** (*Segmenting Objects by Locations*, Wang et al., 2020) reformule la segmentation d'instances sans détection préalable. L'image est divisée en une grille $S \times S$. Chaque cellule prédit directement si elle est responsable d'une instance (celle dont le centre tombe dans la cellule) et le masque de cette instance.

Cette approche évite la cascade détection → masque et traite toutes les instances en parallèle. **SOLOv2** améliore SOLOv1 par une prédiction de masque dynamique plus efficace.

### 4.3 SAM dans ce panorama

SAM (*Segment Anything Model*, section 3.3 du cours) est une architecture de segmentation d'instances **promptable** — à mi-chemin entre la segmentation interactive et la segmentation automatique. Il ne prédit pas un ensemble fixe d'instances (comme Mask R-CNN) mais répond à des prompts spécifiant quelle instance segmenter.

Positionnement de SAM dans le panorama :
- Ni classificateur (pas d'étiquettes sémantiques)
- Ni détecteur standard (pas d'anchors, pas de grille)
- Segmentation d'instances promptable, zero-shot

Sa force pour les manuscrits médiévaux est précisément cette flexibilité : on peut lui demander de segmenter une lettrine, une enluminure, ou une colonne de texte sans avoir jamais entraîné un modèle sur ces catégories spécifiques.

---

## 5. Segmentation panoptique

La **segmentation panoptique** (Kirillov et al., 2019) unifie segmentation sémantique et d'instances en un seul paradigme. Elle distingue :

- Les **"things"** (*choses*) : objets comptables, avec des instances individuelles (personnes, voitures, enluminures). Traités comme en segmentation d'instances.
- Les **"stuff"** (*matière*) : régions de fond amorphes, non dénombrables (ciel, parchemin, espace inter-lignes). Traités comme en segmentation sémantique.

La qualité d'un système panoptique est mesurée par la **PQ** (*Panoptic Quality*) :

$$\text{PQ} = \underbrace{\frac{\sum_{(p,g) \in TP} \text{IoU}(p,g)}{|TP|}}_{\text{Qualité de segmentation}} \times \underbrace{\frac{|TP|}{|TP| + \frac{1}{2}|FP| + \frac{1}{2}|FN|}}_{\text{Qualité de reconnaissance}}$$

**Mask2Former** (Cheng et al., 2022) est l'architecture panoptique de référence. Elle reformule toutes les tâches de segmentation (sémantique, d'instances, panoptique) comme un problème de prédiction de masques par un décodeur Transformer. Un ensemble de requêtes apprenables sont affinées par cross-attention sur les features de l'image à plusieurs résolutions.

---

## 6. Récapitulatif comparatif

### 6.1 Tableau de comparaison des familles d'architectures

| Critère | Classification | Détection | Seg. sémantique | Seg. instances | Seg. panoptique |
|---------|---------------|-----------|-----------------|----------------|-----------------|
| **Sortie** | Vecteur de proba | Boîtes + classes | Carte pixel→classe | Masques par instance | Tout |
| **Granularité** | Image entière | Objet | Pixel | Pixel + instance | Pixel + instance + stuff |
| **Architecture type** | CNN + GAP + FC | RPN + Tête det. | Encodeur-décodeur | Mask R-CNN | Mask2Former |
| **Loss principale** | Entropie croisée | Cls + Régression | Entropie croisée | Cls + Reg + Masque | PQ + sous-losses |
| **Anchors ?** | Non | Oui (détect.) / Non (DETR) | Non | Oui (Mask R-CNN) / Non (SOLO) | Non |
| **Skip connections ?** | Non (ResNet: oui) | FPN | Oui (U-Net) | Oui (FPN) | Oui |
| **Gère multi-échelles ?** | Implicitement | FPN | ASPP / Skip | FPN | Multi-scale Transformer |
| **Modèle de référence** | ResNet, ViT | Faster R-CNN, DETR | DeepLab v3+, SegFormer | Mask R-CNN | Mask2Former |

### 6.2 Quel paradigme pour les manuscrits médiévaux ?

Chaque étape du pipeline HTR correspond à une tâche de vision distincte.

| Tâche dans le pipeline | Paradigme | Modèle utilisé dans ce cours |
|------------------------|-----------|------------------------------|
| Classifier une page (texte vs illustration vs mixte) | Classification | DINOv2 + sonde linéaire (section 3.1) |
| Localiser les régions de layout (colonnes, marges) | Détection / Seg. sémantique | SAM (section 3.3) |
| Isoler chaque ligne de texte | Segmentation d'instances | Kraken BLLA (section 4.2) |
| Décrire le contenu d'une illustration | Classification multi-label | CLIP (section 3.2) |
| Binariser les pixels d'encre | Seg. sémantique binaire | Sauvola (section 4.1) — pas de deep learning |

**Pourquoi SAM pour la segmentation de layout plutôt que Mask R-CNN ?**

Mask R-CNN nécessite des données annotées spécifiques au domaine (des milliers de pages de manuscrits annotées en boîtes et masques pour chaque type de région). SAM, entraîné sur 1,1 milliard de masques génériques, généralise sans annotation — au prix d'une absence de classification sémantique native. Pour un corpus de manuscrits très diversifié avec peu de données annotées, SAM + CLIP est plus pragmatique que Mask R-CNN entraîné de zéro.

**Pourquoi Kraken BLLA pour les lignes et non SAM ?**

Kraken BLLA est spécifiquement entraîné sur des documents patrimoniaux. Ses baselines et polygones sont mieux adaptés à la géométrie réelle des lignes de texte (courbures, ascendants, descendants) que les masques rectangulaires ou les polygones convexes que SAM produit naturellement sur les zones de texte dense.

---

## 7. Métriques d'évaluation spécifiques à chaque tâche

### 7.1 Détection : IoU, mAP

$$\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|} \in [0, 1]$$

La **mAP** (*mean Average Precision*) calcule l'aire sous la courbe précision-rappel pour chaque classe, à un seuil d'IoU donné, et moyenne sur les classes.

$$\text{mAP} = \frac{1}{C} \sum_{c=1}^C \text{AP}_c$$

La mAP@0.5 utilise IoU=0.5 comme seuil de correspondance. La mAP@0.5:0.95 moyenne sur des seuils de 0.5 à 0.95 par pas de 0.05 — plus exigeante.

### 7.2 Segmentation : mIoU, Dice

**mIoU** (*mean Intersection over Union*) : IoU moyen sur toutes les classes.

$$\text{mIoU} = \frac{1}{C} \sum_{c=1}^C \frac{\text{TP}_c}{\text{TP}_c + \text{FP}_c + \text{FN}_c}$$

**Coefficient de Dice** (équivalent au F1-score) :

$$\text{Dice} = \frac{2|\text{TP}|}{2|\text{TP}| + |\text{FP}| + |\text{FN}|} = \frac{2 \cdot |A \cap B|}{|A| + |B|}$$

Le Dice et l'IoU sont liés par : $\text{Dice} = \frac{2 \cdot \text{IoU}}{1 + \text{IoU}}$

```python
import numpy as np

def miou(predictions: np.ndarray, references: np.ndarray,
         n_classes: int) -> float:
    """
    Calcule le mIoU (mean Intersection over Union) pour la segmentation sémantique.

    Args:
        predictions, references : tableaux entiers (H, W) avec valeurs dans [0, n_classes-1]
    """
    ious = []
    for c in range(n_classes):
        pred_c = predictions == c
        ref_c  = references  == c
        intersection = (pred_c & ref_c).sum()
        union        = (pred_c | ref_c).sum()
        if union == 0:
            continue   # Classe absente : ignorer
        ious.append(intersection / union)
    return float(np.mean(ious)) if ious else 0.0


def dice_coefficient(pred_mask: np.ndarray, ref_mask: np.ndarray) -> float:
    """Coefficient de Dice pour un masque binaire."""
    intersection = (pred_mask & ref_mask).sum()
    total = pred_mask.sum() + ref_mask.sum()
    return float(2 * intersection / total) if total > 0 else 1.0
```

---

## Glossaire des termes avancés

**Anchor (boîte d'ancrage)**
Boîte de référence prédéfinie (taille et ratio fixes) utilisée dans les détecteurs comme Faster R-CNN ou YOLO. Le modèle prédit des corrections (*deltas*) par rapport à ces boîtes de référence. Un ensemble d'anchors couvre différentes tailles et ratios pour détecter des objets variés.

**ASPP (*Atrous Spatial Pyramid Pooling*)**
Module de DeepLab appliquant des convolutions dilatées à plusieurs taux $r$ en parallèle pour capturer un contexte multi-échelle sans réduire la résolution.

**Convolution dilatée (*atrous convolution*)**
Convolution avec espacement $r-1$ entre les éléments du noyau. Le taux de dilatation $r$ contrôle le champ réceptif sans augmenter le nombre de paramètres ni réduire la résolution.

**Dice coefficient**
Métrique de segmentation équivalente au F1-score : rapport entre deux fois l'intersection et la somme des cardinaux. Plus robuste que l'accuracy pixel quand les classes sont déséquilibrées.

**Encodeur-décodeur**
Paradigme architectural pour la segmentation : l'encodeur réduit progressivement la résolution spatiale en augmentant la profondeur sémantique ; le décodeur reconstruit la résolution originale. U-Net est l'exemple canonique.

**Focal Loss**
Variante de la cross-entropie qui down-pondère les exemples bien classifiés par le facteur $(1-p_t)^\gamma$. Conçue pour les détecteurs en une étape souffrant de déséquilibre classes foreground/background.

**FPN (*Feature Pyramid Network*)**
Architecture qui construit une pyramide de feature maps à plusieurs résolutions en combinant des features d'encodage descendantes (haute résolution, faible sémantique) avec des features montantes (basse résolution, haute sémantique). Permet la détection d'objets à toutes les échelles.

**GAP (*Global Average Pooling*)**
Opération qui calcule la moyenne spatiale de chaque canal d'une feature map, produisant un vecteur de dimension égale au nombre de canaux. Remplace les couches fully connected pour la classification, réduit le nombre de paramètres et permet des tailles d'entrée variables.

**Hungarian matching (algorithme hongrois)**
Algorithme de résolution du problème d'affectation minimale (attribuer $N$ prédictions à $M$ objets réels de façon à minimiser le coût global). Utilisé dans DETR pour associer les prédictions aux vérités terrain sans anchors ni NMS.

**IoU (*Intersection over Union*)**
Métrique de correspondance entre deux régions : rapport entre l'aire de leur intersection et l'aire de leur union. Seuil typique de 0.5 pour qu'une détection soit considérée correcte.

**mAP (*mean Average Precision*)**
Métrique standard pour la détection d'objets. Calcule l'aire sous la courbe précision-rappel pour chaque classe à un seuil IoU donné, puis moyenne sur les classes.

**mIoU (*mean Intersection over Union*)**
Métrique standard pour la segmentation sémantique. Calcule l'IoU pour chaque classe, puis moyenne sur les classes. Sensible aux classes rares.

**NMS (*Non-Maximum Suppression*)**
Algorithme d'élimination des détections redondantes : parmi des boîtes se chevauchant fortement (IoU > seuil), seule la plus confiante est retenue. Remplacé par le matching hongrois dans DETR.

**RoI Align**
Version améliorée du RoI Pooling pour Mask R-CNN. Utilise l'interpolation bilinéaire pour éviter les erreurs d'arrondi lors de l'alignement des régions candidates sur la feature map — crucial pour la précision des masques.

**RoI Pooling (*Region of Interest Pooling*)**
Opération extrayant une feature map de taille fixe pour une région candidate quelconque, par max-pooling dans une grille $H \times W$. Permet le partage du calcul de backbone entre toutes les régions détectées.

**RPN (*Region Proposal Network*)**
Réseau léger glissant sur la feature map pour proposer des régions candidates susceptibles de contenir des objets. Composant central de Faster R-CNN, entraîné conjointement avec le reste du détecteur.

**Segmentation panoptique**
Tâche de vision unifiant segmentation sémantique (pour les régions de fond, *stuff*) et segmentation d'instances (pour les objets dénombrables, *things*). Évaluée par la Panoptic Quality (PQ).

**Skip connection**
Connexion directe entre un niveau de l'encodeur et le niveau correspondant du décodeur dans une architecture encodeur-décodeur. Transmets l'information de localisation fine préservée dans les niveaux superficiels de l'encodeur.

**Stuff / Things**
Distinction de la segmentation panoptique. *Things* : catégories d'objets comptables (voitures, personnes, enluminures). *Stuff* : régions de fond amorphes (ciel, sol, parchemin). Les *things* reçoivent des masques d'instance, les *stuff* reçoivent des masques sémantiques.

---

## Bibliographie de référence

### Classification

- **LeCun, Y., Bottou, L., Bengio, Y., Haffner, P.** (1998). *Gradient-based learning applied to document recognition*. Proceedings of the IEEE, 86(11), 2278–2324. — LeNet : l'architecture fondatrice.

- **He, K., Zhang, X., Ren, S., Sun, J.** (2016). *Deep Residual Learning for Image Recognition*. CVPR 2016. [arXiv:1512.03385] — ResNet et les connexions résiduelles.

- **Tan, M., Le, Q.** (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*. ICML 2019. [arXiv:1905.11946] — Compound scaling.

- **Dosovitskiy, A. et al.** (2020). *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale*. ICLR 2021. [arXiv:2010.11929] — ViT : l'application des Transformers à la classification d'images.

### Détection d'objets

- **Girshick, R., Donahue, J., Darrell, T., Malik, J.** (2014). *Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation*. CVPR 2014. [arXiv:1311.2524] — R-CNN.

- **Girshick, R.** (2015). *Fast R-CNN*. ICCV 2015. [arXiv:1504.08083] — RoI Pooling, architecture accélérée.

- **Ren, S., He, K., Girshick, R., Sun, J.** (2016). *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks*. NeurIPS 2015. [arXiv:1506.01497] — **Article central sur le RPN, à lire impérativement.**

- **Redmon, J., Divvala, S., Girshick, R., Farhadi, A.** (2016). *You Only Look Once: Unified, Real-Time Object Detection*. CVPR 2016. [arXiv:1506.02640] — YOLO, détection en une étape.

- **Lin, T.-Y., Goyal, P., Girshick, R., He, K., Dollár, P.** (2017). *Focal Loss for Dense Object Detection*. ICCV 2017. [arXiv:1708.02002] — RetinaNet et Focal Loss.

- **Lin, T.-Y., Dollár, P., Girshick, R., He, K., Hariharan, B., Belongie, S.** (2017). *Feature Pyramid Networks for Object Detection*. CVPR 2017. [arXiv:1612.03144] — FPN : référence pour la détection multi-échelles.

- **Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., Zagoruyko, S.** (2020). *End-to-End Object Detection with Transformers (DETR)*. ECCV 2020. [arXiv:2005.12872] — Détection par Transformer sans anchors.

### Segmentation sémantique

- **Long, J., Shelhamer, E., Darrell, T.** (2015). *Fully Convolutional Networks for Semantic Segmentation*. CVPR 2015. [arXiv:1411.4038] — FCN : réseau entièrement convolutif pour la segmentation.

- **Ronneberger, O., Fischer, P., Brox, T.** (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI 2015. [arXiv:1505.04597] — **U-Net : architecture de référence pour la segmentation encoder-décodeur.**

- **Chen, L.-C., Papandreou, G., Kokkinos, I., Murphy, K., Yuille, A. L.** (2017). *DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs*. IEEE TPAMI, 40(4). [arXiv:1606.00915] — Convolutions dilatées et ASPP.

- **Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J. M., Luo, P.** (2021). *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers*. NeurIPS 2021. [arXiv:2105.15203] — SegFormer : Transformer pour la segmentation.

### Segmentation d'instances et panoptique

- **He, K., Gkioxari, G., Dollár, P., Girshick, R.** (2017). *Mask R-CNN*. ICCV 2017. [arXiv:1703.06870] — **Référence pour la segmentation d'instances ; introduit RoI Align.**

- **Wang, X., Kong, T., Shen, C., Jiang, Y., Li, L.** (2020). *SOLO: Segmenting Objects by Locations*. ECCV 2020. [arXiv:1912.04488] — Segmentation d'instances sans détection.

- **Kirillov, A., He, K., Girshick, R., Rother, C., Dollár, P.** (2019). *Panoptic Segmentation*. CVPR 2019. [arXiv:1801.00868] — Définition formelle de la tâche panoptique et de la métrique PQ.

- **Cheng, B., Misra, I., Schwing, A. G., Kirillov, A., Garg, R.** (2022). *Masked-attention Mask Transformer for Universal Image Segmentation (Mask2Former)*. CVPR 2022. [arXiv:2112.01527] — Unified architecture for all segmentation tasks.

### Applications aux documents

- **Shen, Z. et al.** (2021). *LayoutParser: A Unified Toolkit for Deep Learning Based Document Image Analysis*. ICDAR 2021. [arXiv:2103.15348] — Panorama des architectures de détection/segmentation appliquées aux documents.

- **Oliveira, S. A. et al.** (2021). *dhSegment-T: Contextualized Historical Document Segmentation*. ICDAR 2021. — U-Net adapté aux documents historiques.

- **Gruning, T. et al.** (2018). *A Two-Stage Method for Text Line Detection in Historical Documents*. IJDAR, 22(3). — Détection de lignes par méthodes de segmentation dans les documents patrimoniaux.

---

*Support de cours rédigé pour le module Computer Vision · Promotion MD5 · Mai 2026*
*Ce document constitue la section introductive 2.0 du Jour 2.*
*Il précède les sections 2.1 (CNN et limites) et 2.2 (mécanisme d'attention).*
*Durée estimée de lecture : 70–80 minutes.*
