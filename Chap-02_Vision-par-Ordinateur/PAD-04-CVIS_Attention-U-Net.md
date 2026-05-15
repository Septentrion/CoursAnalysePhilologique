# Cours 2.2b — Architecture U-Net : segmentation avec et sans attention

**Module 2 · Computer Vision appliquée aux manuscrits médiévaux · MD5**
**Section intercalée entre 2.2 (mécanisme d'attention) et 2.3 (TP)**

---

> *« La segmentation est une tâche de traduction : traduire chaque pixel d'une image en une décision. U-Net a compris avant tout le monde qu'une bonne traduction ne sacrifie pas l'original — elle le conserve intégralement le long du chemin, pour y revenir quand c'est nécessaire. »*

---

## Introduction : pourquoi U-Net mérite une section à part entière

La section 2.0 a présenté U-Net comme l'architecture de référence pour la segmentation sémantique, et la section 2.2 a établi les fondements du mécanisme d'attention. Cette section fait la synthèse : elle explique en détail *pourquoi* U-Net fonctionne si bien pour la segmentation, puis montre comment l'intégration de l'attention — mécanisme que nous venons d'étudier — l'améliore de façon qualitative et pas seulement quantitative.

Pour notre projet, U-Net est omniprésent :

- La binarisation des pixels d'encre (Sauvola est une heuristique — un U-Net appris fait mieux sur les parchemins dégradés).
- BLLA, le module de segmentation de Kraken, repose sur une architecture U-Net convolutif.
- dhSegment, l'outil standard de segmentation de layout pour les documents historiques, est un U-Net ResNet.
- La séparation texte/illustration dans les pages enluminées est une tâche de segmentation sémantique binaire — encore U-Net.

Comprendre U-Net en profondeur, c'est comprendre le fondement technique de plusieurs étapes clés du pipeline.

---

## 1. Le problème fondamental de la segmentation : localisation fine vs compréhension globale

### 1.1 La tension spatiale de la segmentation

Les réseaux de classification résolvent un problème simple : *compresser* une image en un vecteur d'information. Plus on descend dans le réseau, plus la résolution spatiale diminue et plus la sémantique augmente. Cette compression est une feature, pas un bug — elle rend le modèle invariant aux translations et robuste aux variations de position.

La segmentation a besoin du contraire : *préserver* la résolution spatiale jusqu'à la sortie pour pouvoir décider pixel par pixel. Un modèle qui a réduit une image de 512×512 à un tenseur 16×16 a perdu la résolution nécessaire pour distinguer les pixels d'un jambage d'ascendant de ceux de la ligne adjacente.

La tension fondamentale de la segmentation est donc :

```
Besoin 1 — Compréhension sémantique haute  → nécessite de la PROFONDEUR
                                              → nécessite de la COMPRESSION
                                              → tue la résolution spatiale

Besoin 2 — Localisation pixel précise       → nécessite de la RÉSOLUTION
                                              → nécessite de ne PAS compresser
                                              → empêche la profondeur sémantique
```

Les premières approches de segmentation (FCN, section 2.0) ont résolu cette tension de façon brutale : comprimer jusqu'à un bottleneck profond, puis upsampler directement. Les masques produits sont sémantiquement corrects mais spatialement flous — les bords des objets sont imprécis.

**U-Net résout cette tension différemment : conserver les deux, simultanément.**

### 1.2 L'idée clé : le passage latéral d'information

Au lieu de faire passer toute l'information par le bottleneck (goulot d'étranglement), U-Net maintient des **connexions latérales** entre l'encodeur et le décodeur à chaque niveau de résolution. L'information à haute résolution (précision spatiale) de l'encodeur est transmise directement au décodeur, sans passer par le bottleneck.

```
Information haute résolution (encodeur)  → directement → décodeur (même niveau)
Information sémantique profonde          → bottleneck  → décodeur (remonte)
```

Le décodeur n'a plus à « deviner » où se trouvent les bords à partir d'une représentation compressée — il a accès à l'information originale haute résolution. Son seul travail est d'utiliser la sémantique du bottleneck pour *sélectionner* ce qui, dans cette information haute résolution, est pertinent.

---

## 2. Architecture U-Net standard

### 2.1 Vue d'ensemble

U-Net (Ronneberger et al., 2015) doit son nom à sa forme caractéristique en U quand on représente les dimensions spatiales en abscisse et la profondeur du réseau en ordonnée :

```
Résolution  512    256    128    64     32   ← Encodeur (descente)
             │      │      │      │      │
             │      │      │      │ Bottleneck (32×32, profond)
             │      │      │      │      │
Résolution  512    256    128    64     32   ← Décodeur (montée)

             ↑      ↑      ↑      ↑          Skip connections (latérales)
```

Plus précisément, pour une image d'entrée de $H \times W$ pixels avec $C_{in}$ canaux :

```
INPUT  : (H, W, C_in)          ex : (512, 512, 1) — image de ligne en niveaux de gris

ENCODEUR :
  Bloc 1 → (H/2,  W/2,  64)   → Conv×2 + ReLU + BatchNorm + MaxPool
  Bloc 2 → (H/4,  W/4,  128)  → Conv×2 + ReLU + BatchNorm + MaxPool
  Bloc 3 → (H/8,  W/8,  256)  → Conv×2 + ReLU + BatchNorm + MaxPool
  Bloc 4 → (H/16, W/16, 512)  → Conv×2 + ReLU + BatchNorm + MaxPool

BOTTLENECK :
  Bloc 5 → (H/16, W/16, 1024) → Conv×2 + ReLU + BatchNorm  [pas de pool]

DÉCODEUR :
  Bloc 6 → (H/8,  W/8,  512)  → TranspConv ↑ + Concat(skip4) + Conv×2
  Bloc 7 → (H/4,  W/4,  256)  → TranspConv ↑ + Concat(skip3) + Conv×2
  Bloc 8 → (H/2,  W/2,  128)  → TranspConv ↑ + Concat(skip2) + Conv×2
  Bloc 9 → (H,    W,    64)   → TranspConv ↑ + Concat(skip1) + Conv×2

TÊTE :
  Conv 1×1 → (H, W, n_classes) → carte de classes pixel par pixel
```

### 2.2 Les composants essentiels

**Le double bloc convolutif**

La brique de base de U-Net est une séquence de deux convolutions 3×3, chacune suivie d'une normalisation et d'une activation :

```python
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """
    Double convolution 3×3 — brique élémentaire de U-Net.
    Deux convolutions successives permettent au modèle d'apprendre
    des interactions entre les features sans augmenter le champ réceptif
    autant qu'une convolution 5×5 (qui serait plus coûteuse).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int = None,
        dropout: float = 0.0
    ):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        couches = [
            nn.Conv2d(in_channels, mid_channels, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            couches.append(nn.Dropout2d(p=dropout))

        self.bloc = nn.Sequential(*couches)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bloc(x)
```

**L'encodeur (descente)**

```python
class BlocEncodeur(nn.Module):
    """
    Un bloc de l'encodeur U-Net : MaxPool + DoubleConv.
    Le MaxPool 2×2 divise la résolution par 2 et transmet le contexte
    spatial à grande échelle au bloc suivant.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)
```

**La convolution transposée (upsampling appris)**

La montée en résolution dans le décodeur peut se faire par deux approches :

1. **Upsampling bilinéaire + Conv 1×1** : simple, aucun paramètre supplémentaire, artefacts éventuels.
2. **Convolution transposée** (ou *déconvolution*) : upsampling appris, plus de paramètres, peut introduire des artefacts en damier (*checkerboard artifacts*).

```python
class BlocDecodeur(nn.Module):
    """
    Un bloc du décodeur U-Net :
    Upsampling × 2 + Concat(skip) + DoubleConv.

    La skip connection apporte l'information à haute résolution
    de l'encodeur correspondant — c'est ce qui fait la spécificité de U-Net.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mode_upsample: str = "bilinear"   # "bilinear" ou "transpose"
    ):
        super().__init__()

        if mode_upsample == "transpose":
            # Upsampling appris : ConvTranspose2d
            # in_channels/2 car le canal est divisé par 2 avant concat
            self.upsample = nn.ConvTranspose2d(
                in_channels, in_channels // 2,
                kernel_size=2, stride=2
            )
        else:
            # Upsampling bilinéaire + réduction de canal par Conv 1×1
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
            )

        # Double convolution après concaténation
        # Entrée : in_channels//2 (upsample) + in_channels//2 (skip) = in_channels
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(
        self,
        x: torch.Tensor,      # Feature map du niveau inférieur (basse résolution)
        skip: torch.Tensor    # Skip connection de l'encodeur (haute résolution)
    ) -> torch.Tensor:
        x = self.upsample(x)

        # Gérer les différences de taille dues aux arondis de MaxPool
        if x.shape != skip.shape:
            x = nn.functional.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=True
            )

        # Concaténation sur la dimension des canaux
        x = torch.cat([skip, x], dim=1)   # skip en premier : convention U-Net
        return self.conv(x)
```

### 2.3 U-Net complet

```python
class UNet(nn.Module):
    """
    U-Net complet (Ronneberger et al., 2015).

    Paramètres :
        in_channels  : canaux d'entrée (1=niveaux de gris, 3=RGB)
        n_classes    : nombre de classes de sortie
        features     : nombre de filtres à chaque niveau
        mode_upsample: stratégie d'upsampling du décodeur
        dropout      : taux de dropout dans les blocs convolutifs

    Exemple d'usage pour la binarisation d'une image de ligne :
        modele = UNet(in_channels=1, n_classes=2)  # fond vs encre
    """
    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = 2,
        features: list[int] = [64, 128, 256, 512],
        mode_upsample: str = "bilinear",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encodeurs = nn.ModuleList()
        self.decodeurs = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)

        # Premier bloc (pas de MaxPool en entrée)
        self.entree = DoubleConv(in_channels, features[0], dropout=dropout)

        # Blocs encodeurs
        for i in range(len(features) - 1):
            self.encodeurs.append(
                BlocEncodeur(features[i], features[i + 1])
            )

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2, dropout=dropout)

        # Blocs décodeurs (ordre inverse)
        for i in reversed(range(len(features))):
            in_ch = features[i] * 2 if i == len(features) - 1 else features[i + 1]
            self.decodeurs.append(
                BlocDecodeur(in_ch * 2 if i == len(features)-1 else features[i+1],
                             features[i],
                             mode_upsample=mode_upsample)
            )

        # Tête de classification pixel par pixel
        self.tete = nn.Conv2d(features[0], n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encodage + collecte des skip connections
        skips = [self.entree(x)]
        for enc in self.encodeurs:
            skips.append(enc(skips[-1]))

        # Bottleneck
        x = self.pool(skips[-1])
        x = self.bottleneck(x)

        # Décodage avec skip connections (ordre inverse)
        for dec, skip in zip(self.decodeurs, reversed(skips)):
            x = dec(x, skip)

        return self.tete(x)


# ── Exemple d'utilisation ──────────────────────────────────────────────────────
if __name__ == "__main__":
    modele = UNet(in_channels=1, n_classes=2, features=[64, 128, 256, 512])

    # Bilan des paramètres
    n_params = sum(p.numel() for p in modele.parameters())
    print(f"Paramètres U-Net : {n_params:,}")   # ~31M

    # Test de forme (le modèle doit être équivariant à la taille d'entrée)
    for taille in [(256, 256), (512, 512), (384, 768)]:
        x   = torch.randn(2, 1, *taille)          # batch=2
        out = modele(x)
        assert out.shape == (2, 2, *taille), \
            f"Forme incorrecte : {out.shape}"
    print("✓ Formes de sortie correctes pour toutes les tailles d'entrée")
```

### 2.4 La loss de segmentation

Pour la segmentation binaire (encre vs fond), on utilise une combinaison de Binary Cross-Entropy et de Dice Loss. La BCE seule pénalise chaque pixel indépendamment, ce qui la rend sensible aux déséquilibres de classes. Le Dice complète en mesurant le recouvrement global des masques.

```python
import torch.nn.functional as F

class LossSegmentation(nn.Module):
    """
    Combinaison BCE + Dice pour la segmentation.
    Adaptée aux cas où une classe est minoritaire (pixels d'encre
    représentent typiquement 10-20% d'une image de manuscrit binarisé).
    """
    def __init__(
        self,
        alpha: float = 0.5,   # Poids de la BCE
        beta: float  = 0.5,   # Poids du Dice
        smooth: float = 1e-6  # Éviter la division par zéro
    ):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.smooth = smooth

    def dice_loss(
        self,
        pred: torch.Tensor,   # (B, C, H, W) — logits
        target: torch.Tensor  # (B, H, W) — indices de classe
    ) -> torch.Tensor:
        # Convertir les logits en probabilités et aplatir spatialement
        pred_soft = F.softmax(pred, dim=1)
        n_classes = pred.shape[1]

        # Calcul du Dice pour chaque classe
        dice_total = 0.0
        for c in range(n_classes):
            pred_c   = pred_soft[:, c].flatten(1)      # (B, H*W)
            target_c = (target == c).float().flatten(1) # (B, H*W)
            intersection = (pred_c * target_c).sum(1)
            dice_c = (2 * intersection + self.smooth) / \
                     (pred_c.sum(1) + target_c.sum(1) + self.smooth)
            dice_total += dice_c.mean()

        return 1.0 - dice_total / n_classes

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        bce  = F.cross_entropy(pred, target)
        dice = self.dice_loss(pred, target)
        return self.alpha * bce + self.beta * dice
```

---

## 3. Pourquoi U-Net est particulièrement adapté aux manuscrits médiévaux

### 3.1 La hiérarchie des features d'un scan de manuscrit

Pour comprendre pourquoi U-Net excelle sur les documents historiques, il faut comprendre *ce que* chaque niveau d'encodage capture.

**Niveau 1 (haute résolution, 512×512) :** textures locales — la rugosité du parchemin, le grain de l'encre, les micro-variations de couleur.

**Niveau 2 (256×256) :** contours des traits — les bords des jambages, les zones de transition encre/fond.

**Niveau 3 (128×128) :** formes locales — les ascendants, les boucles, les ligatures.

**Niveau 4 (64×64) :** patterns de mots et de groupes de lettres — l'espacement inter-mot, les zones denses.

**Bottleneck (32×32) :** structure globale de la page — la présence d'une illustration, la densité d'une colonne, la position des marges.

Pour la segmentation de lignes, **tous ces niveaux sont nécessaires** :
- Le bottleneck décide si une région est du texte ou une illustration.
- Les niveaux intermédiaires localisent les lignes.
- Les niveaux hauts résolution précisent les contours exacts.

Un réseau sans skip connections doit reconstruire cette hiérarchie depuis le bottleneck seul — ce qu'il fait de façon approximative. U-Net conserve chaque niveau et le restitue directement là où il est utile.

### 3.2 La robustesse aux données limitées

U-Net a été conçu pour la segmentation d'images biomédicales, où les données annotées sont rares et coûteuses — exactement comme dans notre contexte de manuscrits médiévaux. Son article fondateur montre qu'il peut produire des résultats compétitifs avec seulement quelques dizaines d'images annotées, grâce à deux propriétés :

**Les skip connections réduisent le nombre de paramètres à apprendre.** L'information de localisation ne doit pas être apprise depuis zéro dans le décodeur — elle est transmise directement. Le décodeur apprend seulement à *filtrer* l'information reçue, pas à la *reconstruire*.

**L'augmentation de données élastique.** L'article original démontre qu'une augmentation par déformations élastiques aléatoires (section 5.2 de ce cours) est particulièrement efficace sur des textures comme les tissus biologiques — et, par analogie, les traits d'écriture manuscrite.

### 3.3 L'équivariance spatiale

U-Net est un réseau entièrement convolutif (*Fully Convolutional Network*) — il ne contient aucune couche fully connected. Cela lui confère une propriété essentielle pour les documents : **l'équivariance à la translation**. Le même pattern visuel (un trait d'encre, un espace inter-ligne) est reconnu de la même façon qu'il apparaisse en haut ou en bas, à gauche ou à droite de l'image.

De plus, l'absence de couches fully connected signifie que U-Net peut traiter des images de **taille arbitraire** — il suffit que les dimensions soient divisibles par $2^N$ (le nombre de niveaux de pooling). Pour des scans de manuscrits de taille variable (une charte royale n'a pas les mêmes dimensions qu'un feuillet de roman arthurien), c'est une propriété précieuse.

---

## 4. Attention U-Net : le gating spatial

### 4.1 Le problème des skip connections non filtrées

Dans U-Net standard, chaque skip connection transmet *toutes* les features du niveau correspondant de l'encodeur. Pour une image de ligne de manuscrit avec une tache d'humidité au coin supérieur droit, les features de ce coin sont transmises fidèlement au décodeur — qui doit alors apprendre à les ignorer pendant la segmentation de l'encre.

Le décodeur réussit souvent à filtrer ce bruit contextuel, mais pas toujours. Sur des images très dégradées (show-through, taches, encre corrodée), les skip connections non filtrées peuvent **propager du bruit** vers la prédiction finale.

**Attention U-Net** (Oktay et al., 2018) répond à ce problème par un **mécanisme de gating** sur les skip connections : avant d'être concaténées, les features de l'encodeur sont modifiées par un score d'attention qui dépend à la fois du contenu local de l'encodeur *et* du contexte sémantique du décodeur.

### 4.2 Le mécanisme d'Attention Gate

Pour chaque pixel $(i, j)$ de la skip connection, l'Attention Gate calcule un score $\alpha_{ij} \in [0, 1]$ qui pondère la contribution de ce pixel. Un score proche de 1 signifie que ce pixel est pertinent pour la tâche de segmentation courante ; proche de 0, il doit être ignoré.

**Les entrées du gate :**

- $\mathbf{g} \in \mathbb{R}^{H' \times W' \times C_g}$ : le vecteur de gating, issu du décodeur (niveau inférieur, basse résolution, haute sémantique). Il encode le *contexte* — « que suis-je en train de segmenter ? »
- $\mathbf{x} \in \mathbb{R}^{H \times W \times C_x}$ : les features de la skip connection (encodeur, haute résolution, faible sémantique). Il encode le *contenu* à filtrer.

**Le calcul du score d'attention :**

$$\alpha_{ij} = \sigma\left(W_\psi \cdot \text{ReLU}\left(W_g \mathbf{g}_{ij} + W_x \mathbf{x}_{ij} + b\right)\right)$$

où $W_g \in \mathbb{R}^{C_{int} \times C_g}$, $W_x \in \mathbb{R}^{C_{int} \times C_x}$, $W_\psi \in \mathbb{R}^{1 \times C_{int}}$ sont des convolutions 1×1 apprenantes, $\sigma$ est la sigmoïde et $C_{int}$ est la dimension intermédiaire (typiquement $C_{int} = C_x / 2$).

**La sortie filtrée :**

$$\hat{\mathbf{x}}_{ij} = \alpha_{ij} \cdot \mathbf{x}_{ij}$$

Les features de la skip connection sont pondérées pixel par pixel par leur pertinence contextuelle — les pixels non pertinents sont atténués avant la concaténation.

```python
class AttentionGate(nn.Module):
    """
    Attention Gate pour Attention U-Net (Oktay et al., 2018).

    Sélectionne les features de la skip connection (x) en fonction
    du contexte sémantique du décodeur (g).

    Args:
        F_g   : nombre de canaux du vecteur de gating (décodeur)
        F_x   : nombre de canaux de la skip connection (encodeur)
        F_int : dimension intermédiaire (typiquement F_x // 2)
    """
    def __init__(self, F_g: int, F_x: int, F_int: int):
        super().__init__()

        # Projection du vecteur de gating
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        # Projection de la skip connection
        self.W_x = nn.Sequential(
            nn.Conv2d(F_x, F_int, kernel_size=1, stride=1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        # Calcul du score scalaire par pixel
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(
        self,
        g: torch.Tensor,   # Vecteur de gating (décodeur, basse résolution)
        x: torch.Tensor    # Skip connection (encodeur, haute résolution)
    ) -> torch.Tensor:
        # Aligner les résolutions si nécessaire
        g_up = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=True)

        # Calcul du score d'attention
        att = self.relu(self.W_g(g_up) + self.W_x(x))
        att = self.psi(att)   # Score ∈ [0, 1] par pixel — (B, 1, H, W)

        # Pondération de la skip connection
        return att * x, att   # Retourner aussi att pour visualisation


class BlocDecodeurAttention(nn.Module):
    """
    Bloc décodeur avec Attention Gate.
    Remplace le BlocDecodeur standard dans Attention U-Net.
    """
    def __init__(
        self,
        F_g: int,    # Canaux du vecteur de gating (sortie du niveau inférieur)
        F_x: int,    # Canaux de la skip connection
        F_out: int,  # Canaux de sortie
        mode_upsample: str = "bilinear"
    ):
        super().__init__()
        self.attention = AttentionGate(F_g=F_g, F_x=F_x, F_int=F_x // 2)

        if mode_upsample == "transpose":
            self.upsample = nn.ConvTranspose2d(F_g, F_g // 2, kernel_size=2, stride=2)
        else:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(F_g, F_g // 2, kernel_size=1)
            )

        # Après concat : F_g//2 (upsample) + F_x (skip filtré)
        self.conv = DoubleConv(F_g // 2 + F_x, F_out)

    def forward(
        self,
        g: torch.Tensor,   # Feature map décodeur (entrée du bloc)
        x: torch.Tensor    # Skip connection encodeur
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Upsampling du vecteur de gating
        g_up = self.upsample(g)

        # Filtrage de la skip connection par l'attention
        x_att, att_map = self.attention(g, x)

        # Concaténation et convolution
        out = self.conv(torch.cat([g_up, x_att], dim=1))
        return out, att_map   # att_map pour visualisation des zones d'attention
```

### 4.3 Visualisation des cartes d'attention

La carte d'attention $\alpha_{ij}$ est directement interprétable : elle indique quels pixels de la skip connection ont été jugés pertinents par le gate contextuel. Sur des images de manuscrits, cela permet de visualiser exactement ce sur quoi le modèle se concentre.

```python
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def visualiser_attention_unet(
    modele_att_unet,
    image_pil: Image.Image,
    niveau: int = 1,    # 0=plus proche entrée, 3=plus proche bottleneck
    nom_figure: str = "attention_unet"
) -> None:
    """
    Visualise la carte d'attention d'un Attention U-Net à un niveau donné.

    Args:
        modele_att_unet : instance d'Attention U-Net avec accès aux att_maps
        image_pil       : image de page de manuscrit
        niveau          : niveau du décodeur à visualiser (0–3)
    """
    # Prétraitement
    img_np = np.array(image_pil.convert("L")) / 255.0
    x = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    modele_att_unet.eval()
    with torch.no_grad():
        pred, att_maps = modele_att_unet(x, return_attention=True)

    att = att_maps[niveau].squeeze().cpu().numpy()   # (H_level, W_level)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"Attention U-Net — Niveau décodeur {niveau+1}",
        fontsize=12, fontweight="bold"
    )

    # Image originale
    axes[0].imshow(image_pil, cmap="gray")
    axes[0].set_title("Image originale", fontsize=10)
    axes[0].axis("off")

    # Carte d'attention
    im = axes[1].imshow(att, cmap="hot", vmin=0, vmax=1, interpolation="bilinear")
    axes[1].set_title(
        f"Carte d'attention α (niveau {niveau+1})\n"
        f"Résolution : {att.shape[0]}×{att.shape[1]} px",
        fontsize=10
    )
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    # Superposition attention × image
    att_resized = np.array(
        Image.fromarray((att * 255).astype(np.uint8)).resize(
            image_pil.size, Image.BILINEAR
        )
    ) / 255.0
    img_arr = np.array(image_pil.convert("RGB")) / 255.0
    overlay = img_arr * 0.5 + plt.cm.hot(att_resized)[:, :, :3] * 0.5

    axes[2].imshow(overlay)
    axes[2].set_title("Superposition attention × image\n"
                      "(zones chaudes = pertinentes pour la segmentation)",
                      fontsize=10)
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(f"figures/{nom_figure}_niveau{niveau}.png",
                dpi=150, bbox_inches="tight")
    plt.show()
```

---

## 5. Variantes modernes : U-Net avec Transformers

### 5.1 TransUNet : CNN encoder + Transformer bottleneck

**TransUNet** (Chen et al., 2021) remplace le bottleneck convolutif par un encodeur Transformer. L'idée : le bottleneck est exactement l'endroit où l'information globale est nécessaire — et les Transformers excellent à capturer des dépendances globales par self-attention.

```
Image
  ↓
[ CNN Encoder (ResNet) ]    → Features à stride 4, 8, 16
       ↓                           ↓↓↓ (skip connections)
[ Patch Embedding ]
       ↓
[ Transformer Encoder ]     → Self-attention globale sur les patches
       ↓
[ Cascade Upsampling ]      ← Récupère les skip connections CNN
       ↓
Carte de segmentation
```

L'avantage : le CNN encoder extrait des features locales robustes aux transformations géométriques, pendant que le Transformer au bottleneck intègre l'information globale de la page entière — les deux forces exploitées simultanément.

### 5.2 Swin U-Net : Transformers à tous les niveaux

**Swin U-Net** (Cao et al., 2022) remplace *tous* les blocs convolutifs (encodeur, bottleneck, décodeur) par des blocs Swin Transformer (attention par fenêtres décalées). La skip connection reste, mais connecte des représentations Transformer à des représentations Transformer.

```python
# Schéma conceptuel de Swin U-Net
# (Les blocs Transformer remplacent les DoubleConv à chaque niveau)

# Encodeur Swin :
# Patch Embedding (4×4) → SwinBlock (W-MSA) → Patch Merging ×3
#
# Bottleneck Swin :
# SwinBlock (W-MSA) × 2
#
# Décodeur Swin :
# Patch Expanding ↑ → Concat(skip) → SwinBlock (W-MSA)
#
# Avantage : champ réceptif global dès les premières couches
# Coût : plus de paramètres, entraînement plus lent
```

**SegFormer** (Xie et al., 2021) propose une vision plus légère : un encodeur Transformer hiérarchique (MiT) et un décodeur MLP minimal. Il atteint les performances de Swin U-Net avec moins de paramètres et moins de mémoire.

### 5.3 Tableau comparatif des variantes

| Architecture | Encoder | Bottleneck | Decoder | Paramètres | mIoU ADE20k |
|-------------|---------|------------|---------|------------|-------------|
| U-Net | CNN | CNN | CNN | ~31 M | ~37% |
| Attention U-Net | CNN + AG | CNN | CNN | ~35 M | ~40% |
| TransUNet | CNN (ResNet) | Transformer | CNN | ~105 M | ~44% |
| Swin U-Net | Transformer | Transformer | Transformer | ~41 M | ~44% |
| SegFormer-B5 | MiT (Transformer) | — | MLP | ~82 M | ~51% |

*mIoU sur ADE20k (segmentation sémantique générale) — les classements relatifs sont indicatifs du potentiel architectural, pas directement transposables aux manuscrits médiévaux.*

---

## 6. Application aux manuscrits médiévaux : quel modèle pour quelle tâche ?

### 6.1 Binarisation adaptative par U-Net

La binarisation par seuillage de Sauvola (section 4.1) est une heuristique non apprise. Un U-Net entraîné sur des paires (scan couleur, masque binaire de référence) peut apprendre à gérer les cas difficiles — show-through, encre pâlie, fond hétérogène — que Sauvola traite de façon sous-optimale.

```python
# Configuration recommandée pour la binarisation par U-Net
modele_binarisation = UNet(
    in_channels=3,   # Entrée RGB : exploiter l'information de couleur
                     # (l'encre ferro-gallique brunie a un spectre distinct du fond)
    n_classes=2,     # Fond (0) vs encre (1)
    features=[32, 64, 128, 256],   # Plus léger qu'un U-Net standard
    dropout=0.2      # Régularisation plus forte : données limitées
)

# Données d'entraînement : paires (scan RGB, masque binaire DIBCO)
# Référence benchmark : DIBCO (Document Image Binarization Contest)
# CER HTR attendu : amélioration de 1-3 points sur parchemins dégradés
```

### 6.2 Segmentation de layout : U-Net pour séparer texte et illustrations

Pour séparer les colonnes de texte des enluminures et des marges, un U-Net entraîné sur des pages annotées avec l'ontologie SegmOnto est plus précis que SAM sur les pages à mise en page standard.

```python
# Configuration pour la segmentation de layout selon SegmOnto
CLASSES_SEGMONTO = [
    "MainZone",           # Colonne de texte principal
    "MarginTextZone",     # Zone de texte marginal
    "DropCapitalZone",    # Zone de lettrine ornée
    "GraphicZone",        # Illustration, enluminure
    "TitlePageZone",      # Titre de page, rubrique
    "NumberingZone",      # Numérotation de folio
    "Background",         # Fond vide
]

modele_layout = AttentionUNet(
    in_channels=3,
    n_classes=len(CLASSES_SEGMONTO),
    features=[64, 128, 256, 512]
)
# L'Attention U-Net est préféré ici : il peut apprendre à distinguer
# les zones de texte dense (attention forte) des zones de fond (attention faible)
# sans être perturbé par les ornements et dégradations dans les zones de fond.
```

### 6.3 BLLA de Kraken : un U-Net pour les baselines

Le module BLLA de Kraken que nous utilisons en section 4.2 est lui-même un réseau de type U-Net convolutif — entraîné spécifiquement pour détecter les baselines et les polygones englobants dans les documents patrimoniaux.

Sa tâche de sortie est légèrement différente d'une segmentation sémantique standard : il produit une carte de pixels de baselines (les pixels qui font partie d'une ligne de base) et une carte de régions (les pixels appartenant à chaque type de zone). L'architecture U-Net est bien adaptée car :

- Les baselines sont des structures localement fines (quelques pixels d'épaisseur) qui nécessitent une résolution haute → skip connections nécessaires.
- La détection des baselines dépend du contexte global de la page (nombre de colonnes, présence d'illustrations) → bottleneck sémantique nécessaire.

---

## 7. Entraînement pratique d'un U-Net sur des données de manuscrits

### 7.1 Initialisation et transfert

Entraîner un U-Net de zéro sur des données de manuscrits (< 1000 images annotées) conduit souvent à l'overfitting. Deux stratégies d'initialisation améliorent significativement les résultats :

**Transfert depuis ImageNet (encodeur pré-entraîné) :**

```python
import torchvision.models as models

# Encodeur ResNet-34 pré-entraîné comme backbone U-Net
# (segmentation-models-pytorch facilite cette combinaison)
import segmentation_models_pytorch as smp

modele = smp.Unet(
    encoder_name="resnet34",           # Encodeur pré-entraîné
    encoder_weights="imagenet",        # Poids ImageNet
    in_channels=3,
    classes=len(CLASSES_SEGMONTO),
    activation=None,                   # Logits bruts (softmax dans la loss)
)

# Seul l'encodeur est pré-entraîné ; le décodeur est initialisé aléatoirement
# Stratégie recommandée :
# Étape 1 (5 epochs) : geler l'encodeur, entraîner uniquement le décodeur
# Étape 2 (20 epochs) : dégeler tout, LR plus faible pour l'encodeur
for param in modele.encoder.parameters():
    param.requires_grad_(False)   # Geler l'encodeur en étape 1
```

**Initialisation par He (Kaiming) :**

```python
def initialiser_poids(modele: nn.Module) -> None:
    """
    Initialisation de He pour les couches convolutives.
    Recommandée pour les réseaux avec ReLU.
    """
    for m in modele.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
```

### 7.2 Stratégie d'entraînement progressive

```python
def schedule_lr_unet(
    optimizer,
    epoch: int,
    n_epochs_gele: int = 5,
    lr_encodeur: float = 1e-5,
    lr_decodeur: float = 1e-4
) -> None:
    """
    Schedule d'entraînement progressif :
    - Epochs 0 à n_epochs_gele : encoder gelé, LR élevé sur décodeur
    - Epochs n_epochs_gele+ : encoder dégelé, LR différenciés
    """
    for param_group in optimizer.param_groups:
        if param_group["name"] == "encodeur":
            param_group["lr"] = 0.0 if epoch < n_epochs_gele else lr_encodeur
        elif param_group["name"] == "decodeur":
            param_group["lr"] = lr_decodeur
```

---

## Glossaire des termes avancés

**Attention Gate (AG)**
Mécanisme de filtrage des skip connections dans Attention U-Net. Pour chaque pixel, calcule un score d'attention $\alpha_{ij} \in [0,1]$ en combinant le contenu local de la skip connection et le contexte sémantique du décodeur. Atténue les features non pertinentes avant la concaténation.

**BatchNorm (*Batch Normalization*)**
Normalisation des activations sur le batch à chaque couche : soustrait la moyenne et divise par l'écart-type, puis applique deux paramètres apprenants (échelle et biais). Accélère l'entraînement, réduit la sensibilité à l'initialisation, permet des LR plus élevés.

**Bottleneck (goulot d'étranglement)**
Le niveau le plus profond et le moins résolu du U-Net, après toutes les opérations de pooling. Concentre l'information sémantique globale de l'image. Sa petite taille spatiale force la représentation à capturer le contenu général plutôt que les détails locaux.

**Convolution transposée (*transposed convolution*, *deconvolution*)**
Opération d'upsampling appris : augmente la résolution spatiale d'un facteur 2 (pour stride=2) en appliquant un filtre de façon « inversée ». Peut produire des artefacts en damier (*checkerboard artifacts*) si mal initialisée. L'alternative — upsampling bilinéaire + Conv 1×1 — est souvent préférée en pratique.

**Dice Loss**
Fonction de perte pour la segmentation basée sur le coefficient de Dice. Mesure le recouvrement entre le masque prédit et le masque de vérité terrain. Robuste aux déséquilibres de classes (quand les pixels d'une classe sont rares). Souvent combinée à la BCE pour équilibrer précision locale et recouvrement global.

**dhSegment**
Outil de segmentation de layout pour documents historiques basé sur un U-Net avec encodeur ResNet. Développé à l'EPFL. Standard de facto pour l'analyse de layout dans les projets de numérisation patrimoniale.

**Équivariance à la translation**
Propriété d'un réseau convolutif : si l'entrée est translatée, la sortie est translatée du même vecteur. Assure que le réseau reconnaît le même pattern quelle que soit sa position dans l'image.

**FCN (*Fully Convolutional Network*)**
Réseau de neurones sans couches fully connected. Peut traiter des images de taille arbitraire. Précurseur de U-Net pour la segmentation sémantique.

**MaxPool**
Opération de downsampling : pour chaque fenêtre spatiale de taille $k \times k$, retient la valeur maximale. Réduit la résolution d'un facteur $k$ et introduit une invariance locale à de petites translations. Plus robuste au bruit que l'average pooling pour les features d'encre.

**SegmOnto**
Ontologie de segmentation de layout pour les documents patrimoniaux, définissant un vocabulaire contrôlé de types de zones (MainZone, MarginTextZone, GraphicZone, etc.). Standard adopté par HTR-United et eScriptorium pour les annotations de layout des manuscrits médiévaux.

**Skip connection**
Connexion directe entre un niveau de l'encodeur et le niveau correspondant du décodeur dans U-Net. Transmet l'information à haute résolution (détails fins, localisation précise) sans passer par le bottleneck. Clé de la précision de segmentation de U-Net.

**Swin Transformer**
Vision Transformer à attention par fenêtres décalées (*Shifted Window*), produisant des features hiérarchiques (multi-échelle). Utilisé comme encoder dans Swin U-Net pour combiner l'auto-attention globale avec une complexité computationnelle sous-quadratique.

**TransUNet**
Variante de U-Net combinant un encodeur CNN (ResNet) et un bottleneck Transformer. Le CNN extrait des features locales robustes ; le Transformer intègre l'information globale au bottleneck.

---

## Bibliographie de référence

### Articles fondateurs

- **Ronneberger, O., Fischer, P., Brox, T.** (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI 2015. [arXiv:1505.04597] — **L'article fondateur. À lire intégralement, notamment les sections 2 (architecture) et 3 (données et augmentation). Environ 80 000 citations.**

- **Oktay, O., Schlemper, J., Le Folgoc, L., Lee, M., Heinrich, M., Misawa, K., … Rueckert, D.** (2018). *Attention U-Net: Learning Where to Look for the Pancreas*. MIDL 2018. [arXiv:1804.03999] — Attention U-Net : le mécanisme d'Attention Gate.

### Variantes Transformer

- **Chen, J., Lu, Y., Yu, Q., Luo, X., Adeli, E., Wang, Y., … Zhou, Y.** (2021). *TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation*. [arXiv:2102.04306] — TransUNet : hybride CNN + Transformer.

- **Cao, H., Wang, Y., Chen, J., Jiang, D., Zhang, X., Tian, Q., Wang, M.** (2022). *Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation*. ECCV Workshops 2022. [arXiv:2105.05537] — Swin U-Net : architecture Transformer pure en U.

- **Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J. M., Luo, P.** (2021). *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers*. NeurIPS 2021. [arXiv:2105.15203] — SegFormer : alternative légère et performante.

### Applications aux documents historiques

- **Oliveira, S. A., Seguin, B., Kaplan, F.** (2018). *dhSegment: A generic deep-learning approach for document segmentation*. ICFHR 2018. [arXiv:1804.10371] — dhSegment : U-Net avec encodeur ResNet appliqué aux documents historiques. Référence pratique pour la segmentation de layout.

- **Gruning, T., Leifert, G., Strauß, T., Michael, J., Labahn, R.** (2019). *A Two-Stage Method for Text Line Detection in Historical Documents*. IJDAR, 22(3). — Détection de lignes par segmentation dans les documents patrimoniaux, base du BLLA de Kraken.

- **Gabay, S., Clérice, T., Camps, J.-B., Pinche, A., Chagué, A.** (2021). *Segmonto: Ontology for Page Segmentation of Medieval and Early Modern Documents*. [hal-03336528] — L'ontologie SegmOnto utilisée dans les annotations de layout pour les manuscrits médiévaux.

### Loss functions et évaluation

- **Milletari, F., Navab, N., Ahmadi, S.-A.** (2016). *V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation*. 3DV 2016. [arXiv:1606.04797] — Dice Loss appliquée à la segmentation : article fondateur de cette loss dans le contexte médical.

- **Lin, T.-Y., Goyal, P., Girshick, R., He, K., Dollár, P.** (2017). *Focal Loss for Dense Object Detection*. ICCV 2017. [arXiv:1708.02002] — Focal Loss : adaptable à la segmentation pour les classes très déséquilibrées.

### Initialisation et entraînement

- **He, K., Zhang, X., Ren, S., Sun, J.** (2015). *Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification*. ICCV 2015. [arXiv:1502.01852] — Initialisation de He (Kaiming) pour les réseaux avec ReLU.

- **Isensee, F., Jaeger, P. F., Kohl, S. A. A., Petersen, J., Maier-Hein, K. H.** (2021). *nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation*. Nature Methods, 18, 203–211. — Recettes pratiques d'entraînement U-Net sur des données médicales limitées — directement transposables aux manuscrits.

---

*Support de cours rédigé pour le module Computer Vision · Promotion MD5 · Mai 2026*
*Ce document constitue la section 2.2b du Jour 2, intercalée entre la section 2.2 (mécanisme d'attention) et le TP 2.3.*
*Il suppose la lecture des sections 2.0 (panorama architectures) et 2.2 (attention).*
*Durée estimée de lecture : 70–80 minutes.*
