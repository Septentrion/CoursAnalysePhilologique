# Cours 2.2c — DeepLabv3 et les convolutions dilatées pour la segmentation

**Module 2 · Computer Vision appliquée aux manuscrits médiévaux · MD5**
**Section intercalée entre 2.2b (U-Net) et 2.3 (TP)**

---

> *« U-Net résout le problème de résolution en conservant l'information à chaque niveau et en la restituant par des connexions latérales. DeepLab le résout autrement : en refusant de perdre la résolution dès le début. Ces deux philosophies ne s'opposent pas — elles se complètent, et leur fusion dans DeepLabv3+ donne l'une des architectures de segmentation les plus puissantes disponibles. »*

---

## Introduction : deux philosophies face au même problème

La section 2.2b a montré comment U-Net répond à la tension entre compréhension sémantique et localisation précise : conserver les deux flux en parallèle et les réunir par des skip connections dans le décodeur.

La famille **DeepLab** (Chen et al., 2015–2018) adopte une stratégie fondamentalement différente : **ne pas perdre la résolution en premier lieu**. Plutôt que de la comprimer puis de tenter de la reconstruire, DeepLab maintient une résolution spatiale élevée tout au long du réseau en remplaçant certains poolings par des **convolutions dilatées** — des convolutions à champ réceptif élargi sans réduction de résolution.

Ces deux philosophies correspondent à deux façons de résoudre la même équation :

```
U-Net    : compresser → bottleneck sémantique → reconstruire par skip connections
DeepLab  : ne pas compresser → élargir le champ réceptif sans réduire la résolution
```

Pour les manuscrits médiévaux, chaque approche a ses avantages :
- U-Net excelle quand les structures à segmenter ont des formes irrégulières avec des bords fins à préciser (lignes de texte, polygones de baselines).
- DeepLab excelle quand le contexte multi-échelle est déterminant (distinguer une lettrine du texte courant, séparer une illustration d'une zone de rubrique), grâce à son module ASPP.

Comprendre DeepLab, c'est comprendre l'outil complémentaire de U-Net dans l'arsenal de la segmentation profonde.

---

## 1. Le problème de la dégradation de résolution dans les CNN

### 1.1 Le stride et le pooling : ennemis de la résolution

Un réseau CNN classique (VGG-16, ResNet-50) réduit progressivement la résolution spatiale au fil des couches par deux mécanismes :

- **MaxPooling 2×2** : divise la résolution par 2 après chaque bloc.
- **Strided convolution** : convolution avec un pas (*stride*) > 1, qui réduit la résolution tout en calculant les features.

Pour un ResNet-50 en entrée d'un classificateur, la résolution finale avant le Global Average Pooling est divisée par **32** par rapport à l'entrée. Pour une image de 512×512, la feature map finale est 16×16 — 1 024 fois moins de positions spatiales.

**Pour la classification, ce n'est pas un problème :** l'objet à reconnaître occupe la majorité de l'image, et la précision spatiale de localisation n'est pas requise.

**Pour la segmentation sémantique, c'est catastrophique :** un prédicteur de classe opérant sur une feature map 16×16 ne peut pas produire de frontières précises à la résolution originale. Upsampler brutalement de 16×16 à 512×512 (facteur 32) produit des cartes de segmentation grossièrement pixelisées.

### 1.2 La tentative FCN : upsampling avec skip connections

Les FCN (section 2.0) ont partiellement résolu ce problème par des skip connections depuis les niveaux intermédiaires (stride 8 et stride 16). Mais l'upsampling reste un processus de reconstruction — on essaie de retrouver une précision perdue.

**DeepLab propose une question différente :** et si on ne réduisait pas autant la résolution ? Peut-on conserver les features à stride 8 (et non 32) tout en maintenant un champ réceptif aussi large que nécessaire pour la compréhension sémantique ?

La réponse est oui — grâce aux convolutions dilatées.

---

## 2. La convolution dilatée (*atrous convolution*)

### 2.1 Définition formelle

La **convolution dilatée** (ou *atrous convolution*, de l'anglais *à trous*) étend une convolution standard en insérant des espaces entre les éléments du noyau. Le paramètre $r$ — le **taux de dilatation** (*dilation rate*) — définit le nombre de pixels d'espacement :

$$y[i] = \sum_{k=-K}^{K} x[i + r \cdot k] \cdot w[k]$$

Pour un noyau $3 \times 3$ de taille $k \in \{-1, 0, 1\}$ :

| Taux $r$ | Position des poids dans la fenêtre d'entrée | Champ réceptif effectif |
|---------|---------------------------------------------|------------------------|
| $r=1$ | ⬛⬜⬛ (convolution standard) | $3 \times 3$ |
| $r=2$ | ⬛⬜⬜⬜⬛ (1 pixel d'espacement) | $5 \times 5$ |
| $r=4$ | ⬛⬜⬜⬜⬜⬜⬜⬜⬛ (3 pixels d'espacement) | $9 \times 9$ |
| $r=8$ | Champ réceptif effectif | $17 \times 17$ |
| $r=16$ | Champ réceptif effectif | $33 \times 33$ |

**Propriété essentielle :** quelle que soit la valeur de $r$, un noyau $3 \times 3$ dilaté conserve exactement **9 paramètres**. On obtient un champ réceptif croissant sans augmenter le nombre de paramètres ni réduire la résolution de sortie.

```python
import torch
import torch.nn as nn

# Comparaison entre convolutions standard et dilatées
# Même nombre de paramètres, champs réceptifs différents

# Convolution standard 3×3 — champ réceptif effectif : 3×3
conv_standard = nn.Conv2d(
    in_channels=64, out_channels=64,
    kernel_size=3, padding=1, dilation=1,
    bias=False
)

# Convolution dilatée, r=2 — champ réceptif effectif : 5×5
# padding=2 pour maintenir la résolution de sortie (H×W)
conv_r2 = nn.Conv2d(
    in_channels=64, out_channels=64,
    kernel_size=3, padding=2, dilation=2,
    bias=False
)

# Convolution dilatée, r=4 — champ réceptif effectif : 9×9
conv_r4 = nn.Conv2d(
    in_channels=64, out_channels=64,
    kernel_size=3, padding=4, dilation=4,
    bias=False
)

# Vérification : même nombre de paramètres dans tous les cas
n_params = lambda conv: sum(p.numel() for p in conv.parameters())
assert n_params(conv_standard) == n_params(conv_r2) == n_params(conv_r4)
print(f"Paramètres par convolution : {n_params(conv_standard)}")   # 9×64×64 = 36 864

# Test de forme : résolution d'entrée conservée
x = torch.randn(2, 64, 64, 64)   # (batch, canaux, H, W)
for conv, r in [(conv_standard, 1), (conv_r2, 2), (conv_r4, 4)]:
    y = conv(x)
    champ = 1 + 2 * r   # Pour un noyau 3×3 : 1 + 2*(r*(K-1)/2) = 2r+1
    print(f"  r={r} → champ réceptif {champ}×{champ}, sortie {y.shape[2]}×{y.shape[3]}")
    assert y.shape[2:] == x.shape[2:], "Résolution modifiée !"
```

### 2.2 La règle du padding

Pour qu'une convolution dilatée de taux $r$ avec un noyau $k \times k$ conserve la résolution de sortie, le padding doit être :

$$\text{padding} = r \cdot \lfloor k/2 \rfloor$$

Pour $k=3$ : `padding = r`. Pour $k=1$ : `padding = 0` (pas de padding nécessaire).

### 2.3 L'astuce de DeepLab : supprimer les derniers strides

Au lieu de remplacer toutes les convolutions par des convolutions dilatées, DeepLab adopte une approche chirurgicale : il conserve la plupart du backbone ResNet standard (strides 1, 2, 4, 8) mais **supprime les derniers strides** (les strides du niveau 5 de ResNet) et les remplace par des convolutions dilatées.

```python
import torchvision.models as models

def convertir_resnet_deeplab(
    backbone: nn.Module,
    output_stride: int = 16   # 8 ou 16 — compromis résolution/champ réceptif
) -> nn.Module:
    """
    Modifie un ResNet pour DeepLab en remplaçant les strides par des dilatations.

    output_stride=16 : le stride final est 16 au lieu de 32
                       → feature map 2× plus grande que ResNet standard
                       → r=2 dans la couche 4
    output_stride=8  : le stride final est 8 au lieu de 32
                       → feature map 4× plus grande
                       → r=2 dans couche 3, r=4 dans couche 4
    """
    if output_stride == 16:
        # Supprimer le stride 2 du dernier bloc (layer4) → remplacer par dilation=2
        for m in backbone.layer4.modules():
            if isinstance(m, nn.Conv2d) and m.stride == (2, 2):
                m.stride  = (1, 1)
                m.padding = (2, 2)
                m.dilation = (2, 2)

    elif output_stride == 8:
        # Supprimer les strides 2 de layer3 ET layer4
        for m in backbone.layer3.modules():
            if isinstance(m, nn.Conv2d) and m.stride == (2, 2):
                m.stride  = (1, 1)
                m.padding = (2, 2)
                m.dilation = (2, 2)
        for m in backbone.layer4.modules():
            if isinstance(m, nn.Conv2d):
                if m.stride == (2, 2):
                    m.stride   = (1, 1)
                    m.padding  = (4, 4)
                    m.dilation = (4, 4)
                elif m.dilation == (1, 1) and m.kernel_size == (3, 3):
                    m.padding  = (4, 4)
                    m.dilation = (4, 4)

    return backbone


# Résultat :
# ResNet-50 standard  → stride final 32, feature map 16×16 pour image 512×512
# DeepLab OS=16       → stride final 16, feature map 32×32
# DeepLab OS=8        → stride final 8,  feature map 64×64
# → Plus grande résolution de feature map à coût computationnel modéré
```

---

## 3. L'ASPP : multi-échelle par convolutions parallèles

### 3.1 Le problème de l'échelle unique

Même avec une meilleure résolution grâce aux convolutions dilatées, un seul taux $r$ capte le contexte à une seule échelle. Pour segmenter une page de manuscrit, on a besoin simultanément de :

- **Contexte local** ($r$ petit) : distinguer les pixels d'encre du fond à l'échelle d'un jambage.
- **Contexte moyen** ($r$ intermédiaire) : identifier une ligne de texte parmi plusieurs.
- **Contexte global** ($r$ grand) : comprendre la structure de la page (nombre de colonnes, présence d'une illustration).

Un seul taux de dilatation ne peut capturer ces trois niveaux simultanément.

### 3.2 ASPP : Atrous Spatial Pyramid Pooling

**ASPP** (*Atrous Spatial Pyramid Pooling*, Chen et al., 2017) applique en parallèle plusieurs convolutions dilatées à des taux différents, puis concatène leurs sorties pour obtenir une représentation multi-échelle.

```
Feature map (H/16 × W/16 × 2048)
                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  Conv 1×1         Conv 3×3      Conv 3×3       Conv 3×3            │
│  r=1, d=256       r=6, d=256    r=12, d=256    r=18, d=256         │
│  (contexte         (champ        (champ          (champ             │
│   local)           7×7)          13×13)          21×21)            │
│                                                                     │
│  GAP → Conv 1×1                                                    │
│  (contexte global)                                                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                    ↓ Concaténation
             (H/16 × W/16 × 5×256 = 1280)
                    ↓ Conv 1×1 → 256
             (H/16 × W/16 × 256)
                    ↓ Upsampling bilinéaire ×16
             (H × W × 256)
                    ↓ Conv 1×1 → n_classes
             (H × W × n_classes)
```

Le **Global Average Pooling (GAP)** dans l'ASPP capture le contexte global de l'image entière — son vecteur 1×1 est upsamplé à la résolution de la feature map et ajouté au reste. C'est la branche qui permet au modèle de « voir » la page entière avant de décider pixel par pixel.

```python
class ASPPConv(nn.Sequential):
    """Une branche de l'ASPP : conv 3×3 dilatée + BN + ReLU."""
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, padding=dilation, dilation=dilation,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ASPPPooling(nn.Sequential):
    """Branche de pooling global de l'ASPP."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),          # Global Average Pooling → 1×1
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]                   # Mémoriser la résolution d'entrée
        for mod in self:
            x = mod(x)
        # Upsampler le vecteur 1×1 à la résolution d'entrée
        return nn.functional.interpolate(
            x, size=size, mode="bilinear", align_corners=False
        )


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (Chen et al., 2017 — DeepLabv3).

    Captures le contexte à plusieurs échelles simultanément par des
    convolutions dilatées parallèles à des taux différents.

    Args:
        in_channels  : canaux d'entrée (2048 pour ResNet-50)
        out_channels : canaux de sortie par branche (256 standard)
        taux_dilation: liste des taux de dilatation des branches (hors 1×1 et GAP)
    """
    def __init__(
        self,
        in_channels: int   = 2048,
        out_channels: int  = 256,
        taux_dilation: list = None
    ):
        super().__init__()

        if taux_dilation is None:
            # Taux standard pour output_stride=16
            # Pour output_stride=8, multiplier par 2 : [12, 24, 36]
            taux_dilation = [6, 12, 18]

        # Branche 1×1 (contexte local, champ réceptif 1×1)
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Branches dilatées
        self.branches_dilatees = nn.ModuleList([
            ASPPConv(in_channels, out_channels, r)
            for r in taux_dilation
        ])

        # Branche de pooling global
        self.gap = ASPPPooling(in_channels, out_channels)

        # Projection finale : concaténation de toutes les branches → out_channels
        n_branches = 1 + len(taux_dilation) + 1   # conv1×1 + dilatées + GAP
        self.projet = nn.Sequential(
            nn.Conv2d(n_branches * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Toutes les branches en parallèle
        sorties = [self.conv1x1(x)]
        sorties += [branch(x) for branch in self.branches_dilatees]
        sorties.append(self.gap(x))

        # Concaténation sur la dimension des canaux
        return self.projet(torch.cat(sorties, dim=1))
```

---

## 4. DeepLabv3 : architecture complète

**DeepLabv3** (Chen et al., 2017) assemble les briques présentées :

1. Un backbone ResNet modifié (output stride 8 ou 16).
2. Un module ASPP sur la dernière feature map.
3. Un upsampling bilinéaire final vers la résolution d'entrée.

```python
class DeepLabv3(nn.Module):
    """
    DeepLabv3 (Chen et al., 2017).

    Architecture :
        ResNet backbone (modifié, output stride 8 ou 16)
        → ASPP (multi-échelle)
        → Conv 1×1 (tête de classification)
        → Upsampling bilinéaire ×8 ou ×16

    Args:
        n_classes     : nombre de classes de segmentation
        backbone_name : "resnet50" ou "resnet101"
        output_stride : 8 (plus précis) ou 16 (plus rapide)
        pretrained    : si True, charge les poids ImageNet du backbone
    """
    def __init__(
        self,
        n_classes: int = 7,           # Ex : 7 classes SegmOnto
        backbone_name: str = "resnet50",
        output_stride: int = 16,
        pretrained: bool = True,
    ):
        super().__init__()

        # ── Backbone ──────────────────────────────────────────────────────────
        if backbone_name == "resnet50":
            backbone = models.resnet50(pretrained=pretrained)
        elif backbone_name == "resnet101":
            backbone = models.resnet101(pretrained=pretrained)
        else:
            raise ValueError(f"Backbone non supporté : {backbone_name}")

        # Extraire les couches pertinentes (supprimer GAP et FC)
        self.couche0 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.couche1 = backbone.layer1   # Stride 4
        self.couche2 = backbone.layer2   # Stride 8
        self.couche3 = backbone.layer3   # Stride 16
        self.couche4 = backbone.layer4   # Stride 32 → modifié ci-dessous

        # Modifier le backbone pour l'output stride souhaité
        self.output_stride = output_stride
        self._modifier_strides(output_stride)

        # ── ASPP ──────────────────────────────────────────────────────────────
        # Les taux de dilatation sont adaptés à l'output stride
        if output_stride == 16:
            taux = [6, 12, 18]
        else:   # output_stride == 8
            taux = [12, 24, 36]

        self.aspp = ASPP(in_channels=2048, out_channels=256, taux_dilation=taux)

        # ── Tête de classification ────────────────────────────────────────────
        self.tete = nn.Conv2d(256, n_classes, kernel_size=1)

    def _modifier_strides(self, output_stride: int) -> None:
        """Remplace les strides par des dilatations pour réduire la compression."""
        if output_stride == 16:
            for m in self.couche4.modules():
                if isinstance(m, nn.Conv2d) and m.stride == (2, 2):
                    m.stride   = (1, 1)
                    m.padding  = (2, 2)
                    m.dilation = (2, 2)
        elif output_stride == 8:
            for m in self.couche3.modules():
                if isinstance(m, nn.Conv2d) and m.stride == (2, 2):
                    m.stride   = (1, 1)
                    m.padding  = (2, 2)
                    m.dilation = (2, 2)
            for m in self.couche4.modules():
                if isinstance(m, nn.Conv2d) and m.stride == (2, 2):
                    m.stride   = (1, 1)
                    m.padding  = (4, 4)
                    m.dilation = (4, 4)
                elif isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3):
                    m.padding  = (4, 4)
                    m.dilation = (4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]

        # Encodage par le backbone
        x = self.couche0(x)
        x = self.couche1(x)
        x = self.couche2(x)
        x = self.couche3(x)
        x = self.couche4(x)
        # x : (B, 2048, H/output_stride, W/output_stride)

        # ASPP : représentation multi-échelle
        x = self.aspp(x)
        # x : (B, 256, H/output_stride, W/output_stride)

        # Classification pixel par pixel
        x = self.tete(x)
        # x : (B, n_classes, H/output_stride, W/output_stride)

        # Upsampling final à la résolution d'entrée
        x = nn.functional.interpolate(
            x, size=input_size, mode="bilinear", align_corners=False
        )
        # x : (B, n_classes, H, W)

        return x
```

---

## 5. DeepLabv3+ : le meilleur des deux mondes

### 5.1 La limite de DeepLabv3 : un encodeur sans décodeur fin

DeepLabv3 upsampler sa sortie directement depuis output_stride=16 (ou 8) vers la résolution originale. Même avec un output_stride plus fin, l'upsampling bilinéaire ×16 reste imprécis sur les contours — les bords des objets sont « mous » comparés aux prédictions de U-Net avec ses skip connections.

**DeepLabv3+** (Chen et al., 2018) ajoute un **décodeur léger** inspiré de U-Net : une skip connection depuis les features de bas niveau du backbone (stride 4, donc haute résolution) est ajoutée entre le backbone et la sortie finale.

```
Image (H × W)
    ↓
┌───────────────────────────────────────────────────────────────────┐
│  Encoder : ResNet modifié (output stride = 16)                   │
│                                                                   │
│  couche1 (stride 4) ──────────────────────────────────────────┐  │
│  couche2 (stride 8)                                            │  │
│  couche3 (stride 16)                                           │  │
│  couche4 (stride 16 — modifié)                                 │  │
│                 ↓                                               │  │
│               ASPP                                              │  │
│          (B, 256, H/16, W/16)                                  │  │
└────────────────────────────────────────────────────────────────│──┘
                 ↓ Upsampling ×4                                  │
            (B, 256, H/4, W/4)                                   │ Skip : couche1
                                                                  │ (B, 2048, H/4, W/4)
                 ↓ Concat ←──── Conv 1×1 (48 canaux) ←───────────┘
            (B, 256+48, H/4, W/4)
                 ↓ Conv 3×3 + Conv 3×3
            (B, 256, H/4, W/4)
                 ↓ Conv 1×1 → n_classes
                 ↓ Upsampling ×4
            (B, n_classes, H, W)
```

La skip connection est **peu profonde** (seulement 2 niveaux de stride) et légère (projetée à 48 canaux) — contrairement aux 4 niveaux de skip dans U-Net. DeepLabv3+ conserve l'efficacité de DeepLabv3 tout en améliorant la précision des contours.

```python
class DeepLabv3Plus(nn.Module):
    """
    DeepLabv3+ (Chen et al., 2018).

    Améliore DeepLabv3 par l'ajout d'un décodeur léger avec une skip connection
    depuis les features de bas niveau (stride 4) du backbone.
    Combine la multi-échelle de l'ASPP avec la précision des contours de U-Net.
    """
    def __init__(
        self,
        n_classes: int = 7,
        backbone_name: str = "resnet50",
        output_stride: int = 16,
        pretrained: bool = True,
        canaux_skip: int = 48,    # Canaux de projection de la skip connection
    ):
        super().__init__()

        # ── Encoder (identique à DeepLabv3) ───────────────────────────────────
        if backbone_name == "resnet50":
            backbone = models.resnet50(pretrained=pretrained)
        else:
            backbone = models.resnet101(pretrained=pretrained)

        self.couche0 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.couche1 = backbone.layer1   # Stride 4 → skip connection
        self.couche2 = backbone.layer2
        self.couche3 = backbone.layer3
        self.couche4 = backbone.layer4

        self._modifier_strides_encodeur(output_stride)
        self.output_stride = output_stride

        # ASPP
        taux = [6, 12, 18] if output_stride == 16 else [12, 24, 36]
        self.aspp = ASPP(in_channels=2048, out_channels=256, taux_dilation=taux)

        # ── Décodeur léger ────────────────────────────────────────────────────
        # Projection de la skip connection (couche1 → 256 canaux → canaux_skip)
        canaux_couche1 = 256   # ResNet-50 : couche1 produit 256 canaux
        self.proj_skip = nn.Sequential(
            nn.Conv2d(canaux_couche1, canaux_skip, kernel_size=1, bias=False),
            nn.BatchNorm2d(canaux_skip),
            nn.ReLU(inplace=True),
        )

        # Raffinement après concaténation ASPP upsamplé + skip projetée
        self.raffinement = nn.Sequential(
            nn.Conv2d(256 + canaux_skip, 256, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        self.tete = nn.Conv2d(256, n_classes, kernel_size=1)

    def _modifier_strides_encodeur(self, output_stride: int) -> None:
        if output_stride == 16:
            for m in self.couche4.modules():
                if isinstance(m, nn.Conv2d) and m.stride == (2, 2):
                    m.stride = (1, 1); m.padding = (2, 2); m.dilation = (2, 2)
        elif output_stride == 8:
            for m in self.couche3.modules():
                if isinstance(m, nn.Conv2d) and m.stride == (2, 2):
                    m.stride = (1, 1); m.padding = (2, 2); m.dilation = (2, 2)
            for m in self.couche4.modules():
                if isinstance(m, nn.Conv2d) and m.stride == (2, 2):
                    m.stride = (1, 1); m.padding = (4, 4); m.dilation = (4, 4)
                elif isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3):
                    m.padding = (4, 4); m.dilation = (4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]

        # ── Encodage ──────────────────────────────────────────────────────────
        x = self.couche0(x)           # stride 4
        skip = self.couche1(x)        # stride 4  ← skip connection
        x    = self.couche2(skip)     # stride 8
        x    = self.couche3(x)        # stride 16
        x    = self.couche4(x)        # stride 16 (modifié)

        # ── ASPP ──────────────────────────────────────────────────────────────
        x = self.aspp(x)              # (B, 256, H/16, W/16)

        # ── Décodeur ──────────────────────────────────────────────────────────
        # Upsampling de la sortie ASPP à la résolution de la skip connection (stride 4)
        x = nn.functional.interpolate(
            x, size=skip.shape[2:], mode="bilinear", align_corners=False
        )   # (B, 256, H/4, W/4)

        # Projection et concaténation de la skip connection
        skip_proj = self.proj_skip(skip)   # (B, 48, H/4, W/4)
        x = torch.cat([x, skip_proj], dim=1)   # (B, 256+48, H/4, W/4)

        # Raffinement des contours
        x = self.raffinement(x)            # (B, 256, H/4, W/4)
        x = self.tete(x)                   # (B, n_classes, H/4, W/4)

        # Upsampling final ×4 vers la résolution originale
        x = nn.functional.interpolate(
            x, size=input_size, mode="bilinear", align_corners=False
        )   # (B, n_classes, H, W)

        return x
```

---

## 6. Entraînement et réglage des hyperparamètres

### 6.1 Les taux de dilatation selon l'output stride

Le choix des taux de dilatation dans l'ASPP dépend de l'output stride du backbone. Si l'output stride est 16 (feature map $H/16 \times W/16$), les taux `[6, 12, 18]` couvrent des champs réceptifs effectifs de $13 \times 13$, $25 \times 25$ et $37 \times 37$ pixels dans l'image originale. Si l'output stride est 8, les taux doivent être doublés pour couvrir les mêmes champs réceptifs : `[12, 24, 36]`.

```python
def calculer_champs_receptifs(
    output_stride: int,
    taux_dilation: list[int]
) -> None:
    """Affiche les champs réceptifs effectifs dans l'image originale."""
    print(f"Output stride = {output_stride}")
    for r in taux_dilation:
        # Champ réceptif de la conv 3×3 dilatée dans la feature map
        cf_feature = 1 + 2 * r   # Pour un noyau 3×3
        # Champ réceptif dans l'image originale (multiplier par output_stride)
        cf_image = cf_feature * output_stride
        print(f"  r={r:2d} → champ réceptif feature map : {cf_feature:2d}×{cf_feature:2d}"
              f" → image originale : {cf_image}×{cf_image} px")

calculer_champs_receptifs(output_stride=16, taux_dilation=[6, 12, 18])
# r= 6 → champ feature map : 13×13 → image originale : 208×208 px
# r=12 → champ feature map : 25×25 → image originale : 400×400 px
# r=18 → champ feature map : 37×37 → image originale : 592×592 px

calculer_champs_receptifs(output_stride=8, taux_dilation=[12, 24, 36])
# Mêmes champs réceptifs dans l'image originale
```

### 6.2 Optimiseur et learning rate schedule

DeepLabv3 est sensible au taux d'apprentissage et au schedule. Le schedule **polynomial decay** est recommandé par les auteurs originaux :

$$\text{lr}(t) = \text{lr}_0 \times \left(1 - \frac{t}{T_\text{max}}\right)^{0.9}$$

```python
from torch.optim.lr_scheduler import LambdaLR

def creer_schedule_polynomial(
    optimizer,
    n_steps_total: int,
    puissance: float = 0.9
):
    """
    Schedule polynomial decay — recommandé pour DeepLab.
    Le LR décroît selon (1 - step/max_steps)^puissance.
    """
    def lr_lambda(step: int) -> float:
        return max(0.0, (1 - step / n_steps_total) ** puissance)

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


# Configuration d'entraînement recommandée
optimizer = torch.optim.SGD(
    [
        {"params": modele.couche0.parameters(), "lr": 1e-3},
        {"params": modele.couche1.parameters(), "lr": 1e-3},
        {"params": modele.couche2.parameters(), "lr": 1e-3},
        {"params": modele.couche3.parameters(), "lr": 1e-3},
        {"params": modele.couche4.parameters(), "lr": 1e-3},
        {"params": modele.aspp.parameters(),    "lr": 1e-2},  # ASPP : LR ×10
        {"params": modele.tete.parameters(),    "lr": 1e-2},
    ],
    momentum=0.9,
    weight_decay=4e-5
)
# Note : le backbone pré-entraîné utilise un LR 10× plus faible que l'ASPP
# Pour fine-tuning sur manuscrits : lr_backbone=1e-4, lr_aspp=1e-3
```

### 6.3 Visualisation de l'activation ASPP par branche

Pour comprendre ce que chaque branche de l'ASPP capture sur une image de manuscrit, on peut visualiser les activations de chaque branche indépendamment.

```python
import matplotlib.pyplot as plt
import numpy as np

def visualiser_branches_aspp(
    modele: DeepLabv3,
    image_pil,
    nom_figure: str = "aspp_branches"
) -> None:
    """
    Visualise les activations de chaque branche de l'ASPP
    sur une image de page de manuscrit.
    Montre quelle branche capture quel niveau de contexte.
    """
    x = torch.tensor(
        np.array(image_pil.convert("RGB")) / 255.0, dtype=torch.float32
    ).permute(2, 0, 1).unsqueeze(0)

    modele.eval()
    with torch.no_grad():
        # Extraire la feature map avant l'ASPP
        feat = modele.couche0(x)
        feat = modele.couche1(feat)
        feat = modele.couche2(feat)
        feat = modele.couche3(feat)
        feat = modele.couche4(feat)

        # Activer chaque branche indépendamment
        activations = {
            "Conv 1×1 (local)": modele.aspp.conv1x1(feat),
        }
        taux = [6, 12, 18]   # Adapter si output_stride=8
        for r, branch in zip(taux, modele.aspp.branches_dilatees):
            cf = (1 + 2*r) * modele.output_stride
            activations[f"Dilaté r={r}\n(champ {cf}px)"] = branch(feat)
        activations["GAP (global)"] = modele.aspp.gap(feat)

    # Visualisation : norme L2 des activations par pixel (proxy de l'activation)
    n_branches = len(activations)
    fig, axes = plt.subplots(1, n_branches + 1, figsize=(4*(n_branches+1), 4))
    fig.suptitle("Activations par branche ASPP — manuscrit médiéval",
                 fontsize=11, fontweight="bold")

    axes[0].imshow(image_pil)
    axes[0].set_title("Image originale", fontsize=9)
    axes[0].axis("off")

    for ax, (nom, act) in zip(axes[1:], activations.items()):
        # Norme L2 sur les canaux → carte d'activation 2D
        norme = act.squeeze(0).norm(dim=0).cpu().numpy()
        norme = (norme - norme.min()) / (norme.max() - norme.min() + 1e-8)

        im = ax.imshow(
            norme,
            cmap="viridis",
            interpolation="bilinear",
            aspect="auto"
        )
        ax.set_title(nom, fontsize=8)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig(f"figures/{nom_figure}.png", dpi=150, bbox_inches="tight")
    plt.show()
```

---

## 7. Comparaison U-Net vs DeepLabv3+ pour les manuscrits médiévaux

### 7.1 Tableau de comparaison détaillé

| Critère | U-Net | Attention U-Net | DeepLabv3 | DeepLabv3+ |
|---------|-------|-----------------|-----------|------------|
| **Paradigme** | Encodeur-décodeur + skip | + attention sur skip | Dilatation + ASPP | Dilatation + ASPP + décodeur léger |
| **Résolution feature map** | Reconstruite (×2 par étape) | Reconstruite | Maintenue (OS=16 ou 8) | Maintenue + skip stride 4 |
| **Précision des contours** | Excellente (4 niveaux de skip) | Très bonne | Bonne | Très bonne |
| **Contexte multi-échelle** | Via la profondeur | Via la profondeur + AG | Via l'ASPP | Via l'ASPP |
| **Données nécessaires** | Peu (conçu pour peu de données) | Peu | Plus (backbone ImageNet aide) | Plus |
| **Paramètres** | ~31 M (standard) | ~35 M | ~44 M (ResNet-50) | ~44 M |
| **Mémoire GPU (train)** | Élevée (skip connections) | Élevée | Modérée (OS=16) | Modérée |
| **Vitesse d'inférence** | Rapide | Légèrement + lent | Rapide (OS=16) | Rapide |
| **Performance binarisation** | Bonne | Meilleure | Bonne | Très bonne |
| **Segmentation layout** | Bonne (faible contexte global) | Bonne | Très bonne (ASPP) | Excellente |
| **Détection de lignes** | Excellente (bords fins) | Excellente | Bonne | Très bonne |

### 7.2 Guide de décision pour les manuscrits

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Quelle architecture pour ma tâche ?                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Binarisation (encre vs fond) :                                             │
│    → U-Net léger (features=[32,64,128,256]) — rapide, précis sur les traits │
│                                                                              │
│  Segmentation de lignes (baselines/polygones) :                             │
│    → U-Net standard — excellent sur les structures fines et longues         │
│    → (BLLA de Kraken utilise cette approche)                                │
│                                                                              │
│  Segmentation de layout (colonnes, marges, illustrations) :                 │
│    → DeepLabv3+ si le contexte global est important                        │
│    → (distinguer texte/illustration sur une page à 2 colonnes)             │
│    → Attention U-Net si les données sont limitées (< 500 pages)            │
│                                                                              │
│  Classification de régions (texte/image/rubrique/lettrine) :               │
│    → DeepLabv3+ + fine-tuning sur données SegmOnto                         │
│    → (dhSegment en production : ResNet U-Net hybride)                      │
│                                                                              │
│  Données très limitées (< 50 pages annotées) :                             │
│    → U-Net avec encodeur ResNet pré-entraîné (segmentation-models-pytorch) │
│    → Fine-tuning partiel (geler couches 1-3, entraîner couches 4 + décodeur│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Application pratique : segmentation de layout avec DeepLabv3+

```python
# Entraîner DeepLabv3+ sur les données SegmOnto (HTR-United)
# pour la segmentation de layout de manuscrits médiévaux

CLASSES_SEGMONTO = [
    "MainZone",         # 0 - Texte principal
    "MarginTextZone",   # 1 - Texte marginal
    "DropCapitalZone",  # 2 - Lettrine
    "GraphicZone",      # 3 - Illustration/enluminure
    "TitlePageZone",    # 4 - Titre, rubrique
    "NumberingZone",    # 5 - Numérotation
    "Background",       # 6 - Fond
]

COULEURS_SEGMONTO = {
    0: (70,  130, 180),   # Bleu acier — texte principal
    1: (34,  139, 34),    # Vert forêt — marge
    2: (220, 20,  60),    # Cramoisi — lettrine
    3: (255, 165, 0),     # Orange — illustration
    4: (148, 0,   211),   # Violet — titre
    5: (105, 105, 105),   # Gris — numérotation
    6: (255, 255, 255),   # Blanc — fond
}

modele_layout = DeepLabv3Plus(
    n_classes=len(CLASSES_SEGMONTO),
    backbone_name="resnet50",
    output_stride=16,
    pretrained=True,
    canaux_skip=48
)

# Inférence et visualisation
def segmenter_et_colorier(
    modele: nn.Module,
    image_pil,
    couleurs: dict
):
    """Segmente une page et colorie chaque région selon sa classe."""
    x = torch.tensor(
        np.array(image_pil.convert("RGB")) / 255.0, dtype=torch.float32
    ).permute(2, 0, 1).unsqueeze(0)

    modele.eval()
    with torch.no_grad():
        logits = modele(x)                                # (1, 7, H, W)
        pred   = logits.argmax(dim=1).squeeze().numpy()  # (H, W)

    # Carte colorée
    H, W = pred.shape
    carte = np.zeros((H, W, 3), dtype=np.uint8)
    for classe, couleur in couleurs.items():
        carte[pred == classe] = couleur

    return pred, carte
```

---

## Glossaire des termes avancés

**ASPP (*Atrous Spatial Pyramid Pooling*)**
Module central de DeepLab v2 et v3. Applique en parallèle des convolutions dilatées à plusieurs taux différents (et un Global Average Pooling) sur la même feature map, puis concatène les sorties. Capte simultanément le contexte local, intermédiaire et global sans réduire la résolution.

**Champ réceptif effectif (*effective receptive field*)**
Zone de l'image d'entrée qui influence la valeur d'un neurone donné. Pour une convolution dilatée 3×3 avec taux $r$, le champ réceptif dans la feature map est $(2r+1) \times (2r+1)$. Dans l'image originale (avec output_stride $s$), il est $(2r+1) \times s$.

**Convolution dilatée (*atrous convolution*)**
Convolution avec espacement $r-1$ entre les éléments du noyau (taux de dilatation $r$). Augmente le champ réceptif effectif d'un facteur $r$ sans modifier le nombre de paramètres ni réduire la résolution spatiale. Clé de l'architecture DeepLab.

**DeepLabv3**
Troisième version de l'architecture DeepLab (Chen et al., 2017). Introduit un ASPP amélioré avec normalisation par batch et supprime le post-traitement CRF des versions précédentes. Utilise un backbone ResNet modifié (output stride 8 ou 16 au lieu de 32).

**DeepLabv3+**
Extension de DeepLabv3 (Chen et al., 2018) ajoutant un décodeur léger inspiré de U-Net : une skip connection depuis les features stride 4 est projetée à 48 canaux et concaténée avec la sortie ASPP upsamplée. Améliore la précision des contours à coût computationnel minimal.

**dhSegment**
Outil de segmentation de documents historiques développé à l'EPFL (Oliveira et al., 2018). Repose sur un encodeur ResNet-50 avec décodeur U-Net. Standard pour la segmentation de layout dans les projets de numérisation patrimoniale.

**Global Average Pooling (dans l'ASPP)**
Branche de l'ASPP qui calcule la moyenne de chaque canal sur toute la feature map (résultat : 1×1×C). L'information globale de la page entière est ainsi intégrée avant d'être upsamplée et concaténée avec les branches dilatées locales.

**Output stride**
Rapport entre la résolution de l'image d'entrée et la résolution de la feature map finale du backbone. Standard ResNet : output_stride = 32. DeepLab OS=16 : feature map 2× plus grande, meilleure précision, léger surcoût computationnel. DeepLab OS=8 : feature map 4× plus grande, précision maximale, surcoût important.

**Polynomial decay**
Schedule de taux d'apprentissage décroissant selon $\text{lr}(t) = \text{lr}_0 (1 - t/T)^p$. Recommandé par les auteurs originaux de DeepLab. Le paramètre $p=0.9$ est standard. Produit une décroissance rapide en début d'entraînement puis progressive.

**SegmOnto**
Ontologie de segmentation de layout pour documents historiques (Gabay et al., 2021). Définit les types de zones standards : MainZone, MarginTextZone, DropCapitalZone, GraphicZone, TitlePageZone, NumberingZone. Adoptée par HTR-United et eScriptorium.

**Stride convolutif**
Pas de déplacement du noyau lors d'une convolution. Stride=1 : aucune réduction de résolution. Stride=2 : résolution divisée par 2. Remplacer un stride=2 par une convolution dilatée de taux 2 (stride=1) est l'astuce fondamentale de DeepLab pour maintenir la résolution.

**Taux de dilatation (*dilation rate*, $r$)**
Paramètre contrôlant l'espacement entre les éléments d'un noyau dilaté. $r=1$ : convolution standard. $r=2$ : un pixel d'espacement entre les éléments (champ réceptif élargi). Plus $r$ est grand, plus le contexte capté est large — au prix d'une possible perte de précision locale si $r$ est trop grand.

---

## Bibliographie de référence

### La série DeepLab

- **Chen, L.-C., Papandreou, G., Kokkinos, I., Murphy, K., Yuille, A. L.** (2015). *Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs*. ICLR 2015. [arXiv:1412.7062] — DeepLab v1 : convolutions dilatées + CRF post-traitement.

- **Chen, L.-C., Papandreou, G., Kokkinos, I., Murphy, K., Yuille, A. L.** (2017). *DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs*. IEEE Transactions on Pattern Analysis and Machine Intelligence, 40(4). [arXiv:1606.00915] — DeepLab v2 : introduction de l'ASPP.

- **Chen, L.-C., Papandreou, G., Schroff, F., Adam, H.** (2017). *Rethinking Atrous Convolution for Semantic Image Segmentation*. [arXiv:1706.05587] — **DeepLab v3 : ASPP amélioré, suppression du CRF. Article fondateur à lire impérativement.**

- **Chen, L.-C., Zhu, Y., Papandreou, G., Schroff, F., Adam, H.** (2018). *Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation*. ECCV 2018. [arXiv:1802.02611] — **DeepLab v3+ : ajout du décodeur léger. Article de référence de l'architecture finale.**

### Convolutions dilatées et pyramides de features

- **Yu, F., Koltun, V.** (2015). *Multi-Scale Context Aggregation by Dilated Convolutions*. ICLR 2016. [arXiv:1511.07122] — Analyse théorique des convolutions dilatées pour l'agrégation de contexte multi-échelle.

- **Lin, T.-Y., Dollár, P., Girshick, R., He, K., Hariharan, B., Belongie, S.** (2017). *Feature Pyramid Networks for Object Detection*. CVPR 2017. [arXiv:1612.03144] — Alternative (pyramide de features) à l'ASPP pour la multi-échelle.

### Évaluation et datasets de référence

- **Everingham, M. et al.** (2015). *The PASCAL Visual Object Classes Challenge: A Retrospective*. IJCV. — Le benchmark PASCAL VOC 2012 sur lequel DeepLab est évalué.

- **Zhou, B. et al.** (2017). *Scene Parsing through ADE20K Dataset*. CVPR 2017. — ADE20k : benchmark de segmentation sémantique à 150 classes, référence pour les comparaisons architecturales.

### Applications aux documents historiques

- **Oliveira, S. A., Seguin, B., Kaplan, F.** (2018). *dhSegment: A generic deep-learning approach for document segmentation*. ICFHR 2018. [arXiv:1804.10371] — Segmentation de layout de documents historiques par U-Net ResNet. Référence pratique directement applicable à notre pipeline.

- **Gruning, T. et al.** (2019). *A Two-Stage Method for Text Line Detection in Historical Documents*. IJDAR 22(3). — Comparaison U-Net vs approches classiques pour les documents patrimoniaux.

- **Gabay, S., Clérice, T., Camps, J.-B., Pinche, A., Chagué, A.** (2021). *Segmonto: Ontology for Page Segmentation of Medieval and Early Modern Documents*. [hal-03336528] — L'ontologie SegmOnto, standard pour annoter les données d'entraînement de segmentation de layout.

---

*Support de cours rédigé pour le module Computer Vision · Promotion MD5 · Mai 2026*
*Ce document constitue la section 2.2c du Jour 2, intercalée entre 2.2b (U-Net) et le TP 2.3.*
*Il suppose la lecture des sections 2.0 (panorama), 2.1 (CNN) et 2.2b (U-Net).*
*Durée estimée de lecture : 65–75 minutes.*
