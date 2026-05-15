# Cours 2.2 — Le mécanisme d'attention en vision : des Transformers NLP aux ViT

**Module 2 · Computer Vision appliquée aux manuscrits médiévaux · MD5**

---

> *« L'attention est le mécanisme par lequel un modèle décide quoi regarder, et avec quelle intensité, avant de produire une réponse. C'est, d'une certaine façon, ce que fait un lecteur expert quand il déchiffre un manuscrit. »*

---

## Introduction : la rupture conceptuelle

La section 2.1 a identifié les biais inductifs des CNN — localité, invariance à la translation, partage des paramètres — et montré en quoi ils sont inadaptés aux manuscrits médiévaux. Le mécanisme d'attention est la réponse architecturale à ces limitations. Il apporte :

- Un **champ réceptif global dès la première couche** : chaque token peut interagir directement avec tous les autres, sans empiler des couches pour atteindre un contexte lointain.
- Une **pondération contextuelle dynamique** : l'importance relative de chaque élément de la séquence est calculée à la volée en fonction du contenu — pas fixée par la position ou la proximité spatiale.
- Une **absence de biais de localité** : deux patches éloignés peuvent interagir aussi librement que deux patches adjacents.

Ces propriétés font de l'attention le candidat naturel pour traiter des séquences où le contexte global est déterminant — ce qui inclut précisément la lecture de texte manuscrit, où l'identification d'une lettre peut dépendre de mots situés plusieurs positions plus loin.

---

## 1. Genèse : de l'attention seq2seq à la self-attention

### 1.1 Le problème du goulot d'étranglement dans les RNN encoder-decoder

Les premiers modèles de traduction automatique neurale (Sutskever et al., 2014) utilisaient une architecture encodeur-décodeur entièrement basée sur des LSTM. L'encodeur lit la phrase source et produit un **vecteur de contexte** $c$ de dimension fixe ; le décodeur génère la traduction mot à mot en conditionnant chaque prédiction sur $c$.

Le problème est structurel : toute l'information de la phrase source doit être compressée dans un vecteur $c$ de dimension fixe (typiquement 256 ou 512). Pour des phrases longues, ce vecteur devient un **goulot d'étranglement** (*bottleneck*) — l'information sur les mots précoces est progressivement écrasée par les mots tardifs, ce qui dégrade la qualité de la traduction sur les longues phrases.

### 1.2 L'attention de Bahdanau (2014)

Bahdanau, Cho & Bengio proposent une solution élégante : au lieu d'utiliser un unique vecteur de contexte, le décodeur calcule à chaque pas un **vecteur de contexte dynamique** $c_t$ comme combinaison linéaire pondérée des états cachés de l'encodeur $h_1, \ldots, h_T$ :

$$c_t = \sum_{j=1}^{T} \alpha_{tj} h_j$$

Les **poids d'attention** $\alpha_{tj}$ mesurent la pertinence de l'état encodeur $j$ pour produire le token décodeur $t$ :

$$\alpha_{tj} = \frac{\exp(e_{tj})}{\sum_{k=1}^{T} \exp(e_{tk})}, \quad e_{tj} = a(s_{t-1}, h_j)$$

où $a$ est une petite réseau de neurones feedforward (le **réseau d'alignement** ou *attention scorer*) et $s_{t-1}$ est l'état caché courant du décodeur.

**Intuition :** pour produire le mot « king » en anglais depuis « le roi » en français, le décodeur apprend à allouer un poids $\alpha$ élevé à l'état encodeur correspondant à « roi ». Ce mécanisme est appris de bout en bout sans supervision explicite de l'alignement.

**Pertinence pour les manuscrits :** c'est exactement ce qu'un paléographe fait quand il lit *mnim* : il regarde l'ensemble du mot (et parfois de la phrase) pour décider si c'est *minim*, *minum* ou *minim*. L'attention encode ce comportement.

### 1.3 Vers la self-attention : « Attention is All You Need » (2017)

Vaswani et al. franchissent une étape supplémentaire : ils suppriment complètement la récurrence et font de l'attention le seul mécanisme de propagation de l'information. La self-attention (*intra-attention*) permet à chaque position d'une séquence d'interagir avec *toutes* les autres positions de la *même* séquence — encodeur-décodeur, mais aussi encoder-encoder et décodeur-décodeur.

---

## 2. Le mécanisme de self-attention : formulation complète

### 2.1 L'intuition des requêtes, clés et valeurs

Le mécanisme de self-attention est formalisé avec trois matrices : **Query** (Q), **Key** (K) et **Value** (V). Ces noms sont hérités d'une analogie avec les systèmes de bases de données :

- La **requête** (Q) représente ce que le token courant *cherche* dans le contexte.
- La **clé** (K) représente ce que chaque token *offre* comme information.
- La **valeur** (V) représente le contenu que chaque token *contribue* à la réponse.

L'idée : pour calculer la représentation d'un token $i$, on calcule la similarité entre la requête $q_i$ et toutes les clés $k_j$, on normalise ces similarités en poids d'attention, et on prend la somme pondérée des valeurs $v_j$.

### 2.2 Scaled Dot-Product Attention

Soit une séquence de $n$ tokens représentés par une matrice $X \in \mathbb{R}^{n \times d_{\text{model}}}$. Les matrices Q, K, V sont obtenues par projections linéaires :

$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$

avec $W_Q, W_K \in \mathbb{R}^{d_{\text{model}} \times d_k}$ et $W_V \in \mathbb{R}^{d_{\text{model}} \times d_v}$.

La sortie de la self-attention est :

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V$$

**Décomposition étape par étape :**

**Étape 1 — Matrice des scores** $S = Q K^\top \in \mathbb{R}^{n \times n}$

$S_{ij} = q_i \cdot k_j$ est le produit scalaire entre la requête du token $i$ et la clé du token $j$. Un score élevé indique une forte compatibilité — le token $i$ « cherche » ce que le token $j$ « offre ».

**Étape 2 — Mise à l'échelle** par $1/\sqrt{d_k}$

Sans cette normalisation, pour $d_k$ grand, les produits scalaires ont une variance élevée ($\sim d_k$) et les softmax saturent — les gradients deviennent trop petits. La mise à l'échelle maintient une variance $\sim 1$.

**Preuve :** si $q_i, k_j \sim \mathcal{N}(0, 1)$ i.i.d., alors $q_i \cdot k_j = \sum_{l=1}^{d_k} q_{il} k_{jl}$ a moyenne $0$ et variance $d_k$. Après division par $\sqrt{d_k}$, la variance est 1.

**Étape 3 — Softmax ligne par ligne** : $A_{ij} = \frac{\exp(S_{ij}/\sqrt{d_k})}{\sum_{l=1}^n \exp(S_{il}/\sqrt{d_k})}$

$A \in \mathbb{R}^{n \times n}$ est la **matrice d'attention**. Chaque ligne somme à 1 : c'est une distribution de probabilité sur les tokens sources.

**Étape 4 — Agrégation** : $\text{Out} = A \cdot V \in \mathbb{R}^{n \times d_v}$

La représentation de sortie du token $i$ est $\sum_j A_{ij} v_j$ — une combinaison linéaire des valeurs, pondérée par les scores d'attention.

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Scaled dot-product attention.

    Args:
        Q : (batch, heads, seq_len_q, d_k)
        K : (batch, heads, seq_len_k, d_k)
        V : (batch, heads, seq_len_k, d_v)
        mask : (batch, 1, 1, seq_len_k) — optionnel, masque de padding

    Returns:
        output : (batch, heads, seq_len_q, d_v)
        attn_weights : (batch, heads, seq_len_q, seq_len_k)
    """
    d_k = Q.size(-1)

    # Étape 1 & 2 : scores normalisés
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    # scores : (batch, heads, seq_len_q, seq_len_k)

    # Masquage (positions de padding)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Étape 3 : softmax
    attn_weights = F.softmax(scores, dim=-1)
    # attn_weights : (batch, heads, seq_len_q, seq_len_k)

    # Étape 4 : agrégation
    output = torch.matmul(attn_weights, V)
    # output : (batch, heads, seq_len_q, d_v)

    return output, attn_weights
```

### 2.3 Complexité computationnelle et mémoire

La matrice des scores $S = QK^\top$ est de taille $n \times n$. Son calcul coûte $O(n^2 d_k)$ FLOPs et son stockage $O(n^2)$ en mémoire.

**Conséquence :** la self-attention standard est **quadratique en la longueur de la séquence**. Pour une page de manuscrit découpée en patches $16 \times 16$ pixels à résolution $1024 \times 1024$, on obtient $n = (1024/16)^2 = 4096$ tokens. La matrice d'attention fait $4096^2 \approx 16,8$ millions d'entrées — soit ~64 Mo en float32, pour une seule tête d'attention et un seul exemple du batch.

C'est pourquoi le ViT standard ne peut pas traiter des images haute résolution directement, et pourquoi des variantes comme le Swin Transformer (section 5) ont été développées.

| Longueur $n$ | Taille matrice attention | Coût relatif |
|-------------|--------------------------|-------------|
| 196 (224px / 16px patches) | 38 000 entrées | ×1 (référence) |
| 1 024 (512px) | 1 M entrées | ×27 |
| 4 096 (1024px) | 16,8 M entrées | ×430 |
| 16 384 (2048px) | 268 M entrées | ×6 880 |

### 2.4 Multi-Head Attention (MHA)

Plutôt qu'une seule opération d'attention, le Transformer utilise $h$ « têtes » d'attention **parallèles**, chacune avec ses propres projections $W_Q^{(i)}, W_K^{(i)}, W_V^{(i)}$ de dimension réduite $d_k = d_v = d_{\text{model}} / h$.

$$\text{MHA}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \cdot W_O$$

$$\text{head}_i = \text{Attention}(Q W_Q^{(i)},\ K W_K^{(i)},\ V W_V^{(i)})$$

**Intuition :** chaque tête peut se spécialiser sur un *type* de relation différent. Dans le contexte des manuscrits, une tête pourrait apprendre à aligner les jambages verticaux pour détecter les minimes (contexte immédiat) ; une autre à détecter les structures de ligature sur plusieurs lettres ; une troisième à identifier les formules récurrentes de la langue médiévale (contexte long).

**Paramètres d'une couche MHA :**
- $h$ matrices $W_Q^{(i)}, W_K^{(i)}, W_V^{(i)} \in \mathbb{R}^{d_{\text{model}} \times d_k}$ : $3 h \cdot d_{\text{model}} \cdot d_k = 3 d_{\text{model}}^2$ paramètres (puisque $d_k = d_{\text{model}}/h$).
- Matrice de sortie $W_O \in \mathbb{R}^{h d_v \times d_{\text{model}}}$ : $d_{\text{model}}^2$ paramètres.
- **Total :** $4 d_{\text{model}}^2$ paramètres, indépendant de $h$.

```python
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads

        # Projections Q, K, V et sortie
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, seq, d_model) → (batch, heads, seq, d_k)"""
        B, T, D = x.shape
        x = x.view(B, T, self.n_heads, self.d_k)
        return x.transpose(1, 2)

    def forward(self, x: torch.Tensor, mask=None):
        B, T, _ = x.shape

        # Projections et découpe en têtes
        Q = self.split_heads(self.W_Q(x))  # (B, h, T, d_k)
        K = self.split_heads(self.W_K(x))
        V = self.split_heads(self.W_V(x))

        # Attention
        out, attn = scaled_dot_product_attention(Q, K, V, mask)

        # Concaténation et projection finale
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_O(out), attn
```

---

## 3. Le Transformer complet

### 3.1 Le bloc encodeur

Un bloc encodeur Transformer est composé de deux sous-modules, chacun suivi d'une connexion résiduelle et d'une Layer Normalization (architecture *Pre-LN*, préférée dans les ViT modernes) :

```
x ──→ LayerNorm ──→ MultiHeadAttention ──→ (+) ──→ LayerNorm ──→ FFN ──→ (+) ──→ sortie
│                                            ↑                              ↑
└────────────────────────────────────────────┘                              │
└──────────────────────────────────────────────────────────────────────────┘
```

**Le réseau feedforward (FFN) :**

$$\text{FFN}(x) = \max(0,\ x W_1 + b_1) W_2 + b_2$$

Le FFN traite chaque token **indépendamment** (pas d'interaction entre positions). Il double typiquement la dimensionnalité en interne ($d_{\text{ff}} = 4 d_{\text{model}}$) avant de la réduire. Son rôle est d'ajouter de la non-linéarité et d'augmenter la capacité du modèle à transformer les représentations.

**Paramètres d'un bloc encodeur :**
- MHA : $4 d_{\text{model}}^2$
- FFN : $2 \times d_{\text{model}} \times d_{\text{ff}} = 8 d_{\text{model}}^2$ (pour $d_{\text{ff}} = 4 d_{\text{model}}$)
- LayerNorm (×2) : $4 d_{\text{model}}$ (négligeable)
- **Total :** $\approx 12 d_{\text{model}}^2$ paramètres par bloc.

Pour ViT-B ($d_{\text{model}} = 768$, 12 blocs) : $12 \times 12 \times 768^2 \approx 85$M paramètres (hors patch embedding).

### 3.2 L'encodage positionnel

La self-attention est **invariante à la permutation** : permuter les tokens dans la séquence produit exactement la même sortie (permutée dans le même ordre). Elle n'encode aucune information sur l'ordre ou la position — contrairement aux RNN, dont la récurrence encode naturellement la séquentialité.

Il faut donc injecter explicitement l'information de position dans les tokens. Deux approches principales :

**Encodage sinusoïdal (Vaswani et al., 2017)**

$$\text{PE}_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$
$$\text{PE}_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

Propriétés : déterministe (pas de paramètres appris), généralisable à des longueurs non vues à l'entraînement, encodage relatif implicite (le produit scalaire $\text{PE}_{pos} \cdot \text{PE}_{pos+k}$ ne dépend que de $k$, pas de $pos$).

**Encodage positionnel appris**

On apprend une matrice $E_{\text{pos}} \in \mathbb{R}^{n_{\text{max}} \times d_{\text{model}}}$ pendant l'entraînement. C'est l'approche adoptée dans ViT-B/16 original.

Inconvénient : ne généralise pas à des résolutions non vues. Si le modèle est entraîné sur des images $224 \times 224$ (196 patches), il ne sait pas quoi faire d'une image $448 \times 448$ (784 patches) sans interpolation des encodages positionnels.

**Encodages positionnels relatifs**

Des travaux récents (Shaw et al., 2018 ; Raffel et al., 2020) intègrent l'information de position directement dans les scores d'attention sous forme relative :

$$S_{ij} = \frac{(q_i + r_{i-j})(k_j)^\top}{\sqrt{d_k}}$$

où $r_{i-j}$ encode la *distance* entre les positions $i$ et $j$. Généralise mieux aux longueurs non vues. Adopté dans DeBERTa et plusieurs variantes de ViT.

---

## 4. Vision Transformer (ViT)

### 4.1 L'idée centrale

Dosovitskiy et al. (2020) font une observation simple : le Transformer, conçu pour des séquences de tokens NLP, peut être appliqué directement à des images si on transforme l'image en une séquence de tokens. La question est : comment ?

**La réponse de ViT :** découper l'image en patches non chevauchants, aplatir chaque patch en un vecteur, et le projeter dans l'espace de tokens. On obtient une séquence de $n = HW/P^2$ tokens de dimension $d_{\text{model}}$ (pour une image $H \times W$ découpée en patches $P \times P$).

### 4.2 Patch Embedding

Soit une image $I \in \mathbb{R}^{H \times W \times C}$ (hauteur $H$, largeur $W$, $C$ canaux). On la découpe en $n = (H/P) \times (W/P)$ patches de taille $P \times P \times C$, qu'on aplatit en vecteurs de dimension $P^2 C$ :

$$x_i^{\text{patch}} \in \mathbb{R}^{P^2 C}, \quad i = 1, \ldots, n$$

On projette ensuite chaque patch dans l'espace de tokens par une transformation linéaire $E \in \mathbb{R}^{P^2 C \times d_{\text{model}}}$ :

$$z_i = x_i^{\text{patch}} \cdot E + e_{\text{pos}, i}$$

En pratique, cette projection est implémentée comme une **convolution $P \times P$ avec stride $P$** — ce qui est exactement équivalent mais computationnellement optimisé.

```python
class PatchEmbedding(nn.Module):
    """
    Découpe l'image en patches non chevauchants et les projette
    dans l'espace de tokens via une convolution P×P, stride P.
    """
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        d_model: int = 768
    ):
        super().__init__()
        assert image_size % patch_size == 0, \
            "La taille de l'image doit être divisible par la taille du patch."
        self.n_patches = (image_size // patch_size) ** 2
        self.patch_size = patch_size

        # Convolution P×P, stride P ≡ projection linéaire des patches
        self.projection = nn.Conv2d(
            in_channels, d_model,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, H, W)
        → (B, n_patches, d_model)
        """
        x = self.projection(x)          # (B, d_model, H/P, W/P)
        x = x.flatten(2)                # (B, d_model, n_patches)
        x = x.transpose(1, 2)           # (B, n_patches, d_model)
        return x
```

**Choix de la taille de patch $P$ :**

| $P$ | $n$ (pour 224×224) | $n$ (pour 1024×1024) | Compromis |
|-----|-------------------|---------------------|-----------|
| 32 | 49 | 1 024 | Rapide, moins détaillé |
| 16 | 196 | 4 096 | Équilibre standard (ViT-B/16) |
| 8 | 784 | 16 384 | Très précis, très lent |

Pour les manuscrits à haute résolution, $P = 16$ sur des images redimensionnées à $512 \times 512$ (1 024 patches) est un bon compromis praticable.

### 4.3 Le token de classification [CLS]

À la séquence de $n$ tokens de patches, ViT préfixe un **token de classification** $z_0 = z_{\text{cls}}$ appris, similaire au token [CLS] de BERT :

$$Z = [z_{\text{cls}};\ z_1;\ z_2;\ \ldots;\ z_n] + E_{\text{pos}} \in \mathbb{R}^{(n+1) \times d_{\text{model}}}$$

Ce token n'est associé à aucune région de l'image. À travers les couches du Transformer, il agrège l'information de tous les patches par attention, et sa représentation en sortie ($z_0^{(L)}$, avec $L$ le nombre de blocs) est utilisée comme représentation globale de l'image pour la classification.

**Pourquoi un token séparé ?** Alternativement, on pourrait faire un moyenne-pooling de tous les tokens de patches en sortie. Les deux approches fonctionnent ; le token CLS est légèrement supérieur en pratique et permet une analyse directe des patterns d'attention.

```python
class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        n_classes: int = 1000,
        d_model: int = 768,
        depth: int = 12,           # Nombre de blocs
        n_heads: int = 12,
        d_ff: int = 3072,
        dropout: float = 0.1
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, d_model)
        n_patches = self.patch_embed.n_patches

        # Token CLS et encodages positionnels
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.dropout = nn.Dropout(dropout)

        # Blocs Transformer
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Tête de classification
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)          # (B, n, d_model)

        # Ajout du token CLS
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)    # (B, n+1, d_model)

        # Encodage positionnel
        x = self.dropout(x + self.pos_embed)

        # Blocs Transformer (avec récupération des attention maps)
        attn_maps = []
        for block in self.blocks:
            x, attn = block(x)
            attn_maps.append(attn)

        # Classification via le token CLS
        x = self.norm(x[:, 0])           # (B, d_model) — token CLS uniquement
        return self.head(x), attn_maps
```

### 4.4 Configurations standard de ViT

Les configurations originales de Dosovitskiy et al. (2020) sont devenues des références :

| Modèle | Blocs | $d_{\text{model}}$ | Têtes | Paramètres | ImageNet top-1 |
|--------|-------|------------------|-------|-----------|----------------|
| ViT-Ti/16 | 12 | 192 | 3 | 5,7M | ~72% |
| ViT-S/16 | 12 | 384 | 6 | 22M | ~79% |
| ViT-B/16 | 12 | 768 | 12 | 86M | ~81% |
| ViT-L/16 | 24 | 1024 | 16 | 307M | ~82% |
| ViT-H/14 | 32 | 1280 | 16 | 632M | ~83% |

La notation `/16` indique des patches $16 \times 16$ ; `/14` des patches $14 \times 14$ (plus fins, séquence plus longue).

---

## 5. Variantes du ViT pour les documents haute résolution

### 5.1 DeiT : Data-Efficient Image Transformers

ViT-B entraîné de zéro sur ImageNet (1,2M images) obtient ~77% top-1 — significativement moins que les CNN comparables. Dosovitskiy et al. rapportent que leurs meilleurs résultats nécessitent JFT-300M (300 millions d'images) pour pré-entraînement.

DeiT (Touvron et al., 2021) ramène ces besoins à ImageNet seul, par deux contributions :

**1. Distillation par token.** Outre le token CLS standard, DeiT ajoute un **token de distillation** $z_{\text{dist}}$. La loss est la somme d'une loss de classification (depuis $z_{\text{cls}}$) et d'une loss de distillation (depuis $z_{\text{dist}}$, entraîné à imiter un CNN enseignant comme RegNetY-16GF) :

$$\mathcal{L} = (1 - \lambda) \mathcal{L}_{\text{cls}} + \lambda \mathcal{L}_{\text{distill}}$$

**2. Augmentations et régularisation agressives :** RandAugment, CutMix, Mixup, stochastic depth, repeated augmentation.

**Résultat :** DeiT-B/16 atteint 81,8% top-1 sur ImageNet, entraîné sur ImageNet seul — comparable aux meilleurs CNN.

**Pertinence pour les manuscrits :** DeiT montre qu'il est possible d'entraîner un ViT compétitif avec des données limitées, en utilisant un enseignant CNN bien entraîné. Pour fine-tuner sur des données HTR limitées (quelques milliers de lignes), la distillation peut réduire le besoin en données annotées.

### 5.2 Swin Transformer : l'attention hiérarchique à fenêtres décalées

Le Swin Transformer (Liu et al., 2021) résout le problème de la complexité quadratique en introduisant une **attention locale à fenêtres** (*window attention*).

**Attention par fenêtres (W-MSA)**
L'image est découpée en fenêtres disjointes de $M \times M$ patches. La self-attention est calculée indépendamment *à l'intérieur de chaque fenêtre*, ce qui donne une complexité :

$$\Omega(\text{W-MSA}) = 4n d^2 + 2M^2 n d$$

contre $\Omega(\text{MSA}) = 4n d^2 + 2n^2 d$ pour la self-attention globale.

Pour $M \ll n$, le gain est quadratique en $n/M$.

**Attention à fenêtres décalées (SW-MSA)**
L'attention locale seule empêche les interactions entre fenêtres — les patches aux frontières des fenêtres ne peuvent pas interagir avec leurs voisins dans d'autres fenêtres. La solution : alterner entre des fenêtres standard et des fenêtres **décalées** de $(M/2, M/2)$ patches.

```
Couche paire   : fenêtres normales    Couche impaire : fenêtres décalées
┌──┬──┬──┬──┐                         ┌─┬──┬──┬──┬─┐
│  │  │  │  │                         │ │  │  │  │ │
├──┼──┼──┼──┤                         ├─┼──┼──┼──┼─┤
│  │  │  │  │   ←──── décalage ────→  │ │  │  │  │ │
├──┼──┼──┼──┤                         ├─┼──┼──┼──┼─┤
│  │  │  │  │                         │ │  │  │  │ │
└──┴──┴──┴──┘                         └─┴──┴──┴──┴─┘
```

Cela permet les interactions cross-fenêtres à chaque couche impaire, avec un coût computationnel constant (cyclic shift + masking).

**Architecture hiérarchique**
Swin produit des feature maps multi-échelles (comme un CNN) :
- Stage 1 : patches de $4 \times 4$ pixels, feature maps $H/4 \times W/4$
- Stage 2 : après patch merging $2 \times 2$, feature maps $H/8 \times W/8$
- Stage 3 : feature maps $H/16 \times W/16$
- Stage 4 : feature maps $H/32 \times W/32$

Cette hiérarchie est exactement ce que les backbones CNN classiques (ResNet) produisent — ce qui rend Swin compatible avec tous les frameworks de détection et segmentation conçus pour les CNN (FPN, U-Net, Mask R-CNN).

**Résultat sur les documents :** Swin-B avec un décodeur UperNet atteint un mIoU de 51,6 sur ADE20K (segmentation sémantique), contre 45,9 pour un ResNet-101 comparable. Sa capacité à traiter des images haute résolution avec une complexité linéaire en $n$ le rend particulièrement adapté au layout analysis de pages de manuscrits.

### 5.3 BEiT et MAE : pré-entraînement auto-supervisé par masquage

**BEiT (Bao et al., 2022) — BERT pre-training for Image Transformers**

Reproduit pour les images le principe du masquage de BERT : environ 40% des patches sont masqués aléatoirement, et le modèle est entraîné à prédire leurs **visual tokens** (représentations discrètes issues d'un tokenizeur d'images pré-entraîné, dVAE). C'est l'encodeur de TrOCR.

$$\mathcal{L}_{\text{BEiT}} = -\mathbb{E}\left[\sum_{i \in \mathcal{M}} \log p(z_i | x_{\setminus \mathcal{M}})\right]$$

où $\mathcal{M}$ est l'ensemble des indices masqués et $z_i$ le visual token du patch $i$.

**MAE (He et al., 2022) — Masked Autoencoders**

Plus simple que BEiT : on masque 75% des patches et on entraîne un autoencodeur asymétrique à reconstruire les **pixels** des patches masqués (pas des tokens discrets). L'encodeur ne voit que les patches non masqués (25%), ce qui réduit le coût computationnel. Le décodeur léger reconstruit l'image complète.

$$\mathcal{L}_{\text{MAE}} = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \| x_i - \hat{x}_i \|_2^2$$

Intuition : reconstruire 75% d'une image à partir de 25% des patches force le modèle à apprendre des représentations sémantiques riches. La tâche est difficile — on ne peut pas s'en sortir avec des features de bas niveau.

**Pertinence pour les manuscrits :** MAE et BEiT apprennent des représentations robustes sans annotation. Fine-tuner un MAE pré-entraîné sur des scans de manuscrits (même sans transcriptions) peut améliorer significativement les performances HTR en aval, car le modèle apprend la structure visuelle propre aux documents anciens.

---

## 6. Visualisation et interprétabilité : attention maps vs Grad-CAM

### 6.1 Grad-CAM pour les CNN

**Gradient-weighted Class Activation Mapping** (Selvaraju et al., 2017) est la méthode de visualisation de référence pour les CNN. Elle produit une carte de chaleur indiquant quelles régions de l'image influencent le plus la prédiction d'une classe donnée.

**Principe :** on calcule les gradients de la sortie pour la classe cible $c$ par rapport aux feature maps $A^k$ de la dernière couche convolutive, et on les utilise comme poids :

$$\alpha_k^c = \frac{1}{Z} \sum_{i,j} \frac{\partial y^c}{\partial A_{ij}^k}$$

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\!\left(\sum_k \alpha_k^c A^k\right)$$

La ReLU ne conserve que les régions qui *augmentent* le score de la classe $c$.

```python
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.gradients = None
        self.activations = None

        # Hooks sur la couche cible
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, x: torch.Tensor, class_idx: int = None) -> np.ndarray:
        self.model.eval()
        logits = self.model(x)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[0, class_idx].backward()

        # Poids = gradient moyen par canal
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1).squeeze()   # (H', W')
        cam = torch.relu(cam).numpy()

        # Normalisation et redimensionnement
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam
```

### 6.2 Attention Rollout pour les ViT

Les matrices d'attention de chaque couche du ViT donnent une information directe sur ce que chaque token regarde. Mais avec $L$ couches et $h$ têtes, quelle attention visualiser ?

**Attention de la dernière couche, tête par tête**
Représentation directe mais incomplète : l'attention de la couche $L$ encode les relations à ce niveau de représentation, mais pas l'accumulation d'information des couches précédentes.

**Attention Rollout (Abnar & Zuidema, 2020)**
Propagation récursive des matrices d'attention à travers toutes les couches, en tenant compte des connexions résiduelles :

$$\tilde{A}^l = 0.5 \cdot A^l + 0.5 \cdot I$$
$$\text{Rollout}^L = \tilde{A}^L \cdot \tilde{A}^{L-1} \cdots \tilde{A}^1$$

Le facteur $0.5$ modélise le mélange entre l'information propagée par l'attention et l'information propagée directement par le skip connection (identité). La ligne correspondant au token CLS dans la matrice Rollout indique quels patches contribuent le plus à la représentation globale finale.

```python
def attention_rollout(
    attn_maps: list[torch.Tensor],
    head_fusion: str = "mean"
) -> torch.Tensor:
    """
    Calcule l'attention rollout depuis le token CLS vers les patches.

    Args:
        attn_maps : liste de tenseurs (B, heads, n+1, n+1) — une par couche
        head_fusion : "mean" ou "max" pour agréger les têtes

    Returns:
        rollout : (B, n_patches) — importance de chaque patch pour le CLS
    """
    B = attn_maps[0].shape[0]
    n_tokens = attn_maps[0].shape[-1]
    rollout = torch.eye(n_tokens, device=attn_maps[0].device).unsqueeze(0).expand(B, -1, -1)

    for attn in attn_maps:
        # Agréger les têtes
        if head_fusion == "mean":
            attn_fused = attn.mean(dim=1)   # (B, n+1, n+1)
        elif head_fusion == "max":
            attn_fused = attn.max(dim=1).values

        # Mélange avec l'identité (skip connection)
        attn_fused = 0.5 * attn_fused + 0.5 * torch.eye(
            n_tokens, device=attn.device
        ).unsqueeze(0)

        # Normaliser les lignes
        attn_fused = attn_fused / attn_fused.sum(dim=-1, keepdim=True)

        # Produit matriciel cumulé
        rollout = torch.bmm(attn_fused, rollout)

    # Extraire la ligne du token CLS (index 0)
    cls_rollout = rollout[:, 0, 1:]  # (B, n_patches) — exclure le CLS lui-même
    return cls_rollout


def visualize_rollout(
    image: torch.Tensor,
    rollout: torch.Tensor,
    patch_size: int = 16,
    image_size: int = 224
) -> np.ndarray:
    """Superpose le rollout sur l'image originale."""
    n_patches_side = image_size // patch_size
    mask = rollout.reshape(n_patches_side, n_patches_side).cpu().numpy()

    # Normaliser et redimensionner à la taille de l'image
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    mask_resized = np.array(
        Image.fromarray((mask * 255).astype(np.uint8)).resize(
            (image_size, image_size), Image.BICUBIC
        )
    ) / 255.0

    return mask_resized
```

### 6.3 Comparaison Grad-CAM vs Attention Rollout sur les manuscrits

Sur un scan de manuscrit, les deux méthodes révèlent des patterns très différents :

**Grad-CAM (ResNet-50, classification de type de document)**

La carte de chaleur se concentre typiquement sur les **structures locales saillantes** : les lettrines ornées (fort contraste, formes distinctives), les rubriques rouges (différentes des zones noires), les bords de page. Elle ne « voit » pas la structure globale du texte — les lignes de texte ordinaires sont peu activées.

Interprétation : ResNet classifie le type de document en s'appuyant sur des features locales discriminantes (une lettrine ornée → manuscrit de luxe), pas sur la structure globale de la mise en page.

**Attention Rollout (ViT-B/16, même tâche)**

Le rollout du token CLS montre une attention distribuée sur l'**ensemble des lignes de texte**, avec une légère sur-représentation des premières et dernières lignes (où se trouvent souvent les titres et les souscriptions). Les illustrations reçoivent de l'attention en proportion de leur taille, pas de façon disproportionnée.

Interprétation : ViT intègre l'information globale de la page — la distribution spatiale du texte, l'équilibre entre zones de texte et d'image — pour classifier. C'est un signal structurellement plus riche.

**Pour l'HTR (ViT encoder de TrOCR)**

Sur une image de ligne, le rollout montre que chaque token de sortie attend majoritairement les tokens correspondant à sa position dans la séquence, mais aussi des tokens distants — notamment pour les lettres ambiguës. Un token correspondant à un jambage isolé attend les tokens de ses voisins directs *et* des tokens plus lointains correspondant à des lettres contextuellement contraignantes. Ce comportement est exactement ce qu'on attendrait d'un modèle qui résout les ambiguïtés des minimes par le contexte global.

---

## 7. Pourquoi l'attention résout les problèmes identifiés en 2.1

| Limite des CNN | Solution apportée par la self-attention |
|---------------|----------------------------------------|
| Localité : contexte limité au champ réceptif | Champ réceptif *global dès la première couche* — tous les patches interagissent |
| Invariance à la translation : position ignorée | Encodage positionnel explicite — la position est un signal d'entrée, pas effacée |
| Partage des paramètres : statistiques supposées stationnaires | Poids d'attention calculés *dynamiquement* selon le contenu — pas de partage |
| Complexité quadratique en la résolution | Swin : complexité linéaire via fenêtres locales + décalages |
| Réduction de résolution destructive | Swin : architecture hiérarchique préservant les détails par skip connections |
| Distribution shift ImageNet → documents | BEiT/MAE : pré-entraînement auto-supervisé adaptable à n'importe quelle distribution |

Une nuance importante : l'attention ne résout pas *tous* les problèmes gratuitement. Sa complexité quadratique est réelle et ses performances sur de petits datasets sont inférieures à celles des CNN (les CNN ont un meilleur biais inductif pour les images naturelles en faible données). C'est pourquoi les ViT modernes combinent souvent les deux paradigmes : Swin utilise des convolutions pour le patch merging ; ConvNeXt adopte des design choices de ViT dans une architecture purement convolutive.

---

## 8. Utilisations de HuggingFace Transformers pour les ViT

```python
from transformers import ViTModel, ViTConfig, AutoImageProcessor
import torch

# Charger un ViT-B/16 pré-entraîné
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = ViTModel.from_pretrained("google/vit-base-patch16-224")

# Préparer une image
from PIL import Image
image = Image.open("scan_ligne.png").convert("RGB")
inputs = processor(images=image, return_tensors="pt")

# Forward avec récupération des attention maps
with torch.no_grad():
    outputs = model(
        **inputs,
        output_attentions=True   # Récupérer les matrices d'attention
    )

# Représentation du token CLS (feature globale)
cls_features = outputs.last_hidden_state[:, 0, :]  # (1, 768)

# Représentations de tous les patches
patch_features = outputs.last_hidden_state[:, 1:, :]  # (1, 196, 768)

# Matrices d'attention de chaque couche
# Liste de 12 tenseurs, chacun (1, 12, 197, 197)
attention_maps = outputs.attentions
print(f"Nombre de couches : {len(attention_maps)}")
print(f"Shape d'une attention map : {attention_maps[0].shape}")
# → (batch, n_heads, n_tokens, n_tokens) = (1, 12, 197, 197)

# Calculer et visualiser le rollout
rollout = attention_rollout(attention_maps)
mask = visualize_rollout(inputs.pixel_values[0], rollout[0])
```

---

## Glossaire des termes avancés

**Attention (mécanisme d')**
Opération qui calcule, pour chaque élément d'une séquence, une combinaison pondérée des représentations de tous les autres éléments. Les poids sont déterminés dynamiquement par la compatibilité entre la requête de l'élément courant et les clés de tous les éléments.

**Attention Rollout**
Méthode de visualisation de l'attention dans les Transformers : propagation récursive des matrices d'attention à travers toutes les couches, en tenant compte des connexions résiduelles, pour estimer l'influence de chaque token d'entrée sur la représentation finale.

**Autoencodeur**
Architecture composée d'un encodeur (qui compresse l'entrée en une représentation latente) et d'un décodeur (qui reconstruit l'entrée depuis la représentation latente). L'objectif d'entraînement est la reconstruction fidèle de l'entrée.

**BEiT (*BERT pre-training for Image Transformers*)**
Méthode de pré-entraînement auto-supervisé pour ViT par masquage de patches et prédiction de visual tokens discrets. Analogue de BERT pour les images. C'est l'encodeur de TrOCR.

**Cross-attention**
Variante de l'attention où les requêtes proviennent d'une séquence et les clés/valeurs d'une autre séquence différente. Utilisée dans les décodeurs (pour « lire » la sortie de l'encodeur) et dans certains modèles multimodaux.

**CutMix**
Technique d'augmentation de données : coller une région rectangulaire aléatoire d'une image dans une autre, et mélanger les labels proportionnellement à la surface. Améliore la robustesse et la généralisation.

**DeiT (*Data-Efficient Image Transformers*)**
Variante de ViT entraînable sur ImageNet seul (sans pré-entraînement sur des datasets plus grands) par distillation depuis un modèle enseignant CNN et augmentations agressives.

**Distillation (de connaissances)**
Entraîner un modèle « élève » à imiter les prédictions d'un modèle « enseignant » plus grand ou mieux entraîné. Permet de transférer les performances du grand modèle vers un modèle plus petit ou plus spécialisé.

**dVAE (*Discrete Variational Autoencoder*)**
Autoencodeur variationnel produisant des représentations discrètes (tokens visuels). Utilisé par BEiT comme cible de prédiction pour le masquage de patches.

**Encodage positionnel**
Vecteur ajouté à chaque token pour lui communiquer sa position dans la séquence. Peut être sinusoïdal (déterministe) ou appris (paramétrique). Nécessaire car la self-attention est invariante à la permutation.

**Équivariance à la permutation**
Propriété d'une fonction $f$ telle que $f(\text{Permute}(x)) = \text{Permute}(f(x))$. La self-attention est équivariante à la permutation des tokens — l'ordre n'affecte la sortie que via l'encodage positionnel.

**FPN (*Feature Pyramid Network*)**
Architecture qui agrège des feature maps à différentes résolutions spatiales pour produire des représentations multi-échelles. Utilisée en détection d'objets et en segmentation. Compatible avec Swin.

**Grad-CAM (*Gradient-weighted Class Activation Mapping*)**
Méthode de visualisation pour les CNN : carte de chaleur indiquant quelles régions spatiales influencent le plus la prédiction d'une classe, calculée par rétropropagation des gradients jusqu'à la dernière couche convolutive.

**Invariance à la permutation**
Propriété d'une fonction $f$ telle que $f(\text{Permute}(x)) = f(x)$. Le mean-pooling est invariant à la permutation ; la self-attention sans encodage positionnel également.

**MAE (*Masked Autoencoder*)**
Méthode de pré-entraînement auto-supervisé pour ViT : masquage de 75% des patches, reconstruction des pixels des patches masqués par un décodeur léger. Force l'encodeur à apprendre des représentations sémantiques riches.

**Mixup**
Technique d'augmentation : créer de nouveaux exemples d'entraînement par interpolation linéaire entre paires d'images et de labels ($\tilde{x} = \lambda x_i + (1-\lambda) x_j$). Améliore la calibration et la robustesse.

**mIoU (*mean Intersection over Union*)**
Métrique standard pour la segmentation sémantique : moyenne de l'IoU (intersection sur union) entre les segmentations prédites et les masques de référence, calculée sur toutes les classes.

**Multi-Head Attention (MHA)**
Extension de la self-attention avec $h$ têtes d'attention parallèles, chacune avec ses propres projections Q, K, V de dimension réduite $d_k = d_{\text{model}}/h$. Les sorties sont concaténées et reprojetées.

**Patch embedding**
Transformation qui découpe une image en patches non chevauchants et projette chaque patch aplati dans l'espace de tokens via une transformation linéaire (implémentée comme une convolution $P \times P$ stride $P$).

**Pre-LN / Post-LN**
Deux variantes de placement de la Layer Normalization dans un bloc Transformer. Pre-LN (LN avant MHA et FFN) est plus stable à l'entraînement et préféré dans les ViT modernes. Post-LN (LN après les connexions résiduelles) correspond à l'architecture originale de Vaswani et al.

**RandAugment**
Stratégie d'augmentation automatique : applique séquentiellement $N$ transformations aléatoires (rotation, cisaillement, inversion, posterisation…) choisies dans un espace d'augmentations prédéfini, avec une magnitude $M$ contrôlée.

**Scaled dot-product attention**
Opération fondamentale de l'attention : $\text{softmax}(QK^\top / \sqrt{d_k}) V$. La mise à l'échelle par $1/\sqrt{d_k}$ stabilise les gradients en maintenant une variance constante des scores.

**Stochastic depth (DropPath)**
Technique de régularisation pour les réseaux profonds : ignorer aléatoirement des blocs entiers pendant l'entraînement (avec probabilité croissante avec la profondeur). Analogue du dropout, mais au niveau du bloc résiduel.

**Swin Transformer (*Shifted Window Transformer*)**
Variante de ViT avec attention locale dans des fenêtres de $M \times M$ patches (complexité linéaire en $n$) et mécanisme de fenêtres décalées pour permettre les interactions cross-fenêtres. Architecture hiérarchique multi-échelles.

**Token**
Unité élémentaire d'une séquence. En NLP : un mot ou un sous-mot. En vision : un patch d'image aplati et projeté dans l'espace de tokens. Représenté par un vecteur de dimension $d_{\text{model}}$.

**Token CLS (classification token)**
Token synthétique ajouté en tête de séquence, dont la représentation en sortie de l'encodeur est utilisée comme représentation globale de la séquence pour la classification. Inspiré du token [CLS] de BERT.

**Visual token**
Représentation discrète d'un patch d'image, produite par un tokenizeur visuel (dVAE). Analogue des tokens de sous-mots en NLP. Utilisé comme cible de prédiction dans BEiT.

**W-MSA / SW-MSA (*Window / Shifted-Window Multi-Scale Attention*)**
Variantes de la self-attention utilisées dans le Swin Transformer. W-MSA calcule l'attention indépendamment dans des fenêtres disjointes ; SW-MSA décale les fenêtres de $(M/2, M/2)$ pour permettre les interactions cross-fenêtres.

---

## Bibliographie de référence

### Articles fondateurs du mécanisme d'attention

- **Bahdanau, D., Cho, K., Bengio, Y.** (2015). *Neural Machine Translation by Jointly Learning to Align and Translate*. ICLR 2015. [arXiv:1409.0473] — L'article fondateur de l'attention dans les réseaux de neurones. À lire pour comprendre la genèse du mécanisme.

- **Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., Polosukhin, I.** (2017). *Attention Is All You Need*. NeurIPS 2017. [arXiv:1706.03762] — L'article fondateur du Transformer. Le plus cité de l'histoire du deep learning.

- **Sutskever, I., Vinyals, O., Le, Q. V.** (2014). *Sequence to Sequence Learning with Neural Networks*. NeurIPS 2014. [arXiv:1409.3215] — Les modèles encoder-decoder LSTM dont l'attention vient résoudre les limitations.

### Vision Transformer

- **Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., … Houlsby, N.** (2020). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*. ICLR 2021. [arXiv:2010.11929] — ViT. L'article fondateur.

- **Touvron, H., Cord, M., Douze, M., Massa, F., Sablayrolles, A., Jégou, H.** (2021). *Training Data-Efficient Image Transformers & Distillation Through Attention*. ICML 2021. [arXiv:2012.12877] — DeiT.

- **Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., Guo, B.** (2021). *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows*. ICCV 2021. [arXiv:2103.14030] — Swin. Architecture de référence pour la détection et la segmentation.

- **Zhai, X. et al.** (2022). *Scaling Vision Transformers*. CVPR 2022. [arXiv:2106.04560] — Analyse de l'impact du scaling sur les ViT (ViT-G, 2 milliards de paramètres).

### Pré-entraînement auto-supervisé pour les ViT

- **Bao, H., Dong, L., Piao, S., Wei, F.** (2022). *BEiT: BERT Pre-Training of Image Transformers*. ICLR 2022. [arXiv:2106.08254] — BEiT. Pré-entraînement par masquage de patches.

- **He, K., Chen, X., Xie, S., Li, Y., Dollár, P., Girshick, R.** (2022). *Masked Autoencoders Are Scalable Vision Learners*. CVPR 2022. [arXiv:2111.06377] — MAE. Plus simple que BEiT, plus scalable.

- **Zhou, J. et al.** (2021). *iBOT: Image BERT Pre-Training with Online Tokenizer*. ICLR 2022. [arXiv:2111.07832] — Combine BEiT et DINO pour un pré-entraînement auto-supervisé puissant.

### Interprétabilité et visualisation

- **Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., Batra, D.** (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*. ICCV 2017. [arXiv:1610.02391] — Grad-CAM.

- **Abnar, S., Zuidema, W.** (2020). *Quantifying Attention Flow in Transformers*. ACL 2020. [arXiv:2005.00928] — Attention Rollout.

- **Chefer, H., Gur, S., Wolf, L.** (2021). *Transformer Interpretability Beyond Attention Visualization*. CVPR 2021. [arXiv:2012.09838] — Méthode de relevance propagation pour les Transformers, plus précise que l'Attention Rollout.

### Encodage positionnel

- **Shaw, P., Uszkoreit, J., Vaswani, A.** (2018). *Self-Attention with Relative Position Representations*. NAACL 2018. [arXiv:1803.02155] — Attention relative positionnelle.

- **Raffel, C. et al.** (2020). *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)*. JMLR, 21(140), 1–67. — Section 3.5 sur les encodages positionnels relatifs (T5 bias).

### ViT pour les documents et l'HTR

- **Li, M. et al.** (2021). *TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models*. AAAI 2023. [arXiv:2109.10282] — TrOCR : BEiT encodeur + GPT-2 décodeur pour l'HTR.

- **Blecher, L., Cucurull, G., Scialom, T., Stojnic, R.** (2023). *Nougat: Neural Optical Understanding for Academic Documents*. [arXiv:2308.13418] — Utilise un encodeur Swin pour des documents académiques PDF. Adapté aux documents structurés.

- **Davis, B., Morse, B., Price, B., Tensmeyer, C., Wigington, C.** (2022). *Benchmarking Document Image Transformer Models*. [arXiv:2203.04226] — Évaluation comparée des architectures Transformer sur les tâches de reconnaissance de documents.

### Théorie

- **Tsai, Y.-H. H., Bai, S., Liang, P. P., Kolter, J. Z., Morency, L.-P., Salakhutdinov, R.** (2019). *Multimodal Transformer for Unaligned Multimodal Language Sequences*. ACL 2019. — Analyse théorique des propriétés de la self-attention (équivariance à la permutation, capacité à modéliser des dépendances longue portée).

- **Dong, Y., Cordonnier, J.-B., Loukas, A.** (2021). *Attention is Not All You Need: Pure Attention Loses Rank Doubly Exponentially with Depth*. ICML 2021. — Analyse des limitations théoriques de la self-attention pure sans FFN ni connexions résiduelles. Justifie l'architecture complète du Transformer.

---

*Support de cours rédigé pour le module Computer Vision · Promotion MD5 · Mai 2026*
*Ce document accompagne la séance 2.2 du Jour 2. Il suppose la lecture préalable de la section 2.1.*
*Durée estimée de lecture : 90 minutes.*
