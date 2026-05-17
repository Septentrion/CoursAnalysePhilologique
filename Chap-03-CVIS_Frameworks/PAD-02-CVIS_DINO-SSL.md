# Cours 3.1 — DINO et les représentations auto-supervisées

**Module 3 · Computer Vision appliquée aux manuscrits médiévaux · MD5**

---

> *« Le défi de la vision par ordinateur pour les humanités n'est pas l'architecture — c'est l'annotation. Un million d'images étiquetées, c'est le travail d'une équipe entière pendant des années. DINO propose une autre voie : apprendre sans étiquettes. »*

---

## Introduction : le goulot d'étranglement de l'annotation

L'apprentissage supervisé est puissant, mais il repose sur une hypothèse coûteuse : l'existence d'un grand nombre d'exemples annotés. Pour ImageNet (1,2 million d'images, 1 000 classes), des milliers de personnes ont passé des années à annoter. Pour les manuscrits médiévaux, la situation est radicalement différente.

Constituer un dataset HTR de qualité sur des manuscrits médiévaux coûte entre 0,5 et 2 heures par page de transcription experte — soit 300 à 500 pages par an et par paléographe. La BnF seule conserve plus de 100 000 manuscrits. À ce rythme, annoter 1% du fonds numérisé prendrait plusieurs siècles-personnes de travail.

**L'auto-supervision** est une réponse structurelle à ce problème : apprendre des représentations visuelles riches et généralisables *sans* annotations humaines, en exploitant la structure interne des données elles-mêmes.

DINO (*Self-DIstillation with NO labels*, Caron et al., 2021) est l'une des avancées les plus significatives dans cette direction — et ses propriétés émergentes, notamment la segmentation sémantique spontanée, en font un outil particulièrement adapté au traitement de documents patrimoniaux.

---

## 1. Panorama de l'apprentissage auto-supervisé en vision

### 1.1 Principes généraux

L'apprentissage auto-supervisé (*self-supervised learning*, SSL) repose sur la création de tâches d'entraînement synthétiques — les **tâches prétextes** (*pretext tasks*) — dont les labels sont générés automatiquement depuis les données brutes. Le modèle n'a pas besoin d'annotation humaine : il apprend en résolvant ces tâches, et les représentations qu'il développe sont transferables à des tâches en aval (*downstream tasks*).

La philosophie fondamentale : si un modèle doit résoudre une tâche difficile qui nécessite de *comprendre* le contenu visuel (et non seulement mémoriser des patterns de surface), les représentations intermédiaires qu'il développe seront riches et généralisables.

### 1.2 Les premières tâches prétextes (2014–2019)

**Prédiction de rotation (Gidaris et al., 2018)**
L'image est tournée de 0°, 90°, 180° ou 270°. Le modèle doit prédire l'angle. Pour distinguer une image à l'endroit d'une image retournée, il faut reconnaître des structures sémantiques (visages, objets, textes) — pas seulement des textures.

**Puzzle Jigsaw (Noroozi & Favaro, 2016)**
L'image est découpée en $3 \times 3$ tuiles mélangées. Le modèle prédit la permutation. Résoudre ce puzzle nécessite de comprendre les relations spatiales entre parties de l'image.

**Complétion de couleur (Zhang et al., 2016)**
L'image est convertie en niveaux de gris. Le modèle prédit les couleurs. Coloriser correctement un ciel, une peau, une feuille nécessite de reconnaître le contenu sémantique.

**Limite commune de ces approches**
Ces tâches prétextes sont *ad hoc* : chaque tâche encode des hypothèses spécifiques sur ce qui est une bonne représentation. Elles produisent des features de qualité inégale selon la tâche en aval. Et surtout, elles ne scalent pas aussi bien que l'apprentissage supervisé — augmenter les données ne produit pas d'amélioration aussi nette.

### 1.3 L'apprentissage contrastif (2018–2021)

L'apprentissage **contrastif** reformule le SSL comme un problème de métrique : apprendre un espace où des vues *augmentées* de la même image sont proches, et des vues d'images différentes sont éloignées.

**SimCLR (Chen et al., 2020)**

Pour un mini-batch de $N$ images, on génère $2N$ vues par data augmentation aléatoire (crop, flip, jitter de couleur, flou gaussien). Pour chaque vue $i$, la vue positive est son augmentation jumelle $j$ ; les $2N - 2$ autres vues sont des exemples négatifs.

La **NT-Xent loss** (*Normalized Temperature-scaled Cross-Entropy*) maximise la similarité cosinus entre paires positives et minimise celle entre paires négatives :

$$\mathcal{L}_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k=1}^{2N} \mathbf{1}_{[k \neq i]} \exp(\text{sim}(z_i, z_k) / \tau)}$$

où $z_i = g(f(x_i))$ est la représentation projetée (encodeur $f$ + tête de projection $g$) et $\tau$ est un paramètre de température.

**MoCo (He et al., 2020)**

Introduit une **queue de représentations négatives** et un encodeur momentum (*momentum encoder*) pour stabiliser l'entraînement sans nécessiter de très grands batchs :

$$\theta_k \leftarrow m \cdot \theta_k + (1 - m) \cdot \theta_q$$

où $\theta_q$ sont les paramètres de l'encodeur requête (mis à jour par gradient) et $\theta_k$ ceux de l'encodeur clé (mis à jour par EMA, *Exponential Moving Average*). Ce mécanisme sera réutilisé dans DINO.

**Limite centrale de l'apprentissage contrastif**

Les méthodes contrastives nécessitent des **exemples négatifs** — des paires d'images différentes qu'on apprend à éloigner dans l'espace de représentation. Or :

1. **Le coût computationnel des négatifs.** Pour être efficace, un batchs très grand (SimCLR : 8 192 images) ou une queue longue (MoCo : 65 536 représentations) est nécessaire, ce qui demande une infrastructure GPU conséquente.

2. **Le problème des faux négatifs.** Dans un batch, deux images d'objets similaires (deux chats, deux pages de manuscrit du même copiste) sont traitées comme négatives — ce qui pousse le modèle à les éloigner dans l'espace de représentation, à l'encontre de ce qu'on souhaite.

3. **La dépendance aux augmentations.** Les augmentations définissent implicitement ce qui est « invariant » (et donc ne doit pas distinguer les vues d'un même exemple). Choisir les bonnes augmentations est critique et requiert une connaissance du domaine.

### 1.4 La self-distillation : se passer des négatifs

Une famille de méthodes plus récentes (*BYOL*, *SimSiam*, *DINO*) montrent qu'on peut apprendre sans exemples négatifs, par **distillation d'un réseau sur lui-même** (*self-distillation* ou *self-knowledge distillation*).

L'intuition : au lieu de comparer une vue à d'autres images (négatifs), on compare une vue à la représentation que *le modèle lui-même* produit sur une autre vue de la même image. Si les deux vues sont cohérentes (montrent le même contenu, même partiellement), leurs représentations doivent converger.

La difficulté : sans mécanisme de régularisation, ce type d'apprentissage converge trivialement vers la **solution effondrée** (*collapsed solution*) — toutes les représentations identiques, indépendamment de l'entrée. DINO propose une solution élégante à ce problème.

---

## 2. DINO v1 : architecture et mécanismes

### 2.1 Vue d'ensemble

DINO (*Self-**DI**stillation with **NO** labels*, Caron et al., 2021) est un framework d'apprentissage auto-supervisé pour les Vision Transformers, reposant sur trois composants :

- Une architecture **student-teacher** avec mise à jour momentum du teacher.
- Une stratégie **multi-crop** qui génère plusieurs vues à différentes échelles.
- Un mécanisme de **centering + sharpening** pour prévenir l'effondrement.

```
              ┌─────────────┐
  Vue globale │   Teacher   │──→ représentation centrée
  x_global ──→│   (frozen)  │    p_t = softmax((h_t - c) / τ_t)
              └─────────────┘
                    ↑ EMA
                    │ θ_t ← m·θ_t + (1-m)·θ_s
              ┌─────────────┐
  Vues locales│   Student   │──→ représentation affûtée
  x_local  ──→│  (gradient) │    p_s = softmax(h_s / τ_s)
              └─────────────┘
                    │
              Loss : H(p_t, p_s) = -∑ p_t log p_s
```

### 2.2 Architecture student-teacher

**Deux réseaux identiques** (ViT-S, ViT-B ou ResNet) partagent la même architecture, mais pas les mêmes paramètres.

- Le **student** $f_{\theta_s}$ est mis à jour par descente de gradient stochastique (rétropropagation normale).
- Le **teacher** $f_{\theta_t}$ est mis à jour par **EMA** (*Exponential Moving Average*) des paramètres du student — il ne reçoit aucun gradient direct :

$$\theta_t \leftarrow m \cdot \theta_t + (1 - m) \cdot \theta_s$$

avec $m \in [0.996, 1.0]$ (typiquement $m = 0.9996$ en fin d'entraînement, avec un schedule cosinus depuis $m_0 = 0.996$).

**Pourquoi le teacher par EMA ?**

Le teacher est une version temporellement lissée du student — ses paramètres sont plus stables et moins susceptibles de converger vers une solution effondrée. C'est un mécanisme de *slow moving average* qui agit comme une cible régularisante pour le student. Plus $m$ est proche de 1, plus le teacher est stable (mais lent à évoluer).

**Note :** contrairement à la distillation de connaissances classique, le teacher n'est pas un modèle externe pré-entraîné — il est co-entraîné avec le student. C'est cette propriété récursive qui donne son nom à l'approche : *self*-distillation.

### 2.3 La stratégie multi-crop

Au lieu de générer 2 vues par image (comme SimCLR), DINO génère $V$ vues à différentes échelles :

- 2 **vues globales** (*global crops*) : crops couvrant > 50% de l'image. Même échelle que l'image d'entrée standard (224 × 224 pour ViT-B/16).
- $V - 2$ **vues locales** (*local crops*) : crops couvrant 20% à 50% de l'image, redimensionnés à 96 × 96.

La **règle d'entraînement** est asymétrique :
- Le **teacher** ne traite que les vues globales (vue large du contenu).
- Le **student** traite *toutes* les vues (globales et locales).

**Intuition :** le student apprend que sa vue locale (un crop partiel de la page) doit produire une représentation cohérente avec la vue globale du teacher (toute la page). Cela force le student à *inférer* le contenu global depuis une vue partielle — ce qui nécessite de comprendre les structures sémantiques, pas seulement les textures locales.

Pour les manuscrits : une vue locale d'une ligne de texte doit être cohérente avec la représentation globale de la page. Le modèle apprend que « cette séquence de jambages gothiques sur fond crème » et « cette page de manuscrit du XIVe siècle » sont sémantiquement liés.

```python
import torch
import torchvision.transforms as T
import random
from PIL import Image

class MultiCropAugmentation:
    """
    Génère les vues multi-crop de DINO.
    """
    def __init__(
        self,
        n_global: int = 2,
        n_local: int = 6,
        global_scale: tuple = (0.4, 1.0),
        local_scale: tuple = (0.05, 0.4),
        global_size: int = 224,
        local_size: int = 96,
    ):
        # Augmentations communes (indépendantes de l'échelle)
        color_jitter = T.ColorJitter(0.4, 0.4, 0.2, 0.1)
        augment_base = [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([color_jitter], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

        # Transformation globale
        self.global_transform = T.Compose([
            T.RandomResizedCrop(global_size, scale=global_scale,
                                interpolation=T.InterpolationMode.BICUBIC),
            *augment_base
        ])

        # Transformation locale (crops plus petits)
        self.local_transform = T.Compose([
            T.RandomResizedCrop(local_size, scale=local_scale,
                                interpolation=T.InterpolationMode.BICUBIC),
            *augment_base
        ])

        self.n_global = n_global
        self.n_local = n_local

    def __call__(self, img: Image.Image) -> list[torch.Tensor]:
        """
        Retourne n_global + n_local vues augmentées.
        Les n_global premières sont les vues globales (pour le teacher).
        """
        vues = []
        for _ in range(self.n_global):
            vues.append(self.global_transform(img))
        for _ in range(self.n_local):
            vues.append(self.local_transform(img))
        return vues
```

### 2.4 La fonction de perte

DINO minimise l'**entropie croisée** entre les distributions de sortie du teacher et du student.

Les sorties du réseau (après la tête de projection MLP) sont des vecteurs de dimension $K$ (typiquement $K = 65\,536$). On les transforme en distributions de probabilité par softmax avec température :

$$p^t_k = \frac{\exp\!\left((h^t_k - c_k) / \tau_t\right)}{\sum_{k'} \exp\!\left((h^t_{k'} - c_{k'}) / \tau_t\right)}, \quad p^s_k = \frac{\exp\!\left(h^s_k / \tau_s\right)}{\sum_{k'} \exp\!\left(h^s_{k'} / \tau_s\right)}$$

La perte est la somme de l'entropie croisée sur toutes les paires (vue du student, vue globale du teacher) :

$$\mathcal{L} = \sum_{\substack{x \in \{\text{vues globales}\} \\ x' \in \{\text{toutes les vues}\} \\ x \neq x'}} H(p^t(x),\ p^s(x')) = -\sum_k p^t_k(x) \log p^s_k(x')$$

Le gradient est calculé uniquement par rapport aux paramètres du student $\theta_s$. Le teacher $\theta_t$ est arrêté du gradient (*stop-gradient*) — il reçoit ses mises à jour uniquement via l'EMA.

### 2.5 Prévention de l'effondrement : centering et sharpening

Sans mécanisme de régularisation, le modèle converge vers la solution triviale : toutes les représentations identiques (vecteur constant $c$), ce qui minimise trivialement l'entropie croisée. DINO utilise deux mécanismes complémentaires pour l'éviter.

**Centering (centrage)**

Le vecteur centre $c \in \mathbb{R}^K$ est soustrait des sorties du teacher avant le softmax :

$$c \leftarrow \lambda_c \cdot c + (1 - \lambda_c) \cdot \frac{1}{B} \sum_{i=1}^B h^t_i$$

où $\lambda_c = 0.9$ (momentum du centrage) et $B$ est la taille du batch.

**Effet :** le centrage empêche la **domination d'une seule dimension** — si une dimension de $h^t$ est systématiquement beaucoup plus grande que les autres, le softmax lui attribuerait une probabilité proche de 1 (effondrement vers une distribution ponctuelle constante). En soustrayant la moyenne courante, on maintient les logits centrés autour de 0.

**Sharpening (affûtage)**

Le teacher utilise une température $\tau_t$ **plus basse** que le student ($\tau_t = 0.04$ vs $\tau_s \in [0.1, 0.5]$). Une température basse rend la distribution du teacher **plus piquée** — elle concentre la masse de probabilité sur quelques dimensions.

**Effet :** le sharpening empêche l'**uniformisation** — une distribution uniforme sur $K$ dimensions minimise l'entropie croisée avec n'importe quelle distribution uniforme, ce qui est une autre solution triviale. En forçant le teacher à être confiant (distribution piquée), on force le student à prédire des représentations discriminantes.

**L'équilibre centering-sharpening**

| Mécanisme | Prévient | Risque si absent |
|-----------|----------|-----------------|
| Centering | Domination d'une dimension | Effondrement vers distribution ponctuelle |
| Sharpening | Uniformisation | Effondrement vers distribution uniforme |

Ces deux mécanismes agissent en sens opposé et doivent être calibrés conjointement. C'est l'un des aspects les plus délicats de l'entraînement DINO — et l'une des raisons pour lesquelles les implémentations de référence incluent des schedules de température et des paramètres de momentum soigneusement ajustés.

```python
class DINOLoss(torch.nn.Module):
    """
    Loss DINO : entropie croisée + centering du teacher.
    """
    def __init__(
        self,
        out_dim: int,
        n_global_crops: int = 2,
        n_local_crops: int = 6,
        tau_student: float = 0.1,
        tau_teacher: float = 0.04,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.n_global = n_global_crops
        self.n_local = n_local_crops
        self.tau_s = tau_student
        self.tau_t = tau_teacher
        self.center_momentum = center_momentum
        # Vecteur de centrage (initialisé à 0)
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(
        self,
        student_out: list[torch.Tensor],  # Toutes les vues
        teacher_out: list[torch.Tensor],  # Vues globales seulement
    ) -> torch.Tensor:
        """
        student_out : liste de n_global + n_local tenseurs (B, K)
        teacher_out : liste de n_global tenseurs (B, K)
        """
        # Distribution du teacher : centring + sharpening
        teacher_probs = [
            torch.softmax((t - self.center) / self.tau_t, dim=-1).detach()
            for t in teacher_out
        ]
        # Distribution du student : sharpening uniquement
        student_probs = [
            torch.log_softmax(s / self.tau_s, dim=-1)
            for s in student_out
        ]

        # Entropie croisée sur toutes les paires
        loss = 0.0
        n_paires = 0
        for t_idx, t_prob in enumerate(teacher_probs):
            for s_idx, s_logprob in enumerate(student_probs):
                # Exclure les paires identiques (même vue)
                if t_idx == s_idx:
                    continue
                loss += -(t_prob * s_logprob).sum(dim=-1).mean()
                n_paires += 1

        loss /= n_paires

        # Mise à jour du vecteur de centrage (EMA sur le batch)
        batch_center = torch.cat(teacher_out).mean(dim=0, keepdim=True)
        self.center = (
            self.center * self.center_momentum
            + batch_center * (1 - self.center_momentum)
        )

        return loss
```

### 2.6 La propriété émergente : segmentation sans annotation

La propriété la plus surprenante de DINO est l'émergence spontanée d'une **segmentation sémantique** dans les attention maps des dernières couches du ViT, sans qu'aucune supervision de segmentation n'ait été fournie.

Cette propriété a été observée et documentée dans l'article original (Figure 1 de Caron et al., 2021) : les têtes d'attention de la dernière couche de DINO-ViT-S/8 segmentent les objets avec une précision remarquable sur des images COCO, ImageNet et — ce qui nous intéresse — des documents.

**Pourquoi cette propriété émerge-t-elle ?**

Plusieurs facteurs y contribuent :

1. **La multi-crop force la cohérence locale-globale.** Pour que la représentation d'un crop local soit cohérente avec la représentation globale du teacher, le modèle doit apprendre à identifier les *régions sémantiquement cohérentes* de l'image — ce qui est exactement la définition d'un segment.

2. **L'architecture ViT amplifie cet effet.** Le mécanisme de self-attention globale (pas de biais de localité) permet à chaque patch de « décider » de s'aligner sur d'autres patches sémantiquement proches, même distants spatialement. Les têtes d'attention peuvent se spécialiser sur ce type de groupement.

3. **L'absence de labels évite le raccourci supervisé.** Un modèle supervisé sur ImageNet apprend des features discriminantes pour les 1 000 classes — ce qui peut inclure des features de texture ou de contexte non sémantiques. Sans label, DINO est forcé d'apprendre des features qui permettent de reconstruire les vues cohérentes — des features structurelles et sémantiques.

---

## 3. DINOv2 : vers la robustesse et l'universalité

### 3.1 Motivations et améliorations

DINOv2 (Oquab et al., 2023) reprend les principes de DINO v1 et les améliore sur trois axes : les données, la loss, et l'architecture.

**Axe 1 — Données curées à grande échelle : LVD-142M**

DINO v1 était entraîné sur ImageNet-22k (~14 millions d'images). DINOv2 est entraîné sur **LVD-142M** (*Large-scale Varied Dataset*, 142 millions d'images), un dataset propriétaire curé automatiquement depuis le web par un pipeline de filtrage en plusieurs étapes :

1. **Déduplication** : élimination des images quasi-identiques par hashing perceptuel.
2. **Filtrage de qualité** : elimination des images de mauvaise résolution ou de contenu inapproprié.
3. **Sélection par similarité** : pour chaque image d'un pool de sources web, on calcule sa similarité avec un dataset curé de référence (ImageNet-22k) ; seules les images suffisamment proches d'images de référence sont conservées.

Ce pipeline garantit la qualité et la diversité du dataset sans annotation humaine explicite.

**Axe 2 — Loss composite : DINO + iBOT + KoLeo**

DINOv2 combine trois objectifs d'entraînement :

$$\mathcal{L}_{\text{DINOv2}} = \mathcal{L}_{\text{DINO}} + \lambda_{\text{iBOT}} \mathcal{L}_{\text{iBOT}} + \lambda_{\text{KoLeo}} \mathcal{L}_{\text{KoLeo}}$$

**$\mathcal{L}_{\text{DINO}}$** : la loss standard décrite en section 2, appliquée au token CLS.

**$\mathcal{L}_{\text{iBOT}}$** (*image BERT pretraining with Online Tokenizer*, Zhou et al., 2021) : une loss de masquage de patches, analogue à BEiT. Des patches sont masqués dans la vue du student ; le teacher prédit leurs représentations. Cette composante force l'apprentissage de représentations au niveau des patches (localisation fine), pas seulement au niveau de l'image entière (token CLS).

$$\mathcal{L}_{\text{iBOT}} = -\sum_{i \in \mathcal{M}} \sum_k p^t_k(x_i) \log p^s_k(\hat{x}_i)$$

où $\mathcal{M}$ est l'ensemble des patches masqués, $x_i$ est le patch original (dans la vue du teacher), et $\hat{x}_i$ est le patch masqué (dans la vue du student).

**$\mathcal{L}_{\text{KoLeo}}$** (*Kolmogorov-Leonenko entropy estimator*) : un terme de régularisation qui encourage l'**étalement uniforme** des représentations dans l'espace de features. Il pénalise les représentations qui se concentrent dans des sous-régions denses de l'espace :

$$\mathcal{L}_{\text{KoLeo}} = -\frac{1}{n} \sum_{i=1}^n \log\!\left(\frac{n \cdot d_i^{(k)}}{\text{aire}}\right)$$

où $d_i^{(k)}$ est la distance au $k$-ième plus proche voisin de la représentation $i$. Cette loss décourage les **points de collapse partiel** — des sous-groupes de représentations qui s'effondrent sur le même vecteur tout en restant distincts des autres groupes.

**Axe 3 — Distillation depuis un grand modèle**

L'entraînement commence par un ViT-g/14 (giant, 1,1 milliard de paramètres) entraîné de zéro. Les représentations de ce grand modèle servent ensuite de cibles pour la distillation de modèles plus petits (ViT-S, ViT-B, ViT-L, ViT-G). Cette cascade améliore les performances des petits modèles au-delà de ce qu'ils atteignent en entraînement direct.

### 3.2 Performances et propriétés de DINOv2

DINOv2 établit ou co-établit l'état de l'art sur une large gamme de tâches *sans fine-tuning* (features gelées + sonde linéaire) :

| Tâche | Dataset | Metric | DINOv2-L | Supervisé (référence) |
|-------|---------|--------|-----------|----------------------|
| Classification (sonde linéaire) | ImageNet-1k | top-1 | 86,5% | 87,1% (ViT-L sup.) |
| Segmentation sémantique | ADE20K | mIoU | 53,1% | 58,1% (Mask2Former) |
| Profondeur monoculaire | NYUd | δ₁ | 95,0% | 95,4% (AdaBins) |
| Correspondances visuelles | SPair-71k | PCK | 91,6% | — |
| Récupération d'images | Oxford Hard | mAP | 79,4% | — |

La colonne « supervisé (référence) » représente des modèles fine-tunés avec labels — DINOv2 s'en rapproche de façon remarquable avec des features gelées.

### 3.3 Configurations disponibles et choix pour notre projet

| Modèle | Paramètres | Patch size | d_model | Recommandé pour |
|--------|-----------|------------|---------|----------------|
| DINOv2-S/14 | 21 M | 14 × 14 | 384 | Prototypage rapide, inférence CPU |
| DINOv2-B/14 | 86 M | 14 × 14 | 768 | Équilibre qualité/vitesse (**notre choix**) |
| DINOv2-L/14 | 307 M | 14 × 14 | 1024 | Meilleure qualité, GPU nécessaire |
| DINOv2-G/14 | 1100 M | 14 × 14 | 1536 | Recherche, très lent |

Pour notre projet, **DINOv2-B/14** est un bon compromis : 86 M de paramètres (identique à ViT-B), patch size 14 × 14 (plus fin que les 16 × 16 de ViT-B, donc meilleure résolution spatiale), et des features reconnues pour leurs propriétés de segmentation.

---

## 4. Application : clustering de pages manuscrites par style d'écriture

### 4.1 Motivation

L'une des premières tâches d'un projet de transcription à grande échelle est l'**identification des mains** (*scribal hand identification*) : regrouper les pages selon le copiste qui les a écrites. Des mains différentes produisent des styles graphiques différents, et un modèle HTR entraîné sur une main donnée sera moins performant sur une main différente. Identifier les mains en amont permet de :

- Choisir le modèle HTR le plus adapté à chaque sous-corpus.
- Détecter les changements de main en cours de manuscrit (ce qui peut indiquer une lacune, une interpolation, ou une copie composite).
- Regrouper les pages pour le fine-tuning ciblé.

Les features DINOv2 permettent de réaliser ce clustering **sans annotation** et **sans entraînement spécifique** — en exploitant directement les représentations génériques apprises par auto-supervision.

### 4.2 Pipeline d'extraction de features

```python
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from transformers import AutoImageProcessor, AutoModel
import torchvision.transforms as T

# ─── Chargement du modèle ──────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Chargement de DINOv2-B/14...")
dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
dino_model = AutoModel.from_pretrained("facebook/dinov2-base").to(DEVICE).eval()

# Paramètres DINOv2-B/14
PATCH_SIZE = 14
IMG_SIZE = 518     # Taille d'entrée recommandée pour DINOv2 (multiple de patch_size)
N_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2   # = 1369 patches
D_MODEL = 768

# ─── Extraction de features ────────────────────────────────────────────────────
def extraire_features_dino(
    chemin_image: str,
    retourner_patches: bool = False
) -> dict[str, np.ndarray]:
    """
    Extrait les features DINOv2 d'une image de page de manuscrit.

    Args:
        chemin_image   : chemin vers le scan
        retourner_patches : si True, retourne aussi les features des patches

    Returns:
        Dictionnaire avec :
          - 'cls'     : feature globale (768,) — représentation de la page entière
          - 'patches' : features des patches (N_PATCHES, 768) si demandé
          - 'attn'    : attention maps de la dernière couche (12, N_PATCHES+1, N_PATCHES+1)
    """
    img = Image.open(chemin_image).convert("RGB")

    # Redimensionnement à la taille d'entrée DINOv2 recommandée
    img_resized = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    inputs = dino_processor(images=img_resized, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = dino_model(
            **inputs,
            output_attentions=True,
            output_hidden_states=False
        )

    resultats = {}

    # Feature globale : token CLS de la dernière couche
    cls_feature = outputs.last_hidden_state[:, 0, :]   # (1, 768)
    resultats["cls"] = cls_feature[0].cpu().numpy()

    # Features des patches (si demandé)
    if retourner_patches:
        patch_features = outputs.last_hidden_state[:, 1:, :]  # (1, N_patches, 768)
        resultats["patches"] = patch_features[0].cpu().numpy()

    # Attention maps de la dernière couche
    attn_last = outputs.attentions[-1][0]   # (n_heads, N_tokens, N_tokens)
    resultats["attn"] = attn_last.cpu().numpy()

    return resultats


def extraire_features_corpus(
    repertoire: str,
    extensions: list[str] = [".jpg", ".jpeg", ".png", ".tif", ".tiff"],
    verbose: bool = True
) -> tuple[np.ndarray, list[str]]:
    """
    Extrait les features CLS de toutes les images d'un répertoire.

    Returns:
        features  : (N_images, 768) — une ligne par image
        chemins   : liste des chemins correspondants
    """
    chemins = []
    for ext in extensions:
        chemins.extend(Path(repertoire).glob(f"*{ext}"))
    chemins = sorted(chemins)

    if not chemins:
        raise ValueError(f"Aucune image trouvée dans {repertoire}")

    if verbose:
        print(f"  {len(chemins)} images trouvées dans {repertoire}")

    features = []
    for i, chemin in enumerate(chemins):
        if verbose and i % 10 == 0:
            print(f"  Traitement {i+1}/{len(chemins)} : {chemin.name}")
        try:
            res = extraire_features_dino(str(chemin))
            features.append(res["cls"])
        except Exception as e:
            print(f"    [ERREUR] {chemin.name} : {e}")
            features.append(np.zeros(D_MODEL))  # Feature nulle en cas d'erreur

    return np.array(features), [str(c) for c in chemins]
```

### 4.3 Réduction de dimension et visualisation

Les features DINOv2 sont des vecteurs de dimension 768 — impossibles à visualiser directement. On utilise **UMAP** (*Uniform Manifold Approximation and Projection*) pour les projeter en 2D, puis on colore les points selon des métadonnées disponibles (siècle, type de document, bibliothèque) pour interpréter la structure de l'espace.

```python
import umap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def visualiser_espace_features(
    features: np.ndarray,
    chemins: list[str],
    metadonnees: dict[str, list] = None,
    n_clusters: int = 5,
    nom_figure: str = "clustering"
) -> np.ndarray:
    """
    Réduit les features en 2D par UMAP et visualise le clustering.

    Args:
        features     : (N, 768) — features DINOv2
        chemins      : liste de N chemins d'images
        metadonnees  : dict optionnel {nom: [valeur_par_image]}
                       ex: {"siecle": [13, 14, 14, 15, ...], "scriptorium": [...]}
        n_clusters   : nombre de clusters K-means
        nom_figure   : préfixe pour la sauvegarde

    Returns:
        labels_kmeans : (N,) — étiquettes de cluster
    """
    N = features.shape[0]
    print(f"  {N} images, features de dimension {features.shape[1]}")

    # ── Normalisation L2 des features ──────────────────────────────────────────
    # Les features DINOv2 sont mieux comparées en similarité cosinus.
    # Normaliser en L2 équivaut à travailler en cosinus.
    features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)

    # ── Réduction UMAP ─────────────────────────────────────────────────────────
    print("  Réduction UMAP en cours...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(15, N - 1),    # Nombre de voisins locaux
        min_dist=0.1,                   # Compacité des clusters
        metric="cosine",                # Métrique cosinus adaptée aux features SSL
        random_state=42,
        verbose=False
    )
    embedding = reducer.fit_transform(features_norm)   # (N, 2)
    print(f"  Embedding UMAP : {embedding.shape}")

    # ── Clustering K-means ─────────────────────────────────────────────────────
    print(f"  Clustering K-means avec k={n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels_kmeans = kmeans.fit_predict(features_norm)

    # Score de silhouette (qualité du clustering)
    if N > n_clusters:
        sil = silhouette_score(features_norm, labels_kmeans, metric="cosine")
        print(f"  Score de silhouette : {sil:.3f} (1.0 = parfait, 0.0 = aléatoire)")

    # ── Visualisation ──────────────────────────────────────────────────────────
    n_cols = 1 + (len(metadonnees) if metadonnees else 0)
    fig, axes = plt.subplots(1, n_cols, figsize=(8 * n_cols, 7))
    if n_cols == 1:
        axes = [axes]

    # Panneau 1 : clustering K-means
    couleurs_kmeans = cm.tab10(np.linspace(0, 1, n_clusters))
    for k in range(n_clusters):
        masque = labels_kmeans == k
        axes[0].scatter(
            embedding[masque, 0], embedding[masque, 1],
            c=[couleurs_kmeans[k]], label=f"Cluster {k} (n={masque.sum()})",
            s=40, alpha=0.7, edgecolors="white", linewidths=0.3
        )
    axes[0].set_title(f"K-means (k={n_clusters})\n"
                      f"Silhouette : {sil:.3f}" if N > n_clusters else f"K-means (k={n_clusters})",
                      fontsize=12)
    axes[0].legend(fontsize=8, markerscale=1.5)
    axes[0].set_xlabel("UMAP dim. 1")
    axes[0].set_ylabel("UMAP dim. 2")
    axes[0].grid(True, alpha=0.2)

    # Panneaux suivants : métadonnées
    if metadonnees:
        for panneau_idx, (nom_meta, valeurs) in enumerate(metadonnees.items(), start=1):
            valeurs_arr = np.array(valeurs)
            valeurs_uniques = np.unique(valeurs_arr)
            couleurs_meta = cm.Set2(np.linspace(0, 1, len(valeurs_uniques)))

            for i, val in enumerate(valeurs_uniques):
                masque = valeurs_arr == val
                axes[panneau_idx].scatter(
                    embedding[masque, 0], embedding[masque, 1],
                    c=[couleurs_meta[i]], label=str(val),
                    s=40, alpha=0.7, edgecolors="white", linewidths=0.3
                )
            axes[panneau_idx].set_title(f"Métadonnée : {nom_meta}", fontsize=12)
            axes[panneau_idx].legend(fontsize=8, markerscale=1.5)
            axes[panneau_idx].set_xlabel("UMAP dim. 1")
            axes[panneau_idx].set_ylabel("UMAP dim. 2")
            axes[panneau_idx].grid(True, alpha=0.2)

    plt.suptitle("Espace de features DINOv2-B/14 — Manuscrits médiévaux",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(f"figures/{nom_figure}_umap_dino.png", dpi=150, bbox_inches="tight")
    plt.show()

    return labels_kmeans
```

### 4.4 Analyse des clusters : identification des mains

Une fois les clusters identifiés, il faut les interpréter. Pour chaque cluster, on peut :

**1. Afficher des exemplaires représentatifs** (les images les plus proches du centroïde)

```python
def afficher_exemplaires_clusters(
    features: np.ndarray,
    chemins: list[str],
    labels: np.ndarray,
    n_exemplaires: int = 3,
    nom_figure: str = "clusters"
) -> None:
    """
    Affiche les n_exemplaires images les plus proches du centroïde de chaque cluster.
    """
    n_clusters = len(np.unique(labels))
    fig, axes = plt.subplots(
        n_clusters, n_exemplaires,
        figsize=(5 * n_exemplaires, 4 * n_clusters)
    )
    if n_clusters == 1:
        axes = axes[np.newaxis, :]

    features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)

    for k in range(n_clusters):
        masque = np.where(labels == k)[0]
        features_cluster = features_norm[masque]

        # Centroïde du cluster (dans l'espace normalisé)
        centroide = features_cluster.mean(axis=0)
        centroide /= np.linalg.norm(centroide) + 1e-8

        # Distance cosinus de chaque image au centroïde
        # (distance cosinus = 1 - similarité cosinus)
        distances = 1 - features_cluster @ centroide
        idx_proches = np.argsort(distances)[:n_exemplaires]

        for col, idx in enumerate(idx_proches):
            chemin_img = chemins[masque[idx]]
            try:
                img = Image.open(chemin_img).convert("RGB")
                img_thumb = img.resize((300, 400), Image.LANCZOS)
                axes[k, col].imshow(img_thumb)
                axes[k, col].set_title(
                    f"Cluster {k}\n{Path(chemin_img).stem}\n"
                    f"dist={distances[idx]:.3f}",
                    fontsize=8
                )
            except:
                axes[k, col].set_title(f"[Erreur]\n{Path(chemin_img).stem}", fontsize=8)
            axes[k, col].axis("off")

    plt.suptitle("Exemplaires représentatifs par cluster (images les plus proches du centroïde)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"figures/{nom_figure}_exemplaires.png", dpi=120, bbox_inches="tight")
    plt.show()
```

**2. Analyser les features de patches pour localiser ce qui distingue les clusters**

```python
def carte_discrimination_cluster(
    img_cluster_a: str,
    img_cluster_b: str,
    nom_figure: str = "discrimination"
) -> None:
    """
    Produit une carte indiquant quelles zones d'une image la distinguent
    d'une image d'un autre cluster, via la différence de features de patches.
    """
    # Extraire les features de patches des deux images
    res_a = extraire_features_dino(img_cluster_a, retourner_patches=True)
    res_b = extraire_features_dino(img_cluster_b, retourner_patches=True)

    patches_a = res_a["patches"]   # (N_patches, 768)
    patches_b = res_b["patches"]

    # Norme de la différence par patch (mesure de dissimilarité locale)
    diff = np.linalg.norm(patches_a - patches_b, axis=1)  # (N_patches,)
    n_side = int(np.sqrt(len(diff)))
    diff_map = diff.reshape(n_side, n_side)
    diff_map = (diff_map - diff_map.min()) / (diff_map.max() - diff_map.min() + 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Différence de features de patches DINOv2", fontsize=12)

    for ax, chemin, titre in zip(
        axes[:2],
        [img_cluster_a, img_cluster_b],
        ["Image A", "Image B"]
    ):
        img = Image.open(chemin).convert("RGB").resize((518, 518))
        ax.imshow(img)
        ax.set_title(titre, fontsize=10)
        ax.axis("off")

    im = axes[2].imshow(diff_map, cmap="hot", vmin=0, vmax=1)
    axes[2].set_title("||features_A - features_B|| par patch\n"
                      "(zones chaudes = plus différentes)", fontsize=10)
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046)

    plt.tight_layout()
    plt.savefig(f"figures/{nom_figure}_diff_patches.png", dpi=150, bbox_inches="tight")
    plt.show()
```

### 4.5 Déterminer le nombre optimal de clusters

Le choix de $k$ dans le K-means est un paramètre critique. Deux approches complémentaires :

```python
def trouver_k_optimal(
    features: np.ndarray,
    k_min: int = 2,
    k_max: int = 15
) -> None:
    """
    Affiche la courbe d'inertie (méthode du coude) et les scores de silhouette
    pour aider à choisir k.
    """
    features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)

    inerties = []
    silhouettes = []
    ks = range(k_min, k_max + 1)

    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(features_norm)
        inerties.append(km.inertia_)
        if k > 1 and features.shape[0] > k:
            silhouettes.append(silhouette_score(features_norm, labels, metric="cosine"))
        else:
            silhouettes.append(0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Courbe du coude (inertie)
    ax1.plot(ks, inerties, "o-", color="steelblue", linewidth=2)
    ax1.set_xlabel("Nombre de clusters k")
    ax1.set_ylabel("Inertie (intra-cluster SSE)")
    ax1.set_title("Méthode du coude\n(choisir k au point de flexion)")
    ax1.grid(True, alpha=0.3)

    # Score de silhouette
    k_opt = k_min + int(np.argmax(silhouettes))
    ax2.plot(list(ks)[1:], silhouettes[1:], "o-", color="darkorange", linewidth=2)
    ax2.axvline(k_opt, color="red", linestyle="--", alpha=0.7,
                label=f"k optimal = {k_opt} (silhouette = {max(silhouettes):.3f})")
    ax2.set_xlabel("Nombre de clusters k")
    ax2.set_ylabel("Score de silhouette (cosinus)")
    ax2.set_title("Score de silhouette\n(maximiser)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Choix de k pour le clustering K-means sur features DINOv2",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/clustering_choix_k.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(f"\nRecommandation : k = {k_opt} (silhouette maximale = {max(silhouettes):.3f})")
```

---

## 5. Connexions avec le pipeline global

DINOv2 intervient à deux endroits du pipeline HTR :

**Rôle 1 — Clustering préalable (Jour 3, section 3.1)**
Avant l'HTR, on regroupe les pages par style d'écriture. Chaque cluster reçoit un modèle HTR fine-tuné sur des données représentatives de ce style. Les erreurs du modèle HTR seront plus homogènes et plus facilement corrigibles en post-traitement.

**Rôle 2 — Aide à la segmentation de lignes (Jour 4)**
Les features de patches DINOv2 encodent une segmentation émergente texte/non-texte. Cette information peut être utilisée pour guider ou valider la segmentation de lignes de Kraken — notamment sur les pages à mise en page complexe où Kraken échoue.

**Rôle futur (module NLP)**
Les features DINOv2 peuvent servir de représentations visuelles dans des modèles vision-langage, pour conditionner la correction NLP sur le style visuel de la page — une piste de recherche active.

---

## Glossaire des termes avancés

**Centrage (*centering*)**
Dans DINO, soustraction d'un vecteur moyen $c$ des logits du teacher avant le softmax. Prévient l'effondrement vers une distribution ponctuelle dominée par une seule dimension. Mis à jour par EMA sur les batches.

**Collapse (effondrement de représentations)**
Pathologie d'entraînement où toutes les représentations convergent vers le même vecteur, rendant le modèle inutile. Solutions : exemples négatifs (contrastif), stop-gradient + EMA (BYOL, DINO), centering + sharpening (DINO).

**Cosine similarity (similarité cosinus)**
$\cos(\theta) = \frac{u \cdot v}{\|u\| \|v\|}$. Mesure l'angle entre deux vecteurs, indépendamment de leur norme. Préférée pour comparer des features SSL où la magnitude est moins informative que la direction.

**EMA (*Exponential Moving Average*)**
Mise à jour lissée d'un paramètre : $\theta \leftarrow m \cdot \theta + (1-m) \cdot \theta_{\text{new}}$ avec $m \in [0, 1)$. Produit une version temporellement lissée de $\theta_{\text{new}}$. Utilisée dans DINO pour le teacher : lent à évoluer, stable, moins susceptible de coller au bruit d'optimisation du student.

**Exposure bias**
Biais d'entraînement des modèles autoregressifs : pendant l'entraînement (teacher forcing), le modèle conditionne chaque token sur les tokens de référence corrects ; en inférence, il conditionne sur ses propres prédictions (potentiellement erronées). L'écart entre ces deux régimes dégrade les performances en inférence.

**False negatives (faux négatifs)**
Dans l'apprentissage contrastif, paires d'images de contenu similaire incorrectement traitées comme négatives (à éloigner). Pathologie classique des méthodes contrastives — atténuée par la self-distillation qui ne nécessite pas de négatifs.

**iBOT (*image BERT pretraining with Online Tokenizer*)**
Méthode SSL combinant distillation (DINO) et masquage de patches (BEiT). Le tokenizer est entraîné en ligne (*online*) — contrairement à BEiT qui utilise un tokenizer pré-entraîné figé (dVAE). Composante de la loss DINOv2.

**KoLeo (*Kolmogorov-Leonenko entropy regularization*)**
Terme de régularisation qui maximise l'entropie des représentations en encourageant leur étalement uniforme dans l'espace de features. Estimé à partir des distances aux plus proches voisins. Composante de la loss DINOv2.

**LVD-142M**
Dataset propriétaire de 142 millions d'images curé automatiquement pour l'entraînement de DINOv2. Construit par déduplication, filtrage de qualité et sélection par similarité avec des images de référence.

**Momentum encoder**
Encodeur dont les paramètres ne sont pas mis à jour par gradient, mais par EMA des paramètres d'un autre encodeur. Produit des représentations stables utilisées comme cibles ou clés dans l'apprentissage contrastif ou la self-distillation.

**Multi-crop**
Stratégie de DINO : générer plusieurs vues (crops) à différentes échelles d'une même image. Les vues globales (grand crop) sont traitées par le teacher ; toutes les vues (globales + locales) par le student. Force l'apprentissage de la cohérence local-global.

**NT-Xent loss (*Normalized Temperature-scaled Cross-Entropy*)**
Fonction de perte contrastive de SimCLR. Maximise la similarité entre paires positives (vues augmentées de la même image) et minimise celle avec toutes les autres paires dans le batch.

**Pretext task (tâche prétexte)**
Tâche synthétique générée automatiquement depuis les données brutes (sans annotation humaine), utilisée pour apprendre des représentations en SSL. Exemples : prédiction de rotation, puzzle jigsaw, complétion de couleur, masquage de patches.

**Score de silhouette**
Métrique d'évaluation d'un clustering. Pour chaque point, mesure la différence entre sa distance moyenne aux autres points de son cluster ($a$) et sa distance moyenne aux points du cluster voisin le plus proche ($b$) : $s = (b - a) / \max(a, b)$. Varie dans $[-1, 1]$ ; 1 = clustering parfait, 0 = chevauchement, -1 = mauvais clustering.

**Self-distillation**
Apprentissage où un modèle (student) apprend à reproduire les représentations d'une version de lui-même (teacher) mise à jour par EMA. Évite le besoin d'un enseignant externe et d'exemples négatifs. Risque : effondrement trivial, prévenu par centering/sharpening (DINO) ou stop-gradient + asymétrie (BYOL).

**Sharpening (affûtage)**
Dans DINO, utilisation d'une température basse $\tau_t$ pour le teacher, rendant sa distribution de sortie plus piquée (concentrée sur quelques dimensions). Prévient l'effondrement vers une distribution uniforme.

**Stop-gradient**
Opérateur qui bloque la propagation du gradient à travers un chemin du graphe de calcul. Dans DINO et BYOL, le teacher est protégé par stop-gradient : ses paramètres ne sont pas mis à jour par rétropropagation.

**UMAP (*Uniform Manifold Approximation and Projection*)**
Algorithme de réduction de dimension non linéaire. Modélise la structure locale et globale des données en construisant un graphe de voisinage dans l'espace de haute dimension, puis en optimisant sa projection en basse dimension (2D ou 3D). Souvent supérieur à t-SNE pour préserver la structure globale.

---

## Bibliographie de référence

### Articles fondateurs du SSL pour la vision

- **Gidaris, S., Singh, P., Komodakis, N.** (2018). *Unsupervised Representation Learning by Predicting Image Rotations*. ICLR 2018. [arXiv:1803.07728] — Tâche prétexte de prédiction de rotation.

- **Noroozi, M., Favaro, P.** (2016). *Unsupervised Visual Representation Learning by Solving Jigsaw Puzzles*. ECCV 2016. [arXiv:1603.09246] — Puzzle Jigsaw comme tâche prétexte.

- **Chen, T., Kornblith, S., Norouzi, M., Hinton, G.** (2020). *A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)*. ICML 2020. [arXiv:2002.05709] — Référence de l'apprentissage contrastif.

- **He, K., Fan, H., Wu, Y., Xie, S., Girshick, R.** (2020). *Momentum Contrast for Unsupervised Visual Representation Learning (MoCo)*. CVPR 2020. [arXiv:1911.05722] — Queue de négatifs et momentum encoder.

- **Grill, J.-B. et al.** (2020). *Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning (BYOL)*. NeurIPS 2020. [arXiv:2006.07733] — Self-distillation sans négatifs. Précurseur direct de DINO.

### DINO et DINOv2

- **Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., Joulin, A.** (2021). *Emerging Properties in Self-Supervised Vision Transformers (DINO)*. ICCV 2021. [arXiv:2104.14294] — **L'article fondateur. À lire intégralement, notamment la Figure 1 (segmentation émergente) et l'Annexe B (analyse des têtes d'attention).**

- **Oquab, M., Darcet, T., Moutakanni, T., Vo, H. V., Szafraniec, M., … Jégou, H.** (2023). *DINOv2: Learning Robust Visual Features without Supervision*. TMLR 2024. [arXiv:2304.07193] — DINOv2. Description complète de LVD-142M, iBOT, KoLeo.

- **Zhou, J., Wei, C., Wang, H., Shen, W., Xie, C., Yuille, A., Kong, T.** (2021). *iBOT: Image BERT Pre-Training with Online Tokenizer*. ICLR 2022. [arXiv:2111.07832] — Composante iBOT de DINOv2.

### Analyse théorique et propriétés du SSL

- **Tian, Y., Chen, X., Ganguli, S.** (2021). *Understanding Self-Supervised Learning Dynamics without Contrastive Pairs*. ICML 2021. [arXiv:2102.06810] — Analyse théorique de BYOL et self-distillation. Montre que le momentum teacher agit comme une régularisation implicite.

- **Bordes, F., Balestriero, R., Garrido, Q., Bardes, A., Vincent, P.** (2023). *Guillotine Regularization: Improving Deep Networks Generalization by Removing their Head*. TMLR 2023. [arXiv:2206.13378] — Montre que la qualité des features SSL (et notamment DINO) est localisée dans les couches profondes du backbone, pas dans la tête de projection.

### Réduction de dimension et clustering

- **McInnes, L., Healy, J., Melville, J.** (2018). *UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction*. [arXiv:1802.03426] — Algorithme UMAP. Article de référence.

- **Rousseeuw, P. J.** (1987). *Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis*. Journal of Computational and Applied Mathematics, 20, 53–65. — Score de silhouette.

### Applications aux documents et manuscrits

- **Seuret, M. et al.** (2021). *ICFHR 2020 Competition on Unsupervised and Supervised Manuscript Dating*. ICDAR 2021. — Utilisation de features SSL pour la datation automatique de manuscrits.

- **Christlein, V. et al.** (2019). *Unsupervised Feature Learning for Writer Identification and Writer Retrieval*. ICDAR 2019. — Clustering de mains de copistes par apprentissage non supervisé : l'application directement visée dans ce cours.

- **Cloppet, F. et al.** (2017). *ICDAR 2017 Competition on the Classification of Medieval Handwritings in Latin Script*. ICDAR 2017. — Référence pour l'évaluation de la classification de styles d'écriture médiévale.

---

*Support de cours rédigé pour le module Computer Vision · Promotion MD5 · Mai 2026*
*Ce document accompagne la séance 3.1 du Jour 3. Il suppose la lecture des sections 2.1 à 2.4.*
*Durée estimée de lecture : 75–90 minutes.*
