# Cours 3.2 — CLIP : apprentissage contrastif vision-langage

**Module 3 · Computer Vision appliquée aux manuscrits médiévaux · MD5**

---

> *« Toute image peut être décrite par des mots. Toute description peut évoquer une image. CLIP apprend cette correspondance à une échelle jusqu'alors inimaginable — et en tire une capacité de généralisation zero-shot qui surprend même ses concepteurs. »*

---

## Introduction : le fossé vision-langage

La vision par ordinateur classique est fondamentalement **fermée** : un modèle de classification connaît un ensemble fixe d'étiquettes, définies lors de la conception du dataset d'entraînement. ImageNet a 1 000 classes. Si un objet n'y figure pas — un *psautier enluminé*, une *charte de donation*, une *carte portulane* — aucun classifieur ImageNet ne peut le reconnaître ou le décrire.

Le traitement du langage naturel est **ouvert** : un modèle de langue peut traiter n'importe quelle phrase jamais formulée, pour peu qu'elle soit composée de tokens connus. Cette ouverture vient de la nature compositionnelle du langage — un nombre infini de significations construites depuis un vocabulaire fini.

**CLIP** (*Contrastive Language-Image Pre-Training*, Radford et al., 2021) franchit le fossé entre ces deux paradigmes en apprenant un **espace de représentation commun** aux images et aux textes. Dans cet espace, une image et sa description textuelle sont proches ; deux entités sémantiquement différentes sont éloignées — qu'elles soient image-image, texte-texte, ou image-texte.

Les conséquences sont profondes :

- **Classification zero-shot :** classer une image dans n'importe quelle catégorie, décrite en langage naturel, sans exemple d'entraînement dans cette catégorie.
- **Recherche cross-modale :** trouver des images correspondant à une description textuelle (ou des textes correspondant à une image).
- **Description d'images :** évaluer quelle description textuelle est la plus compatible avec une image donnée.

Pour notre projet, CLIP est l'outil de choix pour **décrire automatiquement les illustrations** des manuscrits — enluminures, dessins à la plume, cartes, diagrammes — sans aucun entraînement spécifique sur des données médiévales.

---

## 1. Architecture et entraînement

### 1.1 Vue d'ensemble

CLIP est composé de deux encodeurs indépendants :

- Un **encodeur d'image** $f : \mathcal{I} \to \mathbb{R}^d$ — une image produit un vecteur de dimension $d$.
- Un **encodeur de texte** $g : \mathcal{T} \to \mathbb{R}^d$ — un texte produit un vecteur de même dimension $d$.

Ces deux encodeurs sont entraînés conjointement pour que les représentations d'une image et de sa description correspondante soient **proches dans l'espace commun**, tandis que les représentations de paires non correspondantes sont **éloignées**.

```
Image I ──→ encodeur image f ──→ z_I ∈ ℝ^d
                                    ↘
                                      similarité cosinus
                                    ↗
Texte T ──→ encodeur texte g ──→ z_T ∈ ℝ^d
```

### 1.2 L'encodeur d'image

Radford et al. évaluent plusieurs architectures d'encodeur image :

- **ResNet** (ResNet-50, ResNet-101, RN50×4, RN50×16, RN50×64) avec attention pooling globale à la place de l'average pooling classique.
- **ViT** (ViT-B/32, ViT-B/16, ViT-L/14) — les configurations qui nous intéressent.

Le modèle le plus performant de l'article original est **CLIP ViT-L/14@336px** (ViT-L avec patches 14×14 et résolution d'entrée 336×336). C'est l'architecture qui obtient les meilleures performances zero-shot sur ImageNet (76,2% top-1).

La sortie de l'encodeur image (token CLS ou global pooling, selon l'architecture) est projetée vers $\mathbb{R}^d$ par une couche linéaire apprise.

### 1.3 L'encodeur de texte

Le texte est encodé par un **Transformer** (l'encodeur du Transformer original) avec des modifications légères :

- Vocabulaire BPE (*Byte Pair Encoding*) de 49 152 tokens.
- Longueur maximale de séquence : 77 tokens.
- Architecture : 12 couches, 512 dimensions, 8 têtes d'attention pour ViT-B/32 ; 16 couches, 768 dimensions, 12 têtes pour ViT-L/14.
- Masque d'attention causal (masque de type GPT) : chaque token ne voit que les tokens précédents. Ceci est un choix de commodité — l'encodeur texte de CLIP ressemble à un décodeur GPT.

La sortie est prise au niveau du token `[EOT]` (*End Of Text*) — la représentation du dernier token, qui a accès à toute la séquence via l'attention causale. Elle est projetée vers $\mathbb{R}^d$.

### 1.4 La normalisation L2 et la similarité cosinus

Avant le calcul de la loss et lors de l'inférence, les représentations $z_I$ et $z_T$ sont **normalisées en L2** :

$$\tilde{z}_I = \frac{z_I}{\|z_I\|_2}, \quad \tilde{z}_T = \frac{z_T}{\|z_T\|_2}$$

La similarité entre une image et un texte est alors leur **produit scalaire** (équivalent à la similarité cosinus sur des vecteurs normalisés) :

$$s(I, T) = \tilde{z}_I \cdot \tilde{z}_T = \cos(\angle(\tilde{z}_I, \tilde{z}_T)) \in [-1, 1]$$

Cette normalisation a deux avantages : elle supprime l'information de norme (magnitude), qui n'est pas informative pour la comparaison sémantique, et elle stabilise l'entraînement en bornant les scores.

### 1.5 La loss contrastive : InfoNCE symétrique

Pour un batch de $N$ paires $(I_i, T_i)_{i=1}^N$, on calcule la matrice de similarité $S \in \mathbb{R}^{N \times N}$ :

$$S_{ij} = \tilde{z}_{I_i} \cdot \tilde{z}_{T_j} \cdot \exp(\tau)$$

où $\tau$ est un paramètre de température **appris** (initialisé à $\log(1/0.07) \approx 2.66$, ce qui correspond à $\tau = 14.3$).

La loss est l'entropie croisée symétrique :

$$\mathcal{L} = \frac{1}{2}\left(\mathcal{L}_{\text{image→texte}} + \mathcal{L}_{\text{texte→image}}\right)$$

avec :

$$\mathcal{L}_{\text{image→texte}} = -\frac{1}{N} \sum_{i=1}^N \log \frac{\exp(S_{ii})}{\sum_{j=1}^N \exp(S_{ij})}$$

$$\mathcal{L}_{\text{texte→image}} = -\frac{1}{N} \sum_{j=1}^N \log \frac{\exp(S_{jj})}{\sum_{i=1}^N \exp(S_{ij})}$$

**Intuition géométrique :** pour chaque image $I_i$, la loss $\mathcal{L}_{\text{image→texte}}$ maximise la similarité avec son texte correct $T_i$ (la diagonale) et minimise la similarité avec les $N-1$ textes incorrects (hors diagonale). La symétrie de la loss fait la même chose depuis la perspective des textes.

```python
import torch
import torch.nn.functional as F

def clip_loss(
    logits_image_vers_texte: torch.Tensor,
    logits_texte_vers_image: torch.Tensor
) -> torch.Tensor:
    """
    Loss contrastive CLIP symétrique.

    Args:
        logits_image_vers_texte : (N, N) — similarités image[i] vs texte[j]
        logits_texte_vers_image : (N, N) — similarités texte[i] vs image[j]
                                  (généralement la transposée du premier)

    Returns:
        loss scalaire
    """
    N = logits_image_vers_texte.shape[0]
    # Les labels corrects sont sur la diagonale
    labels = torch.arange(N, device=logits_image_vers_texte.device)

    loss_img = F.cross_entropy(logits_image_vers_texte, labels)
    loss_txt = F.cross_entropy(logits_texte_vers_image, labels)

    return (loss_img + loss_txt) / 2


class CLIPModule(torch.nn.Module):
    """
    Calcul de la similarité CLIP et de la loss contrastive.
    Illustratif — utiliser les modèles HuggingFace en pratique.
    """
    def __init__(self, temperature_init: float = 0.07):
        super().__init__()
        # Température apprise (CLIP apprend log(1/τ) pour rester positif)
        self.log_temp = torch.nn.Parameter(
            torch.tensor(1.0 / temperature_init).log()
        )

    def forward(
        self,
        image_features: torch.Tensor,   # (N, d) — normalisé L2
        text_features: torch.Tensor     # (N, d) — normalisé L2
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Température (clampée pour éviter l'explosion)
        temp = self.log_temp.exp().clamp(max=100)

        # Matrice de similarité mise à l'échelle
        logits = temp * image_features @ text_features.T   # (N, N)

        loss = clip_loss(logits, logits.T)
        return loss, logits
```

**Le rôle de la température apprise**

La température $\tau$ contrôle la « netteté » de la distribution softmax :
- $\tau$ **petite** (température basse) → softmax piquée → distinctions franches entre les $N$ candidats. Le modèle doit être très confiant sur la paire correcte.
- $\tau$ **grande** (température haute) → softmax plate → loss moins discriminante.

CLIP apprend $\log(1/\tau)$ pour éviter les valeurs négatives, avec un clamp à 100 pour éviter la divergence. La valeur finale apprise converge typiquement vers $\tau \approx 50$–100.

### 1.6 Le dataset d'entraînement : WebImageText (WIT)

CLIP est entraîné sur **400 millions de paires (image, texte)** collectées depuis le web — appelées WebImageText (WIT), distinct du Wikipedia-based Image Text dataset de Google malgré l'acronyme similaire.

**Construction du dataset :**

1. **Collecte :** les auteurs identifient 500 000 requêtes de recherche à partir des termes les plus fréquents dans WikiText-103 (un corpus de textes Wikipedia en anglais) complétés par des termes liés à WordNet (ontologie lexicale).

2. **Filtrage :** pour chaque requête, jusqu'à 20 000 paires (image, texte) sont collectées depuis le web (texte = légende, texte alternatif, titre de page, ou texte avoisinant l'image). Le filtrage conserve uniquement les paires où le texte contient au moins l'un des termes de la requête.

3. **Équilibrage :** les requêtes rares sont sur-échantillonnées pour éviter la domination des requêtes fréquentes.

**Caractéristiques du dataset :**
- 400 millions de paires — environ 13× plus grand qu'ImageNet-21k.
- Textes en anglais exclusivement dans la version originale.
- Très biais vers le web occidental — surreprésentation de certains domaines (pop culture, produits commerciaux) et sous-représentation d'autres (documents anciens, iconographie médiévale).

**Implication pour les manuscrits :** CLIP connaît les « illuminated manuscripts » en tant que concept générique (ils apparaissent fréquemment sur le web), mais il ne connaît pas l'iconographie spécifique des cycles hagiographiques du XIIIe siècle ou la cartographie médiévale de style mappemonde. Ses descriptions d'enluminures rares seront génériques ou approximatives.

---

## 2. Classification zero-shot

### 2.1 Le principe

La classification zero-shot avec CLIP est remarquablement simple. Pour classer une image $I$ parmi $K$ catégories $\{c_1, \ldots, c_K\}$ :

1. Pour chaque catégorie $c_k$, construire une **description textuelle** : *« a photo of a {$c_k$} »* ou une formulation plus descriptive.
2. Encoder chaque description : $\tilde{z}_{T_k} = g(\text{description}_k) / \|g(\text{description}_k)\|$.
3. Encoder l'image : $\tilde{z}_I = f(I) / \|f(I)\|$.
4. Attribuer la catégorie dont la description est la plus similaire à l'image :

$$\hat{c} = \argmax_{k \in \{1,\ldots,K\}} \tilde{z}_I \cdot \tilde{z}_{T_k}$$

Les probabilités softmax donnent une distribution sur les classes :

$$P(c_k | I) = \frac{\exp(\tilde{z}_I \cdot \tilde{z}_{T_k} / \tau)}{\sum_{l=1}^K \exp(\tilde{z}_I \cdot \tilde{z}_{T_l} / \tau)}$$

**La beauté du zero-shot :** $K$ peut varier à l'inférence, les classes peuvent être modifiées sans réentraînement, et les descriptions peuvent être formulées en langage naturel riche — pas nécessairement des étiquettes courtes.

### 2.2 L'ingénierie de prompts (*prompt engineering*)

Le choix de la formulation des descriptions textuelles influence significativement les performances. Radford et al. rapportent une amélioration de 3,5 points de top-1 sur ImageNet en passant de la description minimale *« {class} »* à la formulation *« a photo of a {class} »*.

**Formulations génériques**
```python
# Du moins au plus performant pour ImageNet
templates = [
    "{class}",                              # Minimal — performances basses
    "a photo of a {class}",                 # Standard — bon équilibre
    "a high quality photo of a {class}",    # Qualité explicite
    "a photo of the {class}, a type of document",  # Contexte domaine
]
```

**L'ensembling de prompts**

Plutôt que de choisir un seul template, on peut moyenner les représentations textuelles sur plusieurs formulations — une forme d'ensembling dans l'espace de texte :

$$\tilde{z}_{T_k}^{\text{ensemble}} = \frac{1}{|P|} \sum_{p \in P} \tilde{z}_{T_k^{(p)}}$$

où $P$ est un ensemble de templates et $\tilde{z}_{T_k^{(p)}}$ la représentation normalisée du texte *« template $p$ pour classe $k$ »*.

Cet ensembling améliore les performances zero-shot de 3,5% supplémentaires sur ImageNet (Radford et al., 2021, Annexe D).

**Pour les manuscrits médiévaux — templates adaptés**

```python
TEMPLATES_MANUSCRITS = [
    "a medieval manuscript illustration showing {class}",
    "an illuminated manuscript depicting {class}",
    "a medieval drawing of {class}",
    "a pen and ink illustration of {class} from a medieval manuscript",
    "a medieval miniature representing {class}",
    "an enluminure showing {class}",
    "a detail from a medieval manuscript: {class}",
]

# Classes pour la description d'illustrations médiévales
CLASSES_ILLUSTRATIONS = [
    "a knight on horseback in battle",
    "a religious scene with saints and angels",
    "a king or ruler enthroned",
    "an astronomical or cosmological diagram",
    "a map or geographic illustration",
    "decorative foliage or floral motifs",
    "a decorated initial letter",
    "animals or hybrid creatures",
    "a scene of daily life or labor",
    "a crucifixion or religious martyrdom",
    "an architectural drawing or city view",
    "a heraldic shield or coat of arms",
]
```

### 2.3 Performances zero-shot : comparaisons et limites

| Dataset | CLIP (zero-shot) | Supervisé (fully) | Commentaire |
|---------|-----------------|-------------------|-------------|
| ImageNet | 76,2% top-1 | 88,4% (ViT-G) | -12 pts, mais sans entraînement |
| ImageNet-V2 | 70,1% | 80,7% | Robustesse à la distribution |
| Oxford Flowers | 77,0% | 97,9% | Faible : catégories très fines |
| UCF101 (actions) | 76,6% | 84,5% | Bon : descriptions naturelles |
| MNIST | 76,0% | 99,5% | Très faible : domaine hors-distribution |
| DTD (textures) | 44,0% | 79,0% | Très faible : concept non sémantique |

**Ce que ces résultats révèlent :**

- CLIP excelle sur les datasets où les classes correspondent à des concepts sémantiques décrivables en langage naturel (animaux, objets, actions, scènes).
- CLIP échoue relativement sur les tâches nécessitant une discrimination **fine-grained** (107 espèces de fleurs) ou des concepts **non sémantiques** (textures DTD, chiffres MNIST) — des concepts difficiles à décrire verbalement de façon discriminante.
- La performance zero-shot est hétérogène entre domaines : un concept qui apparaît rarement ou différemment sur le web aura une représentation CLIP de mauvaise qualité.

---

## 3. Au-delà du zero-shot : few-shot et fine-tuning

### 3.1 CLIP comme extracteur de features

Les features CLIP (les vecteurs $\tilde{z}_I$) peuvent être utilisées comme entrée d'un classifieur en aval, avec un nombre très limité d'exemples annotés.

**Sonde linéaire (*linear probe*)**

On entraîne une régression logistique (ou une couche fully connected) sur les features CLIP gelées. Les paramètres de l'encodeur image ne sont pas modifiés.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Extraire les features CLIP pour tous les exemples annotés
# (voir section 4 pour le code d'extraction)
features_train = ...  # (N_train, 512)
labels_train   = ...  # (N_train,) — étiquettes textuelles

# Encoder les labels
le = LabelEncoder()
y_train = le.fit_transform(labels_train)

# Entraîner une régression logistique sur les features CLIP
clf = LogisticRegression(
    max_iter=1000,
    C=0.316,     # Régularisation recommandée par Radford et al.
    random_state=42
)
clf.fit(features_train, y_train)

# Prédire sur de nouvelles images
features_test = ...   # (N_test, 512)
predictions = le.inverse_transform(clf.predict(features_test))
probas = clf.predict_proba(features_test)
```

**Résultats typiques sur notre domaine :** avec seulement 16 exemples annotés par classe de type d'illustration médiévale, une sonde linéaire sur des features CLIP-ViT-L/14 dépasse souvent un ResNet-50 entraîné de zéro sur plusieurs centaines d'exemples.

### 3.2 CoOp : optimisation du contexte textuel (*Context Optimization*)

**CoOp** (Zhou et al., 2022) propose une approche de fine-tuning léger qui n'optimise pas les paramètres des encodeurs, mais les **tokens de contexte** du prompt lui-même.

Au lieu d'un prompt fixe *« a photo of a {class} »*, on apprend $M$ tokens de contexte $\{v_1, \ldots, v_M\}$ en les traitant comme des paramètres continus :

$$\text{Prompt}(c_k) = [v_1, v_2, \ldots, v_M, \text{class}_k]$$

Ces $M \times d_{\text{text}}$ paramètres sont optimisés par descente de gradient sur un petit dataset annoté (16 exemples par classe suffisent souvent), pendant que tous les autres paramètres de CLIP sont gelés.

```python
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPTokenizer

class CoOpCLIP(nn.Module):
    """
    CLIP avec optimisation du contexte textuel (CoOp, Zhou et al., 2022).
    Seuls les tokens de contexte v_1...v_M sont entraînés.
    """
    def __init__(
        self,
        clip_model: CLIPModel,
        class_names: list[str],
        n_context: int = 16,    # Nombre de tokens de contexte
        context_init: str = "a photo of a"  # Initialisation textuelle
    ):
        super().__init__()
        self.clip = clip_model
        self.class_names = class_names
        n_classes = len(class_names)

        # Dimension des embeddings de tokens
        d = clip_model.config.text_config.hidden_size  # 512 pour CLIP-B/32

        # Initialiser le contexte depuis une phrase
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        init_ids = tokenizer(context_init, return_tensors="pt").input_ids[0]
        # On prend les tokens internes (sans [SOT] et [EOT])
        init_ids = init_ids[1:-1][:n_context]

        with torch.no_grad():
            # Récupérer les embeddings de tokens initiaux
            token_emb = clip_model.text_model.embeddings.token_embedding
            init_vectors = token_emb(init_ids)   # (n_init, d)

        # Si n_context > n_init, compléter avec des vecteurs aléatoires
        if init_vectors.shape[0] < n_context:
            padding = torch.randn(n_context - init_vectors.shape[0], d) * 0.02
            init_vectors = torch.cat([init_vectors, padding], dim=0)

        # Les tokens de contexte sont les seuls paramètres entraînés
        self.context_vectors = nn.Parameter(init_vectors[:n_context])  # (M, d)

        # Geler tous les autres paramètres de CLIP
        for param in clip_model.parameters():
            param.requires_grad_(False)

    def encode_text_with_context(
        self, device: torch.device
    ) -> torch.Tensor:
        """
        Construit les représentations textuelles avec les tokens de contexte appris.
        Retourne (n_classes, d) normalisé L2.
        """
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        token_emb = self.clip.text_model.embeddings.token_embedding

        all_text_features = []
        for class_name in self.class_names:
            # Tokens de la classe (sans [SOT] et [EOT] internes)
            class_ids = tokenizer(class_name, return_tensors="pt").input_ids[0]
            class_emb = token_emb(class_ids.to(device))   # (n_cls, d)

            # Construction : [SOT] + contexte + classe + [EOT]
            sot = class_emb[:1]    # Token [SOT]
            eot = class_emb[-1:]   # Token [EOT]
            ctx = self.context_vectors.to(device)
            cls_tokens = class_emb[1:-1]

            prompt_emb = torch.cat([sot, ctx, cls_tokens, eot], dim=0)
            # (1 + M + n_cls_tokens + 1, d)

            # Passer par l'encodeur texte (sans re-tokeniser — on passe les embeddings)
            # Note : nécessite d'accéder à la couche d'encodage directement
            # Simplifié ici pour la clarté pédagogique
            all_text_features.append(prompt_emb.mean(dim=0))  # Approximation

        text_features = torch.stack(all_text_features)
        return text_features / text_features.norm(dim=-1, keepdim=True)
```

**Résultats de CoOp :** avec 16 shots par classe sur 11 datasets de vision, CoOp améliore CLIP zero-shot de 4,6% en moyenne, et surpasse la sonde linéaire de 1,7% — avec un coût d'optimisation bien inférieur au fine-tuning complet.

---

## 4. Implémentation pratique avec HuggingFace

### 4.1 Chargement et inférence de base

```python
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Deux modèles courants :
# "openai/clip-vit-base-patch32"  — rapide, 151 M params, d=512
# "openai/clip-vit-large-patch14" — plus lent, 428 M params, d=768
MODEL_NAME = "openai/clip-vit-large-patch14"

print(f"Chargement de {MODEL_NAME}...")
clip_processor = CLIPProcessor.from_pretrained(MODEL_NAME)
clip_model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()

D_CLIP = clip_model.config.projection_dim   # 768 pour ViT-L/14
print(f"Dimension des features CLIP : {D_CLIP}")


def encoder_image_clip(
    images: list[Image.Image]
) -> torch.Tensor:
    """
    Encode une liste d'images PIL en features CLIP normalisées.
    Retourne (N, D_CLIP) sur CPU.
    """
    inputs = clip_processor(
        images=images,
        return_tensors="pt",
        padding=True
    ).to(DEVICE)

    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
        # Normalisation L2
        features = features / features.norm(dim=-1, keepdim=True)

    return features.cpu()


def encoder_texte_clip(
    textes: list[str]
) -> torch.Tensor:
    """
    Encode une liste de textes en features CLIP normalisées.
    Retourne (N, D_CLIP) sur CPU.
    """
    inputs = clip_processor(
        text=textes,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77   # Longueur maximale CLIP
    ).to(DEVICE)

    with torch.no_grad():
        features = clip_model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)

    return features.cpu()


def calculer_similarites(
    image_features: torch.Tensor,   # (N_img, D)
    text_features: torch.Tensor,    # (N_txt, D)
    temperature: float = 100.0      # Valeur typique après entraînement
) -> torch.Tensor:
    """
    Calcule la matrice de similarité (N_img, N_txt) mise à l'échelle.
    """
    return temperature * image_features @ text_features.T
```

### 4.2 Classification zero-shot d'illustrations médiévales

```python
def classifier_illustration_zero_shot(
    image: Image.Image,
    classes: list[str],
    templates: list[str] = None,
    top_k: int = 3,
    verbose: bool = True
) -> list[tuple[str, float]]:
    """
    Classifie une illustration médiévale parmi un ensemble de classes
    par zero-shot CLIP avec ensembling de prompts.

    Args:
        image    : image PIL de l'illustration (découpée depuis le scan)
        classes  : liste de descriptions de classes candidates
        templates : templates de prompts (None = templates médiévaux par défaut)
        top_k    : nombre de résultats à retourner

    Returns:
        Liste de (classe, probabilité) triée par probabilité décroissante
    """
    if templates is None:
        templates = TEMPLATES_MANUSCRITS   # Définis en section 2.2

    # Encoder l'image
    img_feat = encoder_image_clip([image])   # (1, D)

    # Encoder les textes avec ensembling de templates
    text_features_ensemble = []
    for classe in classes:
        features_templates = []
        for template in templates:
            prompt = template.replace("{class}", classe)
            feat = encoder_texte_clip([prompt])   # (1, D)
            features_templates.append(feat)

        # Moyenner les features sur les templates puis renormaliser
        feat_moy = torch.stack(features_templates).mean(dim=0)  # (1, D)
        feat_moy = feat_moy / feat_moy.norm(dim=-1, keepdim=True)
        text_features_ensemble.append(feat_moy)

    text_feats = torch.cat(text_features_ensemble, dim=0)  # (N_classes, D)

    # Calcul des similarités et softmax
    logits = 100.0 * img_feat @ text_feats.T   # (1, N_classes)
    probas = torch.softmax(logits, dim=-1)[0]   # (N_classes,)

    # Top-k résultats
    top_indices = probas.topk(top_k).indices.tolist()
    resultats = [(classes[i], probas[i].item()) for i in top_indices]

    if verbose:
        print("\nClassification zero-shot CLIP :")
        for classe, prob in resultats:
            barre = "█" * int(prob * 30)
            print(f"  {prob:.1%} {barre} {classe}")

    return resultats
```

### 4.3 Description automatique des illustrations : pipeline complet

```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def decrire_illustrations_page(
    scan_path: str,
    masques_illustrations: list[dict],   # Sortie de SAM pour les régions image
    nom_figure: str = "descriptions"
) -> list[dict]:
    """
    Décrit automatiquement les illustrations détectées dans un scan.

    Args:
        scan_path             : chemin du scan de page
        masques_illustrations : liste de dicts SAM avec 'bbox' et 'segmentation'

    Returns:
        Liste de dicts {region, classe_top1, probas, description_generee}
    """
    page = Image.open(scan_path).convert("RGB")
    page_np = np.array(page)

    resultats = []
    n_illustrations = len(masques_illustrations)

    if n_illustrations == 0:
        print("  Aucune illustration détectée dans la page.")
        return []

    fig, axes = plt.subplots(
        2, max(n_illustrations, 1),
        figsize=(6 * max(n_illustrations, 1), 10)
    )
    if n_illustrations == 1:
        axes = axes[:, np.newaxis]

    # Ligne 1 : page entière avec les régions surlignées
    for col in range(n_illustrations):
        axes[0, col].imshow(page)
        bbox = masques_illustrations[col]["bbox"]  # [x, y, w, h]
        rect = patches.Rectangle(
            (bbox[0], bbox[1]), bbox[2], bbox[3],
            linewidth=2, edgecolor="red", facecolor="none"
        )
        axes[0, col].add_patch(rect)
        axes[0, col].set_title(f"Région {col+1}", fontsize=10)
        axes[0, col].axis("off")

    # Ligne 2 : région découpée + description
    for col, masque in enumerate(masques_illustrations):
        bbox = masque["bbox"]
        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        # Découpe de la région
        region = page.crop((x, y, x + w, y + h))

        # Classification zero-shot
        top_classes = classifier_illustration_zero_shot(
            region,
            CLASSES_ILLUSTRATIONS,
            top_k=3,
            verbose=False
        )

        # Affichage
        axes[1, col].imshow(region)
        titre = "\n".join([f"{p:.0%} {c[:35]}" for c, p in top_classes])
        axes[1, col].set_title(titre, fontsize=7, pad=5)
        axes[1, col].axis("off")

        # Stockage des résultats
        resultats.append({
            "region_id": col,
            "bbox": bbox,
            "classe_top1": top_classes[0][0],
            "confiance_top1": top_classes[0][1],
            "top3": top_classes,
            "description_generee": (
                f"Illustration médiévale : {top_classes[0][0]} "
                f"(confiance : {top_classes[0][1]:.0%})"
            )
        })

    plt.suptitle(
        f"Descriptions automatiques des illustrations — {Path(scan_path).stem}",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(f"figures/{nom_figure}_clip_descriptions.png", dpi=150, bbox_inches="tight")
    plt.show()

    return resultats
```

### 4.4 Recherche cross-modale : trouver des images par description textuelle

Une fonctionnalité particulièrement utile pour les chercheurs en humanités : rechercher dans un corpus de manuscrits les pages ou régions correspondant à une description textuelle.

```python
def recherche_cross_modale(
    requete_textuelle: str,
    features_corpus: np.ndarray,   # (N_pages, D) — features CLIP pré-calculées
    chemins_corpus: list[str],
    top_k: int = 5,
    nom_figure: str = "recherche"
) -> list[tuple[str, float]]:
    """
    Trouve les pages du corpus les plus similaires à une requête textuelle.

    Exemple d'utilisation :
        resultats = recherche_cross_modale(
            "a knight fighting a dragon in a medieval battle scene",
            features_pages, chemins_pages, top_k=5
        )
    """
    # Encoder la requête textuelle
    query_feat = encoder_texte_clip([requete_textuelle])   # (1, D)

    # Similarité cosinus avec toutes les pages du corpus
    # (les features sont déjà normalisées L2)
    features_tensor = torch.tensor(features_corpus)        # (N, D)
    similarites = (features_tensor @ query_feat.T).squeeze()  # (N,)

    # Top-k résultats
    top_indices = similarites.topk(top_k).indices.tolist()
    resultats = [
        (chemins_corpus[i], similarites[i].item())
        for i in top_indices
    ]

    # Visualisation
    fig, axes = plt.subplots(1, top_k, figsize=(5 * top_k, 6))
    fig.suptitle(
        f"Recherche cross-modale CLIP\nRequête : « {requete_textuelle} »",
        fontsize=12, fontweight="bold"
    )

    for col, (chemin, sim) in enumerate(resultats):
        img = Image.open(chemin).convert("RGB")
        img_thumb = img.resize((300, 400), Image.LANCZOS)
        axes[col].imshow(img_thumb)
        axes[col].set_title(
            f"Rang {col+1}\nSimilarité : {sim:.3f}\n{Path(chemin).stem}",
            fontsize=8
        )
        axes[col].axis("off")

    plt.tight_layout()
    plt.savefig(f"figures/{nom_figure}_recherche_clip.png", dpi=150, bbox_inches="tight")
    plt.show()

    return resultats
```

---

## 5. Propriétés, limites et variantes

### 5.1 Ce que CLIP fait bien

**Généralisation zero-shot robuste**
CLIP transfère bien entre domaines pour les concepts sémantiques décrits dans son corpus d'entraînement. Il ne nécessite aucune annotation pour les nouvelles tâches — seulement des descriptions textuelles des classes.

**Robustesse à la distribution shift**
Sur ImageNet-V2 (même distribution qu'ImageNet mais nouvelles images), la plupart des modèles supervisés chutent de 11–14 points. CLIP ne chute que de 6 points — une robustesse accrue attribuée à la diversité du corpus web.

**Représentations sémantiquement riches**
Les features CLIP capturent des concepts sémantiques de haut niveau plus que des textures ou des formes locales. Pour le clustering d'illustrations médiévales, elles capturent le sujet iconographique (ce que représente l'image) mieux que des features DINOv2 ou ResNet.

### 5.2 Limites structurelles de CLIP

**Raisonnement spatial**
CLIP peut identifier qu'une image contient « un cheval et un chevalier », mais distinguer « un chevalier *sur* un cheval » de « un chevalier *à côté d* un cheval » est beaucoup plus difficile. Les relations spatiales et les compositions précises sont mal encodées.

**Comptage**
CLIP peine à distinguer « trois enluminures » de « cinq enluminures » dans une page — le comptage est un concept non sémantique difficile à décrire textuellement de façon discriminante.

**Attributs fins (*fine-grained attributes*)**
Pour l'iconographie médiévale, CLIP peut identifier « une scène de crucifixion » mais aura du mal à distinguer « une crucifixion avec Marie et Jean » de « une crucifixion avec les deux larrons ». Ces distinctions iconographiques fines nécessitent soit un corpus d'entraînement spécialisé, soit un modèle de vision-langage plus puissant (LLaVA, GPT-4V).

**Biais du corpus web**
Le corpus WIT est massivement anglophone et biaisé vers la culture occidentale contemporaine. Les concepts rares dans ce corpus (iconographie byzantine, manuscrits en alphabet hébraïque, calligraphie arabe médiévale) auront des représentations CLIP de mauvaise qualité.

**Absence de compréhension du texte dans les images**
CLIP ne « lit » pas le texte qu'il voit dans une image — il est entraîné sur des descriptions *de* l'image, pas sur des transcriptions *du* texte visible. Il ne peut pas distinguer une page de texte latin d'une page de texte grec à partir de la reconnaissance des caractères.

### 5.3 Évaluation : la matrice de confusion inter-classes

Pour évaluer CLIP sur notre tâche spécifique, on construit une matrice de confusion sur un petit dataset annoté d'illustrations :

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def evaluer_clip_sur_dataset(
    images_annotees: list[tuple[Image.Image, str]],
    classes: list[str],
    templates: list[str] = None
) -> dict:
    """
    Évalue CLIP zero-shot sur un dataset annoté d'illustrations.

    Args:
        images_annotees : liste de (image_PIL, classe_vraie)
        classes         : liste des classes possibles

    Returns:
        Dictionnaire avec précision globale et matrice de confusion
    """
    y_true, y_pred = [], []

    for image, vraie_classe in images_annotees:
        top = classifier_illustration_zero_shot(
            image, classes, templates, top_k=1, verbose=False
        )
        classe_predite = top[0][0]
        y_true.append(vraie_classe)
        y_pred.append(classe_predite)

    # Précision globale
    precision = sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)
    print(f"Précision zero-shot CLIP : {precision:.1%}")

    # Matrice de confusion
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(cm, display_labels=[c[:20] for c in classes])
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title("Matrice de confusion — CLIP zero-shot", fontsize=12)
    plt.tight_layout()
    plt.savefig("figures/clip_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()

    return {"precision": precision, "confusion_matrix": cm, "classes": classes}
```

### 5.4 Variantes et successeurs

**OpenCLIP**
Reproduction open source de CLIP par LAION, entraîné sur des datasets publics (LAION-400M, LAION-2B, DataComp). Performances comparables à CLIP original, avec des modèles plus grands (ViT-G/14, ViT-bigG/14). Recommandé pour les usages académiques qui nécessitent la reproductibilité et l'auditabilité des données d'entraînement.

```bash
pip install open_clip_torch
```

```python
import open_clip

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-L-14",
    pretrained="laion2b_s32b_b82k"   # Pré-entraîné sur LAION-2B
)
tokenizer = open_clip.get_tokenizer("ViT-L-14")
```

**SigLIP (Sigmoid Loss for Language-Image Pre-Training)**
Zhai et al. (2023) remplacent la loss InfoNCE symétrique de CLIP par une **loss sigmoïde** opérant sur chaque paire indépendamment :

$$\mathcal{L}_{\text{SigLIP}} = -\frac{1}{N^2} \sum_{i,j} \left[ y_{ij} \log \sigma(z_{ij}) + (1 - y_{ij}) \log(1 - \sigma(z_{ij})) \right]$$

où $y_{ij} = 1$ si $(i, j)$ est une paire positive, $\sigma(z_{ij}) = 1/(1 + e^{-z_{ij}})$.

Avantages : pas besoin d'un grand batch pour les négatifs (chaque paire est indépendante) ; meilleure scalabilité. SigLIP-So400M atteint 83,2% zero-shot sur ImageNet.

**ALIGN**
Même principe que CLIP mais entraîné par Google sur 1,8 milliard de paires (non publié). Montre que la mise à l'échelle naïve du dataset améliore les performances, même avec une qualité d'annotation inférieure.

**Florence-2 (Microsoft)**
Modèle unificateur qui étend CLIP en ajoutant des capacités de détection, segmentation et description fine. Entraîné sur le dataset FLD-5B (5,4 milliards d'annotations multimodales). Pertinent pour les tâches nécessitant une localisation fine dans les illustrations.

**LLaVA et les vision-language models génératifs**
LLaVA (*Large Language and Vision Assistant*) connecte un encodeur CLIP à un LLM (Llama, Vicuna) pour générer des descriptions libres d'images en langage naturel. C'est un successeur naturel de CLIP pour les tâches de description détaillée — nous y reviendrons dans la section sur les pipelines (Jour 4).

---

## 6. Connexions avec le pipeline global

CLIP intervient à une étape spécifique et délimitée de notre pipeline :

**Étape 3b — Description des illustrations**

Après la segmentation de layout par SAM (qui identifie les régions image), CLIP décrit chaque région en zero-shot. Cette description est stockée dans le JSON de sortie et transmise au module NLP pour être intégrée à la transcription structurée de la page.

**Ce que CLIP apporte que l'HTR ne peut pas faire :**
Les illustrations médiévales ne sont pas du texte — l'HTR n'a aucune prise sur elles. CLIP les transforme en descriptions textuelles qui peuvent être indexées, recherchées, et analysées par les mêmes outils que le texte transcrit.

**Ce que CLIP ne remplace pas :**
Une description iconographique de qualité académique (par exemple, identifier précisément le saint représenté dans une scène hagiographique, ou reconnaître un cycle narratif spécifique) nécessite une expertise humaine ou un modèle spécialisé sur l'iconographie médiévale. CLIP produit des descriptions génériques utiles pour l'indexation, pas des descriptions de niveau édition critique.

---

## Glossaire des termes avancés

**BPE (*Byte Pair Encoding*)**
Algorithme de tokenisation sous-lexicale : les paires de caractères les plus fréquentes sont fusionnées itérativement pour construire un vocabulaire de sous-mots. Utilisé par CLIP pour tokeniser les textes. Permet de représenter des mots inconnus comme séquences de sous-mots connus.

**CoOp (*Context Optimization*)**
Méthode de fine-tuning léger pour CLIP : au lieu d'optimiser les paramètres de l'encodeur, on optimise les tokens de contexte du prompt textuel. Permet d'adapter CLIP à un domaine spécifique avec très peu d'exemples annotés.

**Cross-modal retrieval (recherche inter-modale)**
Tâche de recherche d'information : trouver des éléments d'une modalité (ex : images) correspondant à une requête dans une autre modalité (ex : texte). CLIP résout ce problème en projetant les deux modalités dans un espace commun.

**Embedding space (espace d'embedding)**
Espace vectoriel de représentation dans lequel les éléments de différentes modalités (images, textes) sont projetés. Dans CLIP, l'espace commun image-texte de dimension $d$ permet de mesurer la compatibilité sémantique par similarité cosinus.

**Fine-grained recognition (reconnaissance fine)**
Tâche de classification où les catégories sont très proches visuellement — par exemple, distinguer 200 espèces d'oiseaux ou 107 variétés de fleurs. CLIP est moins performant en zero-shot sur ces tâches que sur la classification sémantique grossière.

**InfoNCE loss (*Noise-Contrastive Estimation*)**
Famille de fonctions de perte contrastives basées sur l'estimation du rapport de densité. La NT-Xent de SimCLR et la loss contrastive de CLIP en sont des instances. Maximise la probabilité des paires positives relativement aux paires négatives dans le batch.

**Joint embedding space**
Voir *Embedding space*. Qualifie spécifiquement le cas où deux modalités différentes (image et texte) partagent le même espace de représentation, permettant des comparaisons directes.

**Linear probe (sonde linéaire)**
Évaluation de la qualité des features d'un encodeur en entraînant uniquement un classifieur linéaire (régression logistique ou couche fully connected) sur les features gelées. Plus performant = features plus informatives. Standard pour évaluer les modèles SSL et pré-entraînés.

**OpenCLIP**
Reproduction open source de CLIP par LAION et collaborateurs, entraîné sur des datasets publics (LAION-400M, LAION-2B). Permet l'audit et la reproductibilité contrairement au CLIP original d'OpenAI, dont les données d'entraînement ne sont pas publiées.

**Prompt engineering**
Conception des formulations textuelles soumises à un modèle vision-langage pour maximiser les performances sur une tâche. Inclut le choix du template (« a photo of a {class} »), le niveau de détail, et l'ensembling de plusieurs formulations.

**Prompt ensembling**
Moyennage des représentations textuelles sur plusieurs formulations (templates) différentes d'une même classe, avant la comparaison avec la représentation image. Améliore les performances zero-shot de 3–5% sur les benchmarks standards.

**SigLIP (*Sigmoid Loss for Language-Image Pre-Training*)**
Variante de CLIP utilisant une loss sigmoïde (cross-entropie binaire) par paire, au lieu de la loss softmax sur le batch complet. Plus scalable (pas besoin de grands batchs), performances comparables ou supérieures.

**Vision-language model (VLM)**
Modèle traitant conjointement des entrées visuelles et textuelles. CLIP est un VLM discriminatif (compare image et texte). LLaVA, GPT-4V, Gemini sont des VLMs génératifs (génèrent du texte à partir d'une image et d'une instruction).

**WebImageText (WIT)**
Dataset propriétaire d'OpenAI composé de 400 millions de paires (image, texte) collectées depuis le web. Utilisé pour pré-entraîner CLIP. Non publié — ce qui a motivé la création de LAION-400M et LAION-2B comme alternatives ouvertes.

**Zero-shot classification**
Classification d'une image dans des catégories non vues à l'entraînement, en s'appuyant sur des descriptions textuelles de ces catégories. Possible grâce à l'espace joint image-texte de CLIP.

---

## Bibliographie de référence

### Article fondateur et analyses

- **Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., … Sutskever, I.** (2021). *Learning Transferable Visual Models From Natural Language Supervision*. ICML 2021. [arXiv:2103.00020] — **L'article CLIP original. À lire intégralement, notamment les Annexes sur le prompt engineering (D), les datasets (E) et les biais (F).**

- **Radford, A. et al.** (2021). *CLIP Model Card*. OpenAI. — Document de model card décrivant les biais et limitations connues de CLIP. Essentiel pour un usage responsable.

- **Wortsman, M. et al.** (2022). *Robust Fine-Tuning of Zero-Shot Models*. CVPR 2022. [arXiv:2109.01903] — Sur comment fine-tuner CLIP sans perdre sa robustesse zero-shot (WiSE-FT).

### Variantes et améliorations

- **Schuhmann, C. et al.** (2022). *LAION-5B: An Open Large-Scale Dataset for Training Next Generation Image-Text Models*. NeurIPS 2022. [arXiv:2210.08402] — Le dataset ouvert comparable à WIT. Base d'OpenCLIP.

- **Ilharco, G. et al.** (2021). *OpenCLIP*. Zenodo. [doi:10.5281/zenodo.5143773] — La bibliothèque OpenCLIP. Référence logicielle.

- **Zhai, X. et al.** (2023). *Sigmoid Loss for Language Image Pre-Training (SigLIP)*. ICCV 2023. [arXiv:2303.15343] — SigLIP : loss sigmoïde, meilleure scalabilité.

- **Yuan, L. et al.** (2021). *Florence: A New Foundation Model for Computer Vision*. [arXiv:2111.11432] — Florence (Microsoft) : extension de CLIP avec détection et segmentation.

### Few-shot et fine-tuning

- **Zhou, K., Yang, J., Loy, C. C., Liu, Z.** (2022). *Learning to Prompt for Vision-Language Models (CoOp)*. International Journal of Computer Vision (IJCV). [arXiv:2109.01134]

- **Zhou, K. et al.** (2022). *Conditional Prompt Learning for Vision-Language Models (CoCoOp)*. CVPR 2022. [arXiv:2203.05557] — Extension de CoOp avec conditionnement sur l'image.

- **Zhang, R., Zhang, W., Fang, R., Gao, P., Li, K., Dai, J., … Li, H.** (2022). *Tip-Adapter: Training-free CLIP-Adapter for Better Vision-Language Modeling*. ECCV 2022. [arXiv:2207.09519] — Alternative à CoOp sans optimisation.

### Vision-Language models génératifs

- **Liu, H., Li, C., Wu, Q., Lee, Y. J.** (2023). *Visual Instruction Tuning (LLaVA)*. NeurIPS 2023. [arXiv:2304.08485] — LLaVA : connecter CLIP à un LLM pour la description générative.

- **Alayrac, J.-B. et al.** (2022). *Flamingo: a Visual Language Model for Few-Shot Learning*. NeurIPS 2022. [arXiv:2204.14198] — Flamingo (DeepMind) : vision-language en few-shot par cross-attention.

### Applications aux documents et humanités numériques

- **Shen, Z. et al.** (2021). *LayoutParser: A Unified Toolkit for Deep Learning Based Document Image Analysis*. ICDAR 2021. [arXiv:2103.15348] — Intègre CLIP pour la classification de régions dans les documents.

- **Stiennon, N. et al.** (2020). *Learning to summarize from human feedback*. NeurIPS 2020. — Pertinent pour comprendre comment les données d'entraînement web biaisent les modèles (y compris CLIP) vers certaines représentations culturelles.

- **Bender, E. M., Gebru, T., McMillan-Major, A., Shmitchell, S.** (2021). *On the Dangers of Stochastic Parrots: Can Language Models Be Too Big?* FAccT 2021. — Sur les biais dans les grandes données web. Applicable à CLIP et à ses implications pour l'iconographie médiévale.

---

*Support de cours rédigé pour le module Computer Vision · Promotion MD5 · Mai 2026*
*Ce document accompagne la séance 3.2 du Jour 3. Il suppose la lecture des sections 3.1 (DINO) et 2.2 (ViT).*
*Durée estimée de lecture : 75 minutes.*
