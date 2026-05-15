# Cours 2.1 — Rappel CNN et limites structurelles pour les documents

**Module 2 · Computer Vision appliquée aux manuscrits médiévaux · MD5**

---

> *« Les biais inductifs d'une architecture ne sont pas des défauts — ce sont des hypothèses sur la structure du monde. Quand ces hypothèses sont vraies, le modèle apprend vite et bien. Quand elles sont fausses, il échoue systématiquement. »*

---

## Introduction : pourquoi revenir sur les CNN ?

Vous connaissez les réseaux de neurones convolutifs. Vous avez probablement entraîné des modèles de classification, de détection ou de segmentation lors de projets précédents. L'objectif de cette section n'est pas de répéter ce que vous savez — c'est de **revisiter les CNN depuis leur biais inductif**, c'est-à-dire depuis les hypothèses structurelles qu'ils encodent sur la nature des images.

Cette lecture est indispensable pour comprendre pourquoi les Vision Transformers ont progressivement supplanté les CNN dans de nombreuses tâches de vision, et en particulier dans les tâches de reconnaissance de documents — notre domaine. Les CNN ne sont pas « mauvais » : ils sont très bien adaptés à un certain type de problèmes. Les manuscrits médiévaux ne font pas partie de ces problèmes, et comprendre pourquoi est la première étape pour choisir la bonne architecture.

---

## 1. La convolution 2D : formalisme et intuitions

### 1.1 Définition formelle

Soit $I \in \mathbb{R}^{H \times W}$ une image en niveaux de gris (hauteur $H$, largeur $W$) et $K \in \mathbb{R}^{k \times k}$ un filtre (ou noyau) de taille $k \times k$. La convolution discrète 2D produit une **carte de caractéristiques** (*feature map*) $F \in \mathbb{R}^{H' \times W'}$ définie par :

$$F[i, j] = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} I[i \cdot s + m,\ j \cdot s + n] \cdot K[m, n] + b$$

où $s$ est le **stride** (pas de déplacement du filtre) et $b$ un biais scalaire.

Les dimensions de la sortie en fonction du padding $p$ sont :

$$H' = \left\lfloor \frac{H - k + 2p}{s} \right\rfloor + 1, \quad W' = \left\lfloor \frac{W - k + 2p}{s} \right\rfloor + 1$$

**Nota bene sur la terminologie :** en deep learning, on parle de « convolution » mais l'opération implémentée est techniquement une **corrélation croisée** (*cross-correlation*) — le filtre n'est pas retourné. La distinction est sans importance pratique puisque les filtres sont appris.

### 1.2 Les paramètres opérationnels

**Le filtre (*kernel*)**
Dans un CNN, les filtres ne sont pas définis à la main — ils sont **appris par rétropropagation**. Un filtre $3 \times 3$ contient 9 paramètres apprenables (plus le biais). Une couche convolutive avec $C_{out}$ filtres sur une entrée à $C_{in}$ canaux contient $C_{out} \times C_{in} \times k^2 + C_{out}$ paramètres.

**Le stride $s$**
Le stride contrôle le sous-échantillonnage spatial. $s = 1$ : pas de réduction de résolution. $s = 2$ : résolution divisée par 2. Augmenter le stride est une alternative à l'average pooling ou au max pooling pour réduire la résolution.

**Le padding $p$**
- *Valid padding* ($p = 0$) : pas de zéros ajoutés, la sortie est plus petite que l'entrée.
- *Same padding* ($p = \lfloor k/2 \rfloor$) : la sortie a la même résolution que l'entrée (pour $s = 1$). Convention quasi-universelle dans les architectures modernes.
- *Causal padding* : uniquement pour les convolutions 1D temporelles (non pertinent ici).

**La dilatation (*atrous convolution*)**
La convolution dilatée insère des zéros entre les éléments du filtre, permettant d'augmenter le **champ réceptif** sans augmenter le nombre de paramètres ni réduire la résolution :

$$F[i, j] = \sum_{m,n} I[i + d \cdot m,\ j + d \cdot n] \cdot K[m, n]$$

où $d$ est le taux de dilatation. Utilisé dans DeepLab, WaveNet, et les modèles de segmentation sémantique.

### 1.3 Le champ réceptif (*receptive field*)

Le **champ réceptif** d'un neurone dans une couche $l$ est l'ensemble des pixels de l'image d'entrée qui influencent la valeur de ce neurone. C'est une notion centrale pour comprendre les limites des CNN.

Pour une séquence de convolutions $3 \times 3$ avec stride 1 et sans pooling, le champ réceptif croît **linéairement** avec la profondeur :

$$\text{RF}(l) = 1 + l \times (k - 1) = 1 + 2l \quad \text{(pour } k=3\text{)}$$

Avec du pooling par facteur 2 après chaque bloc, il croît exponentiellement :

$$\text{RF}_{\text{eff}}(l) \approx k^l$$

Mais ce calcul donne le **champ réceptif théorique**, pas le champ réceptif **effectif**. Des travaux de Luo et al. (2016) ont montré empiriquement que les gradients se distribuent selon une gaussienne dans le champ réceptif théorique : seul le centre contribue significativement. Le champ réceptif effectif est environ $\sqrt{\text{RF}_{\text{théorique}}}$ — soit un carré de côté 11 pour un champ théorique de 121.

> **Implication directe pour les manuscrits :** une page de manuscrit à 400 DPI fait environ 4 000 × 5 500 pixels. Pour qu'un neurone en haut du réseau « voie » la page entière, il faudrait un champ réceptif de 4 000 pixels. Avec des blocs $3 \times 3$ et du pooling ×2, cela nécessite $\log_2(4000) \approx 12$ étapes de pooling — soit une réduction de résolution par 4096. La représentation finale ne contient plus que $\approx 1$ pixel par page. L'information spatiale fine est irrémédiablement perdue.

### 1.4 Pooling et invariance

Le **max pooling** sélectionne la valeur maximale dans une fenêtre locale ; l'**average pooling** en calcule la moyenne. Les deux induisent :

- Une **réduction de résolution** (sous-échantillonnage).
- Une **invariance locale à la translation** : décaler légèrement une feature dans la fenêtre de pooling ne change pas la sortie du max pooling.

Cette invariance à la translation est l'une des forces des CNN sur les images naturelles. Pour un classificateur d'objets (chat, chien, voiture…), qu'un chat soit à gauche ou à droite de l'image importe peu. Pour un HTR, la position est au contraire porteuse d'information : la deuxième lettre d'un mot n'est pas interchangeable avec la cinquième. L'invariance devient un obstacle.

### 1.5 Normalisation et stabilisation de l'entraînement

**Batch Normalization (BatchNorm, Ioffe & Szegedy, 2015)**

Pour chaque canal $c$, BatchNorm normalise les activations sur le mini-batch courant :

$$\hat{x}_c = \frac{x_c - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}}$$

puis applique une transformation affine apprise : $y_c = \gamma_c \hat{x}_c + \beta_c$.

Effets : stabilise les gradients, accélère la convergence, permet des taux d'apprentissage plus élevés, et agit comme une régularisation implicite (réduisant le besoin de dropout).

Limite : se comporte mal pour les petits batch sizes ($N < 8$) et n'est pas applicable séquence par séquence (contrairement à Layer Normalization, préférée dans les Transformers).

**Layer Normalization**
Normalise sur les dimensions de caractéristiques (plutôt que sur le batch). Indépendante de la taille du batch — c'est pourquoi elle s'impose dans les Transformers, où les séquences ont des longueurs variables.

---

## 2. Les grandes architectures CNN : innovations et enseignements

### 2.1 LeNet-5 (LeCun et al., 1998)

La première architecture CNN opérationnelle à grande échelle, conçue pour la reconnaissance de chiffres manuscrits (dataset MNIST). Architecture : 2 couches convolutives (5×5) + 2 couches de pooling moyen + 3 couches fully connected. Total : ~60 000 paramètres.

**Innovation :** démonstration que les features hiérarchiques apprises par rétropropagation surpassent les features artisanales pour la reconnaissance de caractères.

**Limite :** impossible à entraîner efficacement sur GPU à l'époque — le passage à l'échelle attendait AlexNet 14 ans.

### 2.2 AlexNet (Krizhevsky, Sutskever & Hinton, 2012)

Le point d'inflexion historique de la vision par ordinateur moderne. Vainqueur de l'ImageNet Large Scale Visual Recognition Challenge (ILSVRC) 2012 avec un taux d'erreur top-5 de 15,3% contre 26,2% pour le second — une réduction d'erreur relative de 41%.

**Innovations :**
- Entraînement sur GPU (deux GTX 580 en parallèle).
- Fonction d'activation ReLU à la place de tanh/sigmoïde : convergence 6× plus rapide.
- Dropout (taux 0.5) sur les couches fully connected : régularisation efficace sans surcoût computationnel.
- Data augmentation (flips, crops, perturbations couleur) : expansion artificielle du dataset.
- Local Response Normalization (précurseur de BatchNorm, abandonné par la suite).

**Architecture :** 5 couches convolutives (noyaux 11×11, 5×5, 3×3) + 3 couches fully connected. Total : ~62 millions de paramètres.

### 2.3 VGGNet (Simonyan & Zisserman, 2014)

**Innovation principale :** démonstration que la profondeur est le facteur clé de la qualité des représentations, et que des petits filtres $3 \times 3$ empilés sont plus efficaces que de grands filtres. Deux filtres $3 \times 3$ ont le même champ réceptif qu'un filtre $5 \times 5$, mais deux fois moins de paramètres et deux non-linéarités au lieu d'une.

**Architectures :** VGG-16 (16 couches à poids) et VGG-19. Extrêmement régulières : blocs de convolutions $3 \times 3$ séparés par du max pooling. Paramètres : 138M pour VGG-16.

**Héritage :** VGGNet est encore utilisé comme extracteur de features et comme backbone dans des tâches de segmentation, notamment pour l'analyse de documents. Sa régularité la rend facile à comprendre et à modifier.

### 2.4 ResNet (He et al., 2015)

La contribution architecturale la plus influente de la décennie 2010. Résout le **problème de dégradation** : contrairement à l'intuition, les réseaux très profonds (> 20 couches) entraînés naïvement produisent des erreurs d'entraînement *plus élevées* que les réseaux moins profonds — non pas à cause du surapprentissage, mais à cause de la difficulté à optimiser des fonctions profondes.

**Le bloc résiduel (*residual block*) :**

```
x ──────────────────────────────── (+) ── y
│                                   ↑
└── Conv → BN → ReLU → Conv → BN ──┘
         (F(x, W))
```

La sortie est $y = F(x, \{W_i\}) + x$ — le réseau apprend la **résiduelle** $F(x) = y - x$ plutôt que la transformation directe $y = H(x)$.

**Pourquoi ça marche :** si la transformation optimale est proche de l'identité, apprendre $F(x) \approx 0$ est plus facile qu'apprendre $H(x) \approx x$. Les skip connections fournissent des **raccourcis de gradient** qui permettent la rétropropagation à travers des centaines de couches sans dégradation.

**Résultats :** ResNet-152 (152 couches) a remporté ILSVRC 2015 avec 3,57% d'erreur top-5 — sous le niveau humain estimé à ~5%.

**Variantes :** ResNeXt (convolutions groupées), Wide ResNet (plus large, moins profond), ResNet-D (modifications des couches de stride pour améliorer le flux de l'information).

### 2.5 DenseNet (Huang et al., 2016)

Dans un bloc dense, **chaque couche reçoit les feature maps de toutes les couches précédentes** (concaténation, pas addition) :

$$x_l = H_l([x_0, x_1, \ldots, x_{l-1}])$$

**Avantages :** réutilisation maximale des features, gradients courts, régularisation naturelle, réduction du nombre de paramètres par rapport aux réseaux de même profondeur.

**Pertinence pour les documents :** les features de bas niveau (bords, textures locales d'encre) restent accessibles aux couches profondes sans dégradation — ce qui peut être utile pour la détection fine de tracés d'écriture.

### 2.6 EfficientNet (Tan & Le, 2019)

Propose un **scaling composé** (*compound scaling*) : plutôt que d'augmenter la profondeur, la largeur ou la résolution séparément, on les augmente ensemble selon un facteur de scaling $\phi$ :

$$\text{profondeur} : d = \alpha^\phi, \quad \text{largeur} : w = \beta^\phi, \quad \text{résolution} : r = \gamma^\phi$$

avec $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$ (contrainte de budget computationnel).

Le backbone est basé sur des **MBConv** (Mobile Inverted Bottleneck Convolution) avec squeeze-and-excitation, hérités de MobileNetV3.

**Résultat :** EfficientNet-B7 atteint 84,4% de précision top-1 sur ImageNet avec 8× moins de paramètres que les meilleurs modèles ResNet comparables.

**EfficientNetV2** (2021) remplace certains blocs MBConv par des blocs Fused-MBConv pour accélérer l'entraînement.

### 2.7 Architectures spécialisées pour la segmentation

Les architectures précédentes sont des classificateurs (une étiquette par image). Pour la segmentation (une étiquette par pixel), il faut des architectures encodeur-décodeur.

**U-Net (Ronneberger et al., 2015)**
Architecture en U : un encodeur qui sous-échantillonne (perd la résolution pour gagner la sémantique) + un décodeur qui suréchantillonne + des **skip connections** entre encodeur et décodeur au même niveau de résolution. Ces connections transmettent les détails spatiaux fins que l'encodeur aurait autrement perdus.

Conçu initialement pour la segmentation d'images médicales, U-Net est la référence pour la segmentation de layout dans les documents. Une variante avec un encodeur ResNet-50 pré-entraîné est souvent appelée **ResUNet**.

**Fully Convolutional Network (FCN, Long et al., 2015)**
Premier modèle à réaliser une segmentation sémantique dense *entièrement* avec des convolutions (remplacement des couches fully connected par des convolutions $1 \times 1$). Précurseur de toutes les architectures de segmentation modernes.

---

## 3. Les biais inductifs des CNN : formalisation

La notion de **biais inductif** (*inductive bias*) désigne les hypothèses implicites qu'une architecture encode sur la structure des données. Ces hypothèses permettent la généralisation (on ne peut pas apprendre sans hypothèses — c'est le théorème *no free lunch*), mais elles contraignent aussi les types de régularités que le modèle peut apprendre efficacement.

Les CNN encodent trois biais inductifs principaux.

### 3.1 Biais 1 — Invariance à la translation (*translation equivariance / invariance*)

Une convolution est **équivariante à la translation** : si l'entrée est décalée de $(t_x, t_y)$ pixels, la sortie est décalée du même vecteur. Formellement :

$$f(\text{Translate}_{(t_x, t_y)}(I)) = \text{Translate}_{(t_x, t_y)}(f(I))$$

Le pooling ajoute l'**invariance** (le décalage ne change plus la sortie) sur une petite fenêtre.

**Quand c'est utile :** classification d'objets. Un chat en haut à gauche ou en bas à droite doit produire la même étiquette « chat ».

**Quand c'est problématique — les manuscrits :**

- La **position relative** des éléments est porteuse d'information. Dans un mot gothique, le premier jambage et le troisième ne sont pas équivalents — leur position distingue des lettres différentes.
- La **mise en page** est informative : une zone en haut au centre est probablement une rubrique ; une zone en bas à droite est probablement une réclame. Un modèle invariant à la translation ne « sait » pas où il se trouve dans la page.
- L'**ordre de lecture** (de gauche à droite, de haut en bas) est une structure globale que l'invariance locale détruit.

### 3.2 Biais 2 — Localité (*locality*)

Un filtre $k \times k$ ne connecte chaque neurone de sortie qu'à $k^2$ pixels d'entrée. C'est le **biais de localité** : l'hypothèse que les features utiles sont locales — qu'un pixel est plus corrélé à ses voisins immédiats qu'à des pixels distants.

**Quand c'est utile :** pour les images naturelles, c'est généralement vrai. Les bords, textures et formes simples sont des structures locales.

**Quand c'est problématique — les manuscrits :**

- **Les ligatures inter-mots.** Dans un manuscrit cursif, les dernières lettres d'un mot et les premières du suivant peuvent se toucher physiquement. La segmentation correcte dépend d'une information non locale (la frontière de mot n'est pas visible localement).

- **La résolution de l'ambiguïté des minimes.** Rappel (section 1.1) : une suite de jambages verticaux ne peut être interprétée que par le contexte lexical. Ce contexte peut être à plusieurs centaines de pixels de distance — bien au-delà du champ réceptif d'une couche convolutive précoce.

- **La cohérence globale de ligne.** La hauteur de ligne, l'espacement entre les lettres et l'inclinaison du tracé sont des propriétés *de la ligne entière* — une structure globale qu'un filtre local ne perçoit pas directement.

- **Les abréviations.** Un signe abréviatif (un tilde au-dessus d'une lettre) renvoie à une information absente visuellement. Un modèle purement local peut détecter le tilde, mais ne peut pas *associer* le tilde à la lettre qu'il modifie sans une forme de mémoire longue portée.

### 3.3 Biais 3 — Partage des paramètres (*weight sharing*)

Le même filtre est appliqué à toutes les positions spatiales de l'image. Cela suppose que les features utiles sont les mêmes indépendamment de leur localisation.

**Quand c'est utile :** pour la détection de bords ou de textures dans des images naturelles — un bord vertical est un bord vertical qu'il soit à gauche ou à droite.

**Quand c'est problématique — les manuscrits :**

- Un **s long** (ʃ) n'apparaît qu'en début ou en milieu de mot, jamais en fin de mot. La même forme graphique a une signification différente selon sa position. Un filtre partagé appliqué uniformément ne peut pas encoder cette dépendance positionnelle.

- Les **lettrines** sont des lettres en position initiale de section, souvent de grande taille. Leur interprétation (c'est une lettre, pas une illustration) dépend de leur position dans la page — information ignorée par le partage de paramètres.

---

## 4. Coût computationnel du contexte global dans les CNN

### 4.1 Le problème de l'agrégation longue portée

Pour qu'un neurone dans un CNN voit l'intégralité d'une image, il faut empiler suffisamment de couches pour que le champ réceptif couvre la dimension de l'image. Pour une image de dimension $N$, il faut $O(\log N)$ couches de pooling (chacune divisant la résolution par 2) ou $O(N/k)$ couches convolutives sans pooling (avec des filtres de taille $k$).

Ces deux approches ont un coût :
- Le pooling agressif dégrade la résolution spatiale irrémédiablement — on ne peut pas récupérer les détails perdus dans le décodeur sans information supplémentaire (d'où les skip connections de U-Net).
- L'empilement de couches sans pooling est computationnellement coûteux et difficile à optimiser (même avec ResNet).

### 4.2 Complexité computationnelle

Pour une couche convolutive avec des filtres $k \times k$ sur une entrée $C_{in}$ canaux de résolution $H \times W$ produisant $C_{out}$ cartes :

$$\text{FLOPs} = 2 \times H \times W \times C_{in} \times C_{out} \times k^2$$

La complexité est **quadratique en la résolution spatiale** $H \times W$ et **linéaire en la profondeur** (nombre de canaux). Pour traiter une page de manuscrit à résolution native ($4000 \times 5500$), la mémoire nécessaire pour les feature maps intermédiaires est prohibitive sans réduction de résolution agressive.

**En pratique,** les pipelines CNN pour les documents travaillent systématiquement sur des images redimensionnées à $224 \times 224$ ou $448 \times 448$ — une réduction de résolution de 100× à 400× par rapport au scan original. Cette réduction détruit les détails fins des lettres, ce qui est l'une des raisons pour lesquelles les CNN pré-entraînés sur ImageNet échouent sur les manuscrits sans adaptation.

### 4.3 L'approche « sliding window » et ses limites

Une alternative au redimensionnement est de découper l'image en fenêtres (*patches*) et de traiter chaque fenêtre indépendamment, puis d'agréger les résultats.

**Le problème :** cette approche ignore les dépendances entre fenêtres. Un mot coupé en deux par la frontière d'une fenêtre est traité comme deux séquences indépendantes. Sur des manuscrits avec des lignes longues et des mots composés, ce découpage introduit des erreurs systématiques difficiles à corriger en post-traitement.

**La solution partielle :** introduire du recouvrement (*overlap*) entre les fenêtres, au coût d'un calcul redondant. Mais cela ne règle pas fondamentalement le problème de la dépendance entre fenêtres.

---

## 5. Ce que les CNN font bien sur les documents

Pour être complet, il faut souligner que les CNN ne sont pas inadaptés à *tous* les problèmes de traitement de documents. Plusieurs tâches tirent parti de leurs forces.

**Classification de type de document**
Classer une page en « charte », « registre », « roman » ou « traité » est une tâche où l'information est distribuée globalement (texture générale de la page, densité du texte, présence d'illustrations) mais ne requiert pas une localisation fine. Un ResNet-50 pré-entraîné et fine-tuné sur quelques centaines d'exemples obtient d'excellents résultats.

**Détection de bords et de structures locales**
Pour la segmentation de lignes sur des pages à mise en page simple (une colonne, peu de dégradations), des modèles U-Net avec backbone CNN atteignent des performances comparables aux approches Transformer, avec un coût computationnel inférieur.

**Extraction de features pour le clustering**
Les features extraites par un ResNet-50 pré-entraîné (ImageNet) — même sans fine-tuning — capturent suffisamment d'information sur la texture et le style graphique pour que du clustering k-means ou UMAP sur ces features produise des regroupements de pages par main cohérents. DINOv2 fait mieux, mais ResNet reste une baseline utile.

**Reconnaissance de caractères isolés (tâche auxiliaire)**
Pour des tâches d'aide à la paléographie (identifier automatiquement des lettres ambiguës soumises par un chercheur), la classification CNN de caractères isolés reste pertinente. C'est une sous-tâche différente de l'HTR end-to-end.

---

## 6. Architectures hybrides CNN-RNN : le CRNN

Avant l'ère des Transformers, le modèle de référence pour l'HTR était le **CRNN** (*Convolutional Recurrent Neural Network*), introduit par Shi et al. (2015) pour la reconnaissance de texte dans les scènes naturelles, puis adopté massivement pour les manuscrits.

### 6.1 Architecture

```
Image de ligne (H × W × 1)
        ↓
    CNN (VGG-like)
        ↓
Feature maps (H/32 × W/4 × 512)
        ↓
  Collapse vertical
(séquence de colonnes : W/4 × 512)
        ↓
   BiLSTM (2 couches)
        ↓
  Dense → Softmax
        ↓
Séquence de distributions sur l'alphabet
        ↓
    CTC Decoding
        ↓
      Texte
```

**L'étape clé :** après le CNN, on aplatit la dimension verticale (en supposant que la ligne de texte est horizontalement orientée et de hauteur normalisée) pour obtenir une séquence de vecteurs de features — un vecteur par « tranche » verticale de l'image. Cette séquence est passée au BiLSTM, qui peut propager le contexte dans les deux directions horizontales.

### 6.2 La CTC Loss (*Connectionist Temporal Classification*)

Le problème d'entraînement du CRNN est un problème d'alignement : on dispose d'une séquence de labels (le texte de référence) et d'une séquence de prédictions (une distribution sur l'alphabet pour chaque colonne de l'image), mais **l'alignement entre les deux est inconnu** — on ne sait pas quelle colonne de l'image correspond à quelle lettre du texte.

La CTC Loss (Graves et al., 2006) résout ce problème sans annotation d'alignement explicite. Elle somme les probabilités de toutes les séquences d'alignements possibles qui, après décodage (suppression des répétitions et du caractère blanc), donnent la séquence cible.

$$\mathcal{L}_{\text{CTC}} = -\ln P(y | x) = -\ln \sum_{\pi \in \mathcal{B}^{-1}(y)} P(\pi | x)$$

où $\mathcal{B}$ est la fonction de décodage (suppression des blancs et des répétitions) et $y$ est la séquence cible.

Cette loss est calculée efficacement par programmation dynamique (algorithme forward-backward).

### 6.3 Limites du CRNN pour les manuscrits médiévaux

- **Biais horizontal fort :** le collapse vertical suppose que la ligne est parfaitement horizontale et que toute l'information pertinente est dans la dimension horizontale. Les lignes ondulées des manuscrits sur parchemin, ou les lignes légèrement inclinées, dégradent les performances.
- **Mémoire limitée des LSTM :** les longues lignes de texte (> 100 caractères) posent des problèmes de mémoire aux LSTM — les dépendances très longues portée sont mal capturées.
- **Pas de mécanisme d'attention :** le CRNN agrège les features de toute la ligne avec le même poids — il n'y a pas de mécanisme permettant de « focaliser » sur une zone spécifique pour produire un caractère donné.

Ces trois limites sont précisément ce que les architectures basées sur l'attention (TrOCR, section 3 du cours) viennent résoudre.

---

## 7. Récapitulatif : pourquoi les CNN atteignent leurs limites sur les manuscrits médiévaux

| Propriété des CNN | Avantage général | Problème spécifique pour les manuscrits |
|-------------------|-----------------|----------------------------------------|
| Localité des filtres | Efficacité computationnelle | Contexte global nécessaire pour lever les ambiguïtés (minimes, abréviations) |
| Invariance à la translation | Robustesse aux décalages | Position porteuse d'information (ordre de lecture, lettrines, rubriques) |
| Partage des paramètres | Réduction du nombre de paramètres | Statistiques spatiales non stationnaires (s long en début ≠ s en fin) |
| Réduction de résolution (pooling) | Montée en sémantique | Perte irréversible des détails fins des lettres |
| Champ réceptif limité | Extraction de features locales | Impossible de capturer la structure globale de la page |
| Entraînement sur ImageNet | Bonne initialisation | Distribution shift fort : les images naturelles ≠ les manuscripts |

La section 2.2 introduira les Vision Transformers, dont le mécanisme d'attention est conçu précisément pour dépasser ces limites — au prix d'une complexité quadratique en le nombre de tokens, et d'une dépendance aux grandes quantités de données de pré-entraînement.

---

## Glossaire des termes avancés

**Atrous convolution (convolution dilatée)**
Convolution dont le filtre est « dilaté » par insertion de zéros entre ses éléments, permettant d'augmenter le champ réceptif sans augmenter le nombre de paramètres ni réduire la résolution spatiale.

**Biais inductif (*inductive bias*)**
Ensemble des hypothèses implicites encodées dans une architecture sur la structure des données. Permet la généralisation depuis un nombre fini d'exemples. Sans biais inductif, l'apprentissage est impossible (théorème *no free lunch*).

**Batch Normalization**
Technique de normalisation des activations inter-couches sur le mini-batch courant. Stabilise l'entraînement, permet des taux d'apprentissage plus élevés, agit comme régularisation implicite.

**BiLSTM (*Bidirectional Long Short-Term Memory*)**
Réseau récurrent qui traite la séquence dans les deux directions (gauche→droite et droite→gauche) et concatène les états cachés. Permet d'exploiter le contexte gauche et droit simultanément.

**Champ réceptif (théorique / effectif)**
Région de l'image d'entrée qui influence un neurone donné. Le champ réceptif théorique est calculé algébriquement ; le champ réceptif effectif, mesuré empiriquement via les gradients, est significativement plus petit.

**CTC Loss (*Connectionist Temporal Classification*)**
Fonction de perte permettant d'entraîner des modèles séquence-à-séquence sans alignement explicite entre entrée et sortie. Marginalise sur tous les alignements possibles par programmation dynamique.

**Compound scaling (scaling composé)**
Stratégie d'EfficientNet : augmenter simultanément et proportionnellement la profondeur, la largeur et la résolution d'un réseau selon un facteur $\phi$ sous contrainte de budget computationnel fixe.

**Cross-correlation (corrélation croisée)**
Opération mathématiquement proche de la convolution mais sans retournement du filtre. C'est l'opération réellement implémentée dans les CNN (par convention appelée « convolution »).

**DenseNet (*Densely Connected Network*)**
Architecture où chaque couche reçoit les feature maps de *toutes* les couches précédentes (par concaténation). Maximise la réutilisation des features et le flux des gradients.

**Depthwise separable convolution**
Factorisation d'une convolution $k \times k$ en deux opérations : une convolution dépthwise (un filtre par canal, $k \times k$) + une convolution pointwise ($1 \times 1$). Réduit le nombre de FLOPs d'un facteur $\approx k^2$.

**Distribution shift**
Différence entre la distribution des données d'entraînement et la distribution des données en production. Cause principale de la dégradation des performances en déploiement. Pour les manuscripts : shift fort entre ImageNet (photos naturelles) et scans de documents anciens.

**Equivariance à la translation**
Propriété d'un opérateur $f$ tel que $f(\text{Translate}(x)) = \text{Translate}(f(x))$. Contrairement à l'invariance (la sortie ne change pas), l'équivariance préserve l'information de position dans la sortie.

**Feature map (carte de caractéristiques)**
Tenseur de sortie d'une couche convolutive. Chaque « plan » de la feature map correspond à la réponse d'un filtre appliqué à l'entrée.

**FLOPs (*Floating Point Operations*)**
Mesure du coût computationnel d'une opération ou d'un modèle. 1 FLOP ≈ une addition ou multiplication en virgule flottante. Ne doit pas être confondu avec le temps de calcul (qui dépend du matériel).

**Gradient vanishing / exploding**
Phénomènes qui affectent la rétropropagation dans les réseaux profonds : les gradients tendent vers 0 (vanishing) ou divergent (exploding) en se propageant vers les couches précoces. ResNet résout le vanishing par les skip connections.

**Inductive bias** — voir *Biais inductif*.

**Layer Normalization**
Normalisation des activations sur les dimensions de features (plutôt que sur le batch). Indépendante de la taille du batch ; préférée dans les Transformers.

**MBConv (*Mobile Inverted Bottleneck Convolution*)**
Bloc de base de MobileNetV3 et EfficientNet : expansion des canaux par une convolution $1 \times 1$, convolution dépthwise $k \times k$, réduction des canaux par une convolution $1 \times 1$ + squeeze-and-excitation.

**No free lunch (théorème)**
Théorème de Wolpert & Macready (1997) : en moyennant sur tous les problèmes possibles, tous les algorithmes d'optimisation ont la même performance. Toute amélioration sur une classe de problèmes se paye par une dégradation sur d'autres. Implique que les biais inductifs sont inévitables et doivent être choisis en accord avec le domaine.

**Padding (same / valid / causal)**
Ajout de valeurs (généralement 0) autour de l'image avant la convolution. *Same* : préserve la résolution spatiale. *Valid* : pas de padding, la sortie est plus petite. *Causal* : ne regarde que le passé (convolutions 1D).

**ReLU (*Rectified Linear Unit*)**
Fonction d'activation $f(x) = \max(0, x)$. Non saturante pour $x > 0$ — les gradients ne s'annulent pas dans la zone positive, accélérant la convergence. Remplace sigmoid et tanh dans les architectures modernes.

**Residual connection (skip connection)**
Connexion qui contourne une ou plusieurs couches en ajoutant l'entrée à la sortie. Résout le problème de dégradation dans les réseaux très profonds en fournissant des raccourcis de gradient.

**Skip connection** — voir *Residual connection*.

**Squeeze-and-Excitation (SE)**
Module d'attention par canal : agrège globalement les feature maps (squeeze), produit des poids par canal (excitation), et rééchelonne les feature maps en conséquence. Permet au réseau de pondérer l'importance relative des canaux.

**Stride**
Pas de déplacement du filtre lors de la convolution. $s = 1$ : déplacement d'un pixel. $s = 2$ : résolution divisée par 2 (équivalent au pooling).

**Transfer learning (apprentissage par transfert)**
Réutilisation de paramètres appris sur une tâche source (généralement ImageNet) pour initialiser un modèle sur une tâche cible différente. Réduit considérablement la quantité de données nécessaires pour la tâche cible.

**U-Net**
Architecture encodeur-décodeur avec skip connections entre encodeur et décodeur au même niveau de résolution. Référence pour la segmentation sémantique dense, notamment sur les documents.

**Weight sharing (partage des paramètres)**
Utilisation du même filtre à toutes les positions spatiales dans une couche convolutive. Réduit drastiquement le nombre de paramètres et encode le biais d'invariance à la translation.

---

## Bibliographie de référence

### Articles fondateurs des architectures CNN

- **LeCun, Y., Bottou, L., Bengio, Y., Haffner, P.** (1998). *Gradient-based learning applied to document recognition*. Proceedings of the IEEE, 86(11), 2278–2324. — LeNet-5. Ironie historique : le premier grand CNN a été conçu pour lire des chiffres manuscrits.

- **Krizhevsky, A., Sutskever, I., Hinton, G. E.** (2012). *ImageNet Classification with Deep Convolutional Neural Networks*. NeurIPS 2012 (NIPS). — AlexNet. Le point d'inflexion historique. À lire comme document fondateur.

- **Simonyan, K., Zisserman, A.** (2014). *Very Deep Convolutional Networks for Large-Scale Image Recognition*. ICLR 2015. [arXiv:1409.1556] — VGGNet. Analyse rigoureuse de l'impact de la profondeur.

- **He, K., Zhang, X., Ren, S., Sun, J.** (2016). *Deep Residual Learning for Image Recognition*. CVPR 2016. [arXiv:1512.03385] — ResNet. L'article le plus cité de l'histoire de la vision par ordinateur (> 100 000 citations).

- **Huang, G., Liu, Z., van der Maaten, L., Weinberger, K. Q.** (2017). *Densely Connected Convolutional Networks*. CVPR 2017. [arXiv:1608.06993] — DenseNet.

- **Tan, M., Le, Q. V.** (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*. ICML 2019. [arXiv:1905.11946] — Compound scaling.

### Normalisation et régularisation

- **Ioffe, S., Szegedy, C.** (2015). *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*. ICML 2015. [arXiv:1502.03167]

- **Ba, J. L., Kiros, J. R., Hinton, G. E.** (2016). *Layer Normalization*. [arXiv:1607.06450]

- **Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., Salakhutdinov, R.** (2014). *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*. Journal of Machine Learning Research, 15, 1929–1958.

### Champ réceptif et biais inductifs

- **Luo, W., Li, Y., Urtasun, R., Zemel, R.** (2016). *Understanding the Effective Receptive Field in Deep Convolutional Neural Networks*. NeurIPS 2016. [arXiv:1701.04128] — Distinction cruciale entre champ réceptif théorique et effectif.

- **Mitchell, T. M.** (1980). *The Need for Biases in Learning Generalizations*. Technical Report CBM-TR-117, Rutgers University. — Article fondateur sur la notion de biais inductif en apprentissage automatique.

- **Wolpert, D. H., Macready, W. G.** (1997). *No Free Lunch Theorems for Optimization*. IEEE Transactions on Evolutionary Computation, 1(1), 67–82. — Formalisation du théorème *no free lunch*.

### Segmentation et architectures encodeur-décodeur

- **Ronneberger, O., Fischer, P., Brox, T.** (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI 2015. [arXiv:1505.04597] — U-Net. Référence pour la segmentation de layout dans les documents.

- **Long, J., Shelhamer, E., Darrell, T.** (2015). *Fully Convolutional Networks for Semantic Segmentation*. CVPR 2015. — FCN, précurseur de toutes les architectures de segmentation denses.

### HTR et architectures séquentielles

- **Shi, B., Bai, X., Yao, C.** (2016). *An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition*. IEEE TPAMI, 39(11), 2298–2304. [arXiv:1507.05717] — CRNN : l'architecture CNN-BiLSTM-CTC.

- **Graves, A., Fernández, S., Gomez, F., Schmidhuber, J.** (2006). *Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks*. ICML 2006. — CTC Loss.

- **Puigcerver, J.** (2017). *Are Multidimensional Recurrent Layers Really Necessary for Handwritten Text Recognition?* ICDAR 2017. — Analyse comparative montrant que des architectures CNN-LSTM simples surpassent les modèles plus complexes avec une bonne normalisation. Référence pratique pour le design d'architectures HTR.

### CNN pour les documents et les manuscrits

- **Chen, K. et al.** (2021). *A Survey of Vision-Language Pre-trained Models*. [arXiv:2202.10936] — Panorama des modèles vision-langage, utile pour situer TrOCR et CLIP dans l'écosystème général.

- **Simistira, F. et al.** (2016). *Recognition of Historical Greek Polytonic Scripts using LSTM Networks*. ICFHR 2016. — Application des architectures CNN-LSTM aux documents historiques.

- **Quirós, L., Vidal, E.** (2021). *Multi-task Handwritten Document Layout Analysis*. [arXiv:2103.05522] — Analyse conjointe du layout et de la reconnaissance de texte dans les documents historiques.

### Théorie de l'apprentissage profond (niveau avancé)

- **Goodfellow, I., Bengio, Y., Courville, A.** (2016). *Deep Learning*. MIT Press. Chapitres 9 (Convolutional Networks) et 11 (Practical Methodology). — Référence encyclopédique. Chapitre 9 couvre les fondements mathématiques de la convolution en détail.

- **Bishop, C. M., Bishop, H.** (2024). *Deep Learning: Foundations and Concepts*. Springer. — Référence récente, particulièrement claire sur les biais inductifs et la régularisation.

---

*Support de cours rédigé pour le module Computer Vision · Promotion MD5 · Mai 2026*
*Ce document accompagne la séance 2.1 du Jour 2. Il suppose une familiarité avec les réseaux de neurones et la rétropropagation.*
*Durée estimée de lecture : 60–75 minutes.*
