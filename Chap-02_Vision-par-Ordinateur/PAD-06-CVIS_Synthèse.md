# Cours 2.4 — Synthèse architecturale : choisir pour un projet réel

**Module 2 · Computer Vision appliquée aux manuscrits médiévaux · MD5**
**Séance de synthèse et discussion — 1 heure**

---

> *« En ingénierie, le meilleur outil n'est pas celui qui donne le score le plus élevé sur un benchmark — c'est celui qui résout le problème, dans les contraintes du projet, avec les ressources disponibles. »*

---

## Objectif de la séance

Les trois sections précédentes ont établi les fondements théoriques (2.1, 2.2) et la réalité empirique (2.3) des architectures CNN et Transformer pour la vision sur documents. Cette séance de synthèse a un double objectif :

**Pratique :** arrêter des choix architecturaux concrets pour chaque étape du pipeline HTR, avec leur justification. Ces choix guideront les Jours 3 à 5.

**Réflexif :** discuter des questions ouvertes que le TP et les cours ont naturellement soulevées. Ces questions n'ont pas de réponse unique — elles appellent une argumentation, des nuances, et parfois l'acceptation d'une incertitude légitime.

---

## 1. Tableau comparatif des architectures

Le tableau suivant résume les caractéristiques opérationnelles des architectures vues, selon des critères pertinents pour notre projet. Les scores sont indicatifs et issus de la littérature récente — ils ne valent que pour les configurations standard et les datasets de référence.

### 1.1 Architectures générales

| Critère | ResNet-50 | ViT-B/16 | Swin-B | CRNN | TrOCR-base |
|---------|-----------|----------|--------|------|-----------|
| **Paramètres** | 25 M | 86 M | 88 M | ~10 M | 334 M |
| **FLOPs (224²)** | 4,1 G | 17,6 G | 15,4 G | ~2 G | — |
| **ImageNet top-1** | 80,4% | 81,8% | 83,5% | — | — |
| **Données nécess. (scratch)** | ~1 M | >10 M | >1 M | ~100 k | >1 M |
| **Fine-tuning (petites données)** | Très bon | Moyen | Bon | Bon | Très bon |
| **Résolution native** | Flexible | Fixe (224) | Flexible | Flexible | Fixe (384) |
| **Complexité spatiale** | O(HW) | O(n²) | O(n) | O(HW) | O(n²) |
| **Segmentation dense** | Avec U-Net | Difficile | Natif | Non | Non |
| **Interprétabilité** | Grad-CAM | Attention maps | Attention maps | Faible | Moyenne |
| **Vitesse inférence CPU** | Rapide | Lente | Moyenne | Très rapide | Très lente |
| **Maturité de l'écosystème** | Très élevée | Élevée | Élevée | Élevée | Moyenne |

### 1.2 Caractéristiques spécifiques à l'HTR sur manuscrits médiévaux

| Critère | ResNet-50 | ViT-B/16 | Swin-B | CRNN | TrOCR-base |
|---------|-----------|----------|--------|------|-----------|
| **Champ réceptif sur ligne** | Local (limité) | Global | Semi-global | Global (BiLSTM) | Global |
| **Gestion des ligatures** | Faible | Bonne | Bonne | Bonne | Très bonne |
| **Robustesse à l'inclinaison** | Faible | Moyenne | Bonne | Moyenne | Bonne |
| **Gestion des abréviations** | Nulle | Via contexte | Via contexte | Via BiLSTM | Via LM decoder |
| **Sensibilité au bruit** | Élevée | Faible-moyenne | Faible | Moyenne | Faible |
| **Modèles médiévaux disponibles** | Peu | Peu | Peu | Kraken | HTR-United |
| **Dataset nécess. (fine-tuning)** | ~5 k images | ~2 k lignes | ~2 k lignes | ~500 lignes | ~500–2 k lignes |

---

## 2. Recommandations architecturales pour le pipeline HTR

### 2.1 Étape 1 — Prétraitement

**Choix : OpenCV + scikit-image (approche classique)**

Pas de deep learning ici. Les opérations de deskewing, binarisation (Sauvola) et normalisation du contraste (CLAHE) sont efficacement traitées par des algorithmes analytiques. Introduire un modèle neuronal pour le prétraitement ajouterait de la latence et de la complexité sans gain substantiel sur des scans de qualité standard.

**Exception :** pour des scans très dégradés (palimpsestes, transparence extrême, déchirures), des modèles de restauration d'image (comme DocUNet ou des diffusion models spécialisés) peuvent être envisagés — mais ils dépassent le périmètre de ce cours.

---

### 2.2 Étape 2 — Segmentation de layout

**Choix principal : SAM (Segment Anything Model)**

*Justification :*
- SAM ne nécessite pas d'entraînement sur des données de manuscripts : son backbone ViT-H, entraîné sur 1 milliard de masques, généralise remarquablement bien aux documents.
- Son mode de segmentation automatique (`SamAutomaticMaskGenerator`) produit des masques de toutes les régions cohérentes d'une image — y compris les colonnes de texte, les illustrations et les annotations marginales.
- Ses prompts (point, boîte) permettent une interaction humaine légère pour corriger les erreurs sans réentraînement.

*Limites :*
- SAM n'étiquette pas les régions : il dit « voici une région cohérente », pas « voici du texte » ou « voici une illustration ». Un classifieur léger (ResNet ou même des règles heuristiques sur la taille et la texture) est nécessaire en post-traitement.
- SAM-2 (2024) améliore les performances sur les documents mais reste plus lourd.

**Alternative : Swin + FPN + tête de segmentation**

Pour des projets avec suffisamment de données annotées (masques de layout), un modèle Swin fine-tuné sur des pages annotées peut surpasser SAM. C'est l'approche de systèmes comme LayoutParser ou DocBank.

---

### 2.3 Étape 3a — Segmentation de lignes

**Choix principal : Kraken (segmentation de baselines)**

*Justification :*
- Kraken utilise un modèle spécialement conçu pour détecter les **lignes de base** (*baselines*) des manuscrits — le trait sur lequel reposent les lettres — plutôt que des boîtes englobantes rectangulaires. Cette approche gère naturellement les lignes courbées, les lignes avec des éléments ascendants et descendants variables, et les mises en page non rectangulaires.
- Des modèles Kraken pré-entraînés pour les manuscrits médiévaux sont disponibles dans eScriptorium et HTR-United.
- La segmentation de Kraken produit directement des images de lignes prêtes pour l'HTR, dans le format attendu par ses propres modèles de reconnaissance.

**Alternative : DINO features + clustering**

Pour des pages à mise en page non standard (dont Kraken échoue à détecter les lignes), les features DINOv2 permettent un clustering des pixels en zones texte/non-texte, depuis lequel on peut extraire des lignes par projection horizontale. Cette approche est plus robuste mais moins précise.

---

### 2.4 Étape 3b — Description des illustrations

**Choix principal : CLIP (zero-shot) puis LLaVA (few-shot)**

*Justification :*
- CLIP permet une classification zero-shot des types d'illustrations (enluminure narrative, diagramme, carte, lettrine ornée, initiale filigranée) sans aucune annotation. Les descriptions textuelles candidates sont passées à l'encodeur texte et comparées à l'encodeur image.
- Pour des descriptions plus riches (« un ange couronnant un roi devant une cour »), LLaVA ou Florence-2 — des modèles vision-langage — produisent des descriptions en langage naturel adaptées à la documentation académique.

*Limite de CLIP pour les manuscripts :*
CLIP a été entraîné sur des paires image-texte du web — il connaît les « illuminated manuscripts » mais avec un biais vers les représentations les plus célèbres. Des scènes iconographiques rares ou des schémas alchimiques complexes seront décrits de façon générique.

---

### 2.5 Étape 4 — HTR par ligne

**Choix principal : TrOCR fine-tuné sur données médiévales**

*Justification :*
- L'architecture encodeur ViT (BEiT pré-entraîné) + décodeur GPT-2 capture à la fois les représentations visuelles locales (détails des traits) et le contexte global de la ligne (dépendances longue portée).
- Le décodeur autorégressif peut intégrer un modèle de langue en aval — contrairement à la CTC, qui est conditionnellement indépendante. Pour l'ancien français dont le vocabulaire est non normalisé, cette flexibilité est précieuse.
- Des checkpoints pré-entraînés (`microsoft/trocr-base-handwritten`) offrent un bon point de départ, réduit à ~500–2000 lignes annotées pour le fine-tuning.

**Alternative : Kraken (modèle HTR)**

Pour des projets nécessitant une inférence légère (CPU, pas de GPU), Kraken en mode HTR avec un modèle LSTM-CTC pré-entraîné sur des données médiévales (disponibles via HTR-United) est une alternative viable. Moins performant que TrOCR sur les textes difficiles, mais déployable sans infrastructure GPU.

**Comparaison TrOCR vs Kraken sur manuscrits médiévaux**

| Critère | TrOCR (fine-tuné) | Kraken (médiéval) |
|---------|------------------|------------------|
| CER moyen sur corpus médiéval | 8–15% | 10–20% |
| Vitesse inférence (GPU) | ~50 lignes/s | ~200 lignes/s |
| Vitesse inférence (CPU) | ~2 lignes/s | ~20 lignes/s |
| Fine-tuning nécessaire | Oui (~1 000 lignes) | Oui (~500 lignes) |
| Modèles médiévaux disponibles | Non (à fine-tuner) | Oui (HTR-United) |
| Gestion des scripts exotiques | Limitée | Bonne |

---

### 2.6 Récapitulatif du pipeline : architecture choisie

```
Scan TIFF
    ↓
OpenCV + Sauvola                  [Prétraitement — algorithmique]
    ↓
SAM (ViT-H)                       [Layout — zero-shot]
    ↓
Kraken segment                    [Lignes — modèle médiéval]
    ↓
TrOCR fine-tuné                   [HTR — ViT encoder + GPT-2 decoder]
    ↓
Vote pondéré (TrOCR + Kraken)     [Ensemble — robustesse]
    ↓
CLIP / LLaVA                      [Illustrations — zero/few-shot]
    ↓
JSON + TEI optionnel              [Export — module NLP]
```

---

## 3. Questions de discussion — avec éléments de réponse

Les questions suivantes sont issues des observations du TP et des contenus des sections 2.1–2.3. Elles sont posées de façon ouverte, mais accompagnées d'éléments destinés à guider et structurer la discussion.

---

### Question 1 — Faut-il toujours utiliser l'architecture la plus performante sur les benchmarks ?

**Formulation complète**
ViT-B/16 surpasse ResNet-50 sur ImageNet et sur la majorité des benchmarks de vision modernes. Pourtant, pour certaines étapes de notre pipeline, nous choisissons Kraken (architecture LSTM-CTC, conçue en 2019) plutôt que TrOCR. Est-ce une régression technologique ? Quels critères justifient de ne pas toujours choisir l'état de l'art ?

**Éléments pour orienter la discussion**

- **Le benchmark n'est pas le problème.** Un score élevé sur ImageNet mesure la capacité à classer des photos naturelles. Ce n'est pas notre problème. La performance pertinente est le CER sur des manuscrits médiévaux — un domaine où ImageNet-top-1 est un prédicteur faible.

- **Le coût total de possession.** TrOCR-base (334 M paramètres) nécessite un GPU pour une inférence en temps raisonnable. Kraken tourne en CPU. Dans un contexte de déploiement en production (une bibliothèque qui traite 10 000 pages par nuit sur un serveur standard), le choix de l'architecture a des implications budgétaires directes.

- **La maturité de l'écosystème.** Kraken dispose de modèles pré-entraînés sur des données médiévales francophones (HTR-United, CREMMA), validés par la communauté des humanités numériques. TrOCR nécessite un fine-tuning depuis un checkpoint générique — ce qui suppose des données annotées, du temps GPU, et une expertise en entraînement de modèles.

- **La robustesse hors-distribution.** Pour des écritures très rares ou des conditions de numérisation extrêmes, un modèle plus simple mais entraîné sur des données proches est souvent plus robuste qu'un grand modèle pré-entraîné sur des données génériques.

**Synthèse**

Le choix d'une architecture doit être guidé par quatre critères ordonnés : (1) est-elle adaptée au problème et à ses contraintes propres ? (2) dispose-t-on des ressources pour l'entraîner et la déployer ? (3) existe-t-il des données et des modèles pré-entraînés dans le domaine ? (4) quelle est sa performance effective sur les données du projet, mesurée avec les métriques du projet ?

L'état de l'art sur un benchmark général n'est pertinent que si ce benchmark est un proxy fidèle du problème réel — ce qui n'est presque jamais entièrement vrai.

**Idée reçue à déconstruire**
*« Plus grand = meilleur. »* — Non. GPT-4 est un meilleur modèle de langage que distilBERT, mais distilBERT peut être suffisant pour de la classification de sentiment sur des tweets, déployable sur mobile, avec une latence de 5 ms. La question n'est pas quelle architecture est la meilleure en absolu, mais quelle architecture est optimale sous contraintes.

---

### Question 2 — Les CNN sont-ils vraiment dépassés pour les documents ?

**Formulation complète**
Les sections 2.1 et 2.3 ont montré les limites des CNN pour l'HTR. Pourtant, de nombreux systèmes de production utilisent encore des architectures convolutives. ConvNeXt (2022) obtient des performances comparables aux ViT avec une architecture entièrement convolutive. Les CNN sont-ils structurellement inférieurs aux Transformers pour les documents, ou simplement différents ?

**Éléments pour orienter la discussion**

- **ConvNeXt : quand le CNN adopte le design des ViT.** Liu et al. (2022) ont montré que les CNN modernisés (activation GELU, LayerNorm, dépthwise separable convolutions, grands noyaux 7×7, moins de couches avec plus de canaux) atteignent des performances comparables aux Swin Transformers sur ImageNet et les tâches de détection. La frontière CNN/ViT s'estompe.

- **Le biais inductif comme atout sur petites données.** Pour des datasets de fine-tuning petits (< 500 lignes), le biais de localité des CNN peut être un avantage : le modèle converge plus vite et régularise mieux sans avoir besoin du pré-entraînement massif que les ViT requièrent. C'est l'argument principal pour continuer à utiliser des CRNN dans des projets HTR avec peu de données.

- **L'importance de la tâche.** Pour la *classification* de type de document, un ResNet-50 pré-entraîné atteint souvent les mêmes performances qu'un ViT fine-tuné, avec un coût computationnel 4× inférieur. Pour l'HTR sur lignes cursives complexes, les Transformers semblent structurellement avantagés (contexte global, pas de biais de localité). La supériorité de l'une ou l'autre architecture est donc tâche-dépendante.

- **Le problème de la résolution.** Les CNN (et Swin) sont naturellement plus flexibles en résolution que les ViT standard, qui nécessitent une interpolation des encodages positionnels au-delà de leur résolution d'entraînement. Pour le traitement de pages entières haute résolution, Swin ou les approches convolutives conservent un avantage pratique.

**Synthèse**

Les CNN ne sont pas dépassés — ils sont adaptés à un sous-ensemble de tâches et de contraintes. L'évolution récente (ConvNeXt, MobileViT, EfficientViT) tend vers des architectures hybrides qui empruntent les forces des deux paradigmes. La question opérationnelle n'est pas CNN vs Transformer, mais : quel compromis biais inductif / données / coût est optimal pour cette tâche et ces contraintes ?

**Idée reçue à déconstruire**
*« Les Transformers remplacent toujours les CNN. »* — La courbe d'adoption réelle montre que les CNN restent dominants dans les applications embarquées (smartphones, edge computing), les tâches de traitement vidéo temps réel, et les domaines avec peu de données. Les Transformers dominent les tâches de compréhension sémantique sur grandes données. Les deux coexistent et s'hybridisent.

---

### Question 3 — Que nous apprennent les différences d'attention maps entre les modèles supervisés et DINO ?

**Formulation complète**
Dans le TP, les attention maps de DINOv2 (entraîné sans annotation) semblent localiser le texte aussi bien, voire mieux, que celles de ViT-B/16 (entraîné avec labels ImageNet). Comment un modèle entraîné sans supervision peut-il « savoir » où se trouve le texte sur une page qu'il n'a jamais vue ?

**Éléments pour orienter la discussion**

- **Ce que la loss DINO optimise implicitement.** DINO entraîne l'étudiant à prédire les représentations du professeur sur des vues augmentées de la même image. Pour que deux vues très différentes d'une page (avec des crops, des flips, des distorsions de couleur) aient des représentations proches, le modèle doit apprendre des caractéristiques *invariantes aux transformations* — c'est-à-dire des structures sémantiques stables, pas des textures ou des couleurs locales. Le texte est une telle structure stable : il est visuellement cohérent, répété, et forme des patterns reconnaissables indépendamment des variations d'éclairage.

- **L'hypothèse de compacité des objets.** Les objets naturels (et les zones de texte) occupent des régions visuellement cohérentes et homogènes. La self-distillation favorise les représentations qui capturent cette cohérence spatiale — ce qui produit naturellement une segmentation émergente.

- **Pourquoi ViT supervisé (ImageNet) localise moins bien le texte.** ViT-B/16 supervisé a appris à discriminer 1000 classes d'objets naturels. Pour cette tâche, des features locales discriminantes (la forme d'une tête d'oiseau, la texture d'un pelage) sont souvent suffisantes — pas besoin de capturer la structure globale de la page. DINO, en revanche, doit capturer ce qui est *invariant* dans l'image, ce qui le pousse vers des features globales et structurelles.

- **L'implication pour les données d'entraînement.** Un modèle comme DINOv2 peut être pré-entraîné sur des images de manuscrits *sans aucune annotation*, et ses features seront directement exploitables pour le clustering de mains, la segmentation de layout, et l'initialisation de modèles HTR. C'est un argument fort pour l'auto-supervision dans les domaines patrimoniaux où les annotations sont rares et coûteuses.

**Synthèse**

L'auto-supervision force le modèle à apprendre des représentations invariantes aux transformations, ce qui favorise la capture de structures sémantiques globales plutôt que de textures locales discriminantes. Pour les documents, où la structure spatiale (lignes, colonnes, zones d'illustration) est plus informative que la texture locale, cet objectif d'apprentissage est mieux aligné avec la tâche que la classification supervisée sur ImageNet. C'est un exemple où la tâche d'entraînement auxiliaire (*pretext task*) de l'auto-supervision est accidentellement plus proche du problème réel que la tâche de supervision explicite.

**Pour aller plus loin**
Ce phénomène est documenté dans l'article DINO original (Caron et al., 2021, Figure 1) : les attention maps de DINO sur des images COCO segmentent les objets de façon quasi-parfaite, alors que les modèles supervisés comparables ne présentent pas cette propriété. Les auteurs l'attribuent à la combinaison de l'architecture ViT (champ réceptif global) et de la loss de distillation (invariance aux transformations).

---

### Question 4 — Comment choisir entre CTC et décodeur autorégressif pour l'HTR ?

**Formulation complète**
Kraken utilise une loss CTC pour l'HTR ; TrOCR utilise un décodeur autorégressif (cross-attention + softmax). Ce sont deux approches fondamentalement différentes pour produire une séquence de caractères depuis une image de ligne. Quels sont les compromis, et dans quels cas choisir l'une ou l'autre ?

**Éléments pour orienter la discussion**

**CTC (Connectionist Temporal Classification)**

*Avantages :*
- Entraînement sur des paires (image, texte) sans alignement caractère-position.
- Inférence rapide : décodage en une passe (beam search sur la séquence de distributions).
- Robuste aux variations de longueur de séquence.
- Fonctionne bien avec des architectures légères (CRNN, BiLSTM).

*Inconvénients :*
- Hypothèse d'indépendance conditionnelle entre les sorties : $P(\pi | x) = \prod_t P(\pi_t | x)$. Le modèle ne conditionne pas le caractère $t$ sur les caractères $1, \ldots, t-1$ déjà générés. Pour l'ancien français sans norme orthographique, cette absence de modèle de langage implicite est pénalisante.
- Ne peut pas intégrer un modèle de langue de façon native (on peut le faire en post-traitement avec un rescoring, mais c'est moins élégant).
- Tend à produire des erreurs locales (substitution d'un caractère) plutôt que des erreurs globales (confusion de mot entier), ce qui se reflète dans un CER plus bas mais un WER parfois plus élevé.

**Décodeur autorégressif (TrOCR)**

*Avantages :*
- Chaque token est conditionné sur tous les tokens précédents : $P(y_t | x, y_1, \ldots, y_{t-1})$. Le décodeur intègre un modèle de langue implicite puissant (GPT-2).
- Capable de corriger des erreurs locales par le contexte : si les caractères précédents suggèrent un mot, le décodeur peut corriger une ambiguïté graphique.
- Meilleure gestion des abréviations et des mots rares : le modèle de langue peut « compléter » une séquence qu'il a vue en entraînement.
- Plus flexible : on peut guider le décodeur avec des prompts (forcer la transcription diplomatique ou normalisée).

*Inconvénients :*
- Inférence lente : génération séquentielle, un token à la fois (sauf avec speculative decoding ou des optimisations spécifiques).
- Sensible à l'erreur de cascade : une erreur au token $t$ peut propager des erreurs aux tokens $t+1, t+2, \ldots$ (exposure bias).
- Nécessite un tokenizer adapté à la langue cible. Pour l'ancien français, le tokenizer de TrOCR (basé sur GPT-2 anglais) n'est pas optimal — des tokens fréquents en vieux français sont découpés en sous-mots rares, ce qui dégrade les performances.
- Plus difficile à entraîner (teacher forcing, scheduled sampling pour réduire l'exposure bias).

**Règle de décision pratique**

| Situation | Recommandation |
|-----------|---------------|
| Peu de données annotées (< 500 lignes) | CTC (CRNN/Kraken) — moins de paramètres, converge mieux |
| Texte hautement abrégé ou ambiguë | Décodeur AR (TrOCR) — le contexte aide à lever les ambiguïtés |
| Contrainte de latence forte (temps réel) | CTC — inférence en une passe |
| Haute qualité requise (édition académique) | Décodeur AR — meilleur CER final |
| Script non latin ou très rare | CTC avec Kraken — plus flexible sur les alphabets |
| Intégration d'un modèle de langue fort | Décodeur AR — intégration native |

**Synthèse**
La CTC est le choix raisonnable quand les ressources sont limitées et quand la rapidité est prioritaire. Le décodeur autorégressif est le choix optimal quand la qualité finale est prioritaire et quand on dispose de suffisamment de données et de ressources computationnelles. En pratique, l'approche ensemble (vote TrOCR + Kraken) combine les avantages des deux.

---

### Question 5 — L'interprétabilité des attention maps est-elle réelle ou illusoire ?

**Formulation complète**
Les attention maps sont séduisantes : elles donnent l'impression de « voir » ce que le modèle regarde. Mais une attention élevée sur une zone signifie-t-elle que le modèle *comprend* cette zone, ou simplement qu'il y trouve une feature utile pour sa tâche — qui peut être une corrélation spurieuse ? À quel point peut-on faire confiance aux attention maps pour diagnostiquer un modèle HTR ?

**Éléments pour orienter la discussion**

- **L'attention n'est pas l'explication.** Plusieurs travaux (Jain & Wallace, 2019 ; Wiegreffe & Pinter, 2019) ont montré que les poids d'attention ne constituent pas une explication causale des prédictions. On peut modifier les poids d'attention sans changer la prédiction, et deux modèles avec des distributions d'attention très différentes peuvent produire les mêmes prédictions. L'attention est *corrélée* avec ce qui compte pour le modèle, pas *causalement responsable* des prédictions.

- **L'Attention Rollout est une approximation.** L'hypothèse du mélange 50/50 entre attention et identité (skip connection) est arbitraire. Des travaux comme celui de Chefer et al. (2021) proposent des méthodes de relevance propagation plus précises — mais aussi beaucoup plus coûteuses à calculer.

- **Ce que l'attention dit réellement.** Dans un modèle HTR, une attention élevée du token de sortie $t$ sur le patch $j$ signifie que le patch $j$ contribue à la représentation utilisée pour prédire le token $t$. C'est une information sur le *flux d'information*, pas sur la *sémantique* de ce que le modèle a compris.

- **Cas où l'attention est trompeuse.** Un modèle peut « regarder » le bord droit de la ligne (où se trouve souvent la fin de ligne, donc la ponctuation) pour prédire la ponctuation — non pas parce qu'il comprend la ponctuation, mais parce que la position est un raccourci heuristique appris depuis les données d'entraînement. La carte d'attention montrera « le modèle regarde la ponctuation », mais la raison est une corrélation positionnelle, pas une compréhension syntaxique.

- **L'utilité pratique malgré les limites.** Même imparfaites, les attention maps sont utiles pour : identifier les zones systématiquement ignorées par le modèle (problème de segmentation en amont), détecter les cas où le modèle regarde la mauvaise ligne (erreur de segmentation de layout), et comparer qualitativement des modèles entre eux.

**Synthèse**

Les attention maps sont un outil de diagnostic utile mais non suffisant. Elles donnent des indications sur le flux d'information dans le modèle, mais ne constituent pas une explication causale des prédictions. Pour une évaluation rigoureuse, les métriques quantitatives (CER, WER, corrélation entre attention et masques de vérité terrain) doivent compléter l'analyse visuelle. L'interpretabilité est un domaine actif de recherche — les outils de 2024–2025 (LIME, SHAP adapté, relevance propagation) offrent des perspectives plus rigoureuses que la simple visualisation d'attention.

**Pour aller plus loin**
Ce débat est connu sous le nom de *faithfulness vs plausibility* : une explication est *plausible* si elle semble raisonnable à un humain ; elle est *fidèle* si elle reflète réellement le comportement interne du modèle. Les attention maps sont souvent plausibles mais rarement fidèles au sens strict.

---

### Question 6 — La distinction CNN / ViT est-elle encore pertinente en 2025–2026 ?

**Formulation complète**
ConvNeXt adopte des design choices de ViT dans une architecture convolutive et atteint des performances comparables. MobileViT hybride convolution et attention. Les Mamba state-space models offrent une alternative linéaire à l'attention. La frontière architecturale que nous avons tracée pendant ce module est-elle en train de s'effacer ? Quelles leçons tirer pour l'avenir du domaine ?

**Éléments pour orienter la discussion**

- **La convergence des design choices.** L'expérience de Liu et al. (ConvNeXt, 2022) est révélatrice : en adoptant systématiquement les choix de design de Swin (grandes fenêtres, LayerNorm, GELU, moins de downsampling agressif), ils ont transformé un ResNet en un modèle comparable à un Swin, *sans changer le mécanisme fondamental* (convolution vs attention). Cela suggère que les choix de design (normalisation, activation, ratio largeur/profondeur) comptent autant que le mécanisme d'agrégation.

- **Ce qui reste fondamentalement différent.** Malgré la convergence des performances, deux propriétés restent structurellement distinctes : (a) la complexité (O(n²) pour l'attention globale vs O(HW) pour la convolution) et (b) le biais inductif (localité et partage de paramètres vs flexibilité contextuelle). Ces différences ont des implications pratiques sur les domaines de données et les tâches où chaque architecture excelle.

- **L'émergence des State Space Models (SSM).** Mamba (Gu & Dao, 2023) propose un mécanisme de séquence avec complexité *linéaire* en la longueur — une alternative potentielle à l'attention quadratique. VMamba et Vision Mamba l'adaptent à la vision. Ces modèles sont encore jeunes (2023–2024), mais leur évolution pourrait redistribuer les cartes entre architectures.

- **Le problème de l'inductive bias reste entier.** Quelle que soit l'architecture, la question fondamentale demeure : quelles hypothèses sur la structure des données encode-t-elle ? Pour les manuscrits médiévaux, nous avons besoin d'un modèle sensible à la position, capable de contexte global, et robuste aux variations de style — un profil qui favorise les Transformers, qu'ils soient purs ou hybrides. L'architecture spécifique importe moins que son adéquation aux contraintes du problème.

**Synthèse**

La frontière CNN/Transformer s'estompe au niveau des performances sur les benchmarks généraux, mais reste pertinente au niveau des mécanismes et des propriétés structurelles. En pratique, ce qui émergera sera probablement une convergence vers des architectures hybrides, combinant la flexibilité de l'attention pour le contexte global et l'efficacité des convolutions pour les features locales — comme le montrent déjà MobileViT, EfficientViT et les hybrid ViT. Pour notre projet, ce qui compte est de choisir l'architecture appropriée à chaque étape du pipeline, en comprenant ce que chacune apporte structurellement — pas en suivant le classement du dernier benchmark.

**Prolongement prospectif**
La question « quelle architecture demain ? » est ouverte. Les SSM (Mamba) offrent une complexité linéaire potentiellement disruptive. Les diffusion models s'imposent en génération. Les foundation models (GPT-4V, Gemini, Claude) proposent une approche radicalement différente : ne pas choisir d'architecture pour chaque tâche, mais utiliser un modèle généraliste en zero ou few-shot. Pour l'HTR médiéval, des expériences récentes avec GPT-4V montrent des résultats encourageants en zero-shot sur des manuscrits lisibles — mais loin des performances des modèles spécialisés fine-tunés sur des données médiévales.

---

## 4. Récapitulatif des décisions architecturales du Jour 2

| Étape du pipeline | Architecture choisie | Raison principale | Alternative |
|------------------|---------------------|------------------|-------------|
| Prétraitement | OpenCV + Sauvola | Efficace, pas de données nécessaires | DocUNet (dégradations extrêmes) |
| Layout | SAM (ViT-H) | Zero-shot, robuste | Swin-B + UperNet (si données dispo) |
| Lignes | Kraken segment | Spécialisé manuscrits, baselines | DINO + clustering |
| HTR | TrOCR fine-tuné | Contexte global, modèle de langue | Kraken HTR (si contrainte CPU) |
| Ensemble | Vote pondéré TrOCR + Kraken | Complémentarité des erreurs | — |
| Illustrations | CLIP → LLaVA | Zero-shot, descriptions naturelles | Florence-2 |
| Export | JSON + TEI optionnel | Compatibilité humanités numériques | PAGE XML |

Ces choix ne sont pas définitifs. Le Jour 5 (évaluation et robustesse) nous permettra de les remettre en question à la lumière des résultats expérimentaux.

---

## Bibliographie complémentaire pour la synthèse

### Sur le choix d'architectures en pratique

- **Liu, Z. et al.** (2022). *A ConvNet for the 2020s*. CVPR 2022. [arXiv:2201.03545] — ConvNeXt : CNN modernisé rivalisant avec les ViT. Illustre que les design choices comptent autant que le mécanisme.

- **Dehghani, M. et al.** (2023). *Scaling Vision Transformers to 22 Billion Parameters*. ICML 2023. [arXiv:2302.05442] — ViT-22B : limites et possibilités du scaling pur.

- **Touvron, H. et al.** (2022). *Three Things Everyone Should Know About Vision Transformers*. ECCV 2022. [arXiv:2203.09795] — Analyse empirique des propriétés pratiques des ViT (fine-tuning, robustesse, interpretabilité).

### Sur l'interprétabilité

- **Jain, S., Wallace, B. C.** (2019). *Attention is not Explanation*. NAACL 2019. [arXiv:1902.10186] — Article fondateur du débat sur la faithfulness des attention maps.

- **Wiegreffe, S., Pinter, Y.** (2019). *Attention is not not Explanation*. EMNLP 2019. [arXiv:1908.04626] — Réponse nuancée à Jain & Wallace. La situation est plus complexe.

- **Chefer, H., Gur, S., Wolf, L.** (2021). *Transformer Interpretability Beyond Attention Visualization*. CVPR 2021. [arXiv:2012.09838] — Méthode de relevance propagation plus rigoureuse que l'Attention Rollout.

### Sur les architectures hybrides et émergentes

- **Chen, C.-F. et al.** (2021). *CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification*. ICCV 2021. [arXiv:2103.14899]

- **Mehta, S., Rastegari, M.** (2022). *MobileViT: Light-Weight, General-Purpose, and Mobile-Friendly Vision Transformer*. ICLR 2022. [arXiv:2110.02178]

- **Gu, A., Dao, T.** (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. [arXiv:2312.00752] — Architecture SSM à complexité linéaire, alternative potentielle aux Transformers.

- **Zhu, L. et al.** (2024). *Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model*. ICML 2024. [arXiv:2401.13660]

### Sur CTC vs décodeur autorégressif

- **Graves, A. et al.** (2006). *Connectionist Temporal Classification*. ICML 2006. — Fondement de la CTC.

- **Chan, W. et al.** (2016). *Listen, Attend and Spell: A Neural Network for Large Vocabulary Conversational Speech Recognition*. ICASSP 2016. — Premier décodeur autorégressif avec cross-attention pour la reconnaissance de séquences. Base conceptuelle de TrOCR.

- **Dutta, A. et al.** (2018). *Improving CNN-RNN Hybrid Networks for Handwriting Recognition*. ICFHR 2018. — Comparaison empirique CTC vs attention decoder sur des tâches HTR réelles.

---

*Support de cours rédigé pour le module Computer Vision · Promotion MD5 · Mai 2026*
*Ce document accompagne la séance 2.4 du Jour 2 et conclut le Module 2.*
*Il prépare directement les séances 3.1 à 3.5 du Jour 3 (DINO, CLIP, SAM, TrOCR).*
