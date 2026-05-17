# Cours 3.4 — TrOCR : Transformer-based HTR et stratégies de fine-tuning

**Module 3 · Computer Vision appliquée aux manuscrits médiévaux · MD5**

---

> *« Un modèle HTR est fondamentalement une fonction de traduction : elle traduit une image de ligne en une séquence de caractères. L'architecture encodeur-décodeur est le cadre naturel de toute traduction — qu'elle soit image-vers-texte ou français-vers-anglais. »*

---

## Introduction : de la segmentation à la transcription

Les sections 3.1 à 3.3 ont équipé notre pipeline pour détecter et localiser les régions d'une page de manuscrit. SAM isole les colonnes de texte ; Kraken segmente les lignes à l'intérieur de ces colonnes. Il reste maintenant la tâche centrale : **lire le texte dans chaque image de ligne**.

C'est le problème de la **Handwritten Text Recognition** (HTR) — et c'est le plus difficile. La segmentation de layout peut tolérer des imprécisions : une colonne légèrement mal découpée sera lue avec quelques caractères manquants. Une erreur de reconnaissance de caractère, elle, se propage directement dans la transcription et dans toutes les analyses linguistiques en aval.

**TrOCR** (*Transformer-based OCR*, Li et al., 2021) est le modèle HTR de référence dans l'écosystème HuggingFace. Il repose sur une architecture encodeur-décodeur entièrement basée sur des Transformers : un encodeur vision (ViT pré-entraîné par BEiT) et un décodeur de langue (GPT-2 autorégressif). Cette architecture unifie dans un seul modèle la représentation visuelle et la modélisation linguistique — ce qui lui confère une capacité de correction contextuelle absente des modèles CTC.

Cette section couvre l'architecture de TrOCR, les stratégies de fine-tuning adaptées à nos contraintes (données limitées, ressources GPU variables), et l'évaluation rigoureuse des performances sur des données médiévales.

---

## 1. Architecture de TrOCR

### 1.1 Vue d'ensemble : une traduction d'image en texte

TrOCR reformule l'HTR comme un problème de **traduction de séquences** (*sequence-to-sequence*) : l'entrée est une image de ligne (une séquence de patches visuels), la sortie est une séquence de tokens de texte.

```
Image de ligne (H × W × 3)
        ↓
[ Encodeur BEiT (ViT pré-entraîné par masquage) ]
        ↓
Représentations des patches (n_patches × d_encoder)
        ↓
[ Décodeur GPT-2 (avec cross-attention sur l'encodeur) ]
        ↓
Tokens de texte générés autoregressivement
        ↓
Transcription : « En cel tems que li rois Artus... »
```

### 1.2 L'encodeur : BEiT pré-entraîné

L'encodeur de TrOCR est un **ViT pré-entraîné par BEiT** (*BERT pre-training for Image Transformers*, Bao et al., 2022).

**Rappel du pré-entraînement BEiT** (vu en section 2.2) : environ 40% des patches sont masqués aléatoirement, et le modèle est entraîné à prédire leurs **visual tokens** discrets (produits par un dVAE — *discrete Variational AutoEncoder*). Cette tâche force l'encodeur à apprendre des représentations qui capturent la structure sémantique et spatiale des images.

**Pourquoi BEiT plutôt que MAE pour TrOCR ?**

BEiT prédit des tokens discrets (catégories) — analogue à la prédiction de tokens de texte masqués dans BERT. Cette affinité avec le paradigme textuel facilite l'intégration avec le décodeur de langue. MAE prédit des pixels continus — plus proche de la reconstruction d'image que de la compréhension sémantique.

**Configurations disponibles :**

| Modèle | Encodeur | Paramètres encodeur | Résolution | Patches |
|--------|----------|--------------------:|------------|---------|
| TrOCR-small | BEiT-Small | 22 M | 384 × 384 | 768 |
| TrOCR-base | BEiT-Base | 86 M | 384 × 384 | 768 |
| TrOCR-large | BEiT-Large | 307 M | 384 × 384 | 768 |

La résolution d'entrée est fixée à **384 × 384 pixels** — plus grande que les 224 × 224 de ViT-B standard, ce qui permet de capturer les détails fins des traits d'écriture.

Pour une image de ligne de dimensions variables (typiquement 60 × 800 pixels pour un manuscrit), TrOCR redimensionne l'image à 384 × 384 en conservant le ratio d'aspect par padding. Cette opération préserve la géométrie des lettres mais introduit beaucoup d'espace vide — une limite sur laquelle nous reviendrons.

### 1.3 Le décodeur : GPT-2 autorégressif avec cross-attention

Le décodeur de TrOCR est un **Transformer décodeur** de type GPT-2, modifié pour inclure une couche de **cross-attention** à chaque bloc.

**Architecture d'un bloc décodeur :**

```
Token précédents (y_1, ..., y_{t-1})
        ↓
[ Token embedding + Positional embedding ]
        ↓ (×N_layers)
┌─────────────────────────────────────────┐
│  Masked self-attention                  │  ← Chaque token voit les précédents
│           ↓                             │
│  Cross-attention sur l'encodeur         │  ← Lit les représentations de l'image
│           ↓                             │
│  Feed-Forward Network                   │
│           ↓                             │
│  LayerNorm + Residual (×3)              │
└─────────────────────────────────────────┘
        ↓
[ Tête de classification → vocabulaire ]
        ↓
Distribution sur le vocabulaire → token y_t
```

**La cross-attention est la connexion encodeur-décodeur :**

$$\text{CrossAttn}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V$$

- $Q$ : requêtes issues du décodeur (état du décodeur au token $t$)
- $K, V$ : clés et valeurs issues de l'encodeur (représentations des patches de l'image)

Pour produire le token $y_t$, le décodeur consulte les représentations visuelles de l'encodeur et sélectionne les patches les plus pertinents pour le token courant. C'est un mécanisme d'alignement implicite — analogue à l'attention de Bahdanau, mais dans un cadre plus général.

**Le vocabulaire du décodeur**

TrOCR utilise le vocabulaire BPE (*Byte Pair Encoding*) de GPT-2 : 50 257 tokens en anglais pour les versions originales. Pour l'ancien français, ce vocabulaire est sous-optimal — des graphies médiévales fréquentes (*-oit*, *-ier*, *-ier*) sont découpées en sous-mots rares, ce qui augmente la longueur des séquences cibles et dégrade la qualité de la modélisation de langue.

**Solution :** entraîner un tokenizer BPE sur un corpus de vieux et moyen français (HTR-United, dictionnaire DMF), puis initialiser le décodeur avec ce nouveau vocabulaire. Cela nécessite de réinitialiser la couche d'embedding et la tête de classification, mais conserve tous les autres paramètres du décodeur pré-entraîné.

### 1.4 Génération autorégrressive et beam search

Le décodeur génère la transcription token par token. À chaque étape $t$ :

$$y_t = \argmax_{v \in \mathcal{V}} P(v | y_1, \ldots, y_{t-1}, E_I)$$

où $E_I$ est la sortie de l'encodeur (représentations des patches de l'image).

**Greedy decoding** : choisir le token le plus probable à chaque étape. Rapide mais sous-optimal — une mauvaise décision précoce ne peut pas être corrigée.

**Beam search** : maintenir les $B$ hypothèses les plus probables à chaque étape et les développer simultanément. Plus lent (×B) mais meilleur en qualité.

$$\text{Score}(\mathbf{y}) = \frac{1}{|\mathbf{y}|^\alpha} \sum_{t=1}^{|\mathbf{y}|} \log P(y_t | y_{<t}, E_I)$$

Le facteur de normalisation $|\mathbf{y}|^\alpha$ (avec $\alpha \in [0.6, 0.7]$ typiquement) pénalise les séquences trop courtes, qui tendent à avoir un score cumulé plus élevé que les séquences longues même si elles sont moins précises.

```python
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Charger TrOCR-base-handwritten (point de départ pour le fine-tuning)
MODEL_NAME = "microsoft/trocr-base-handwritten"

processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()

def transcrire_ligne(
    image_ligne: Image.Image,
    num_beams: int = 4,
    max_new_tokens: int = 128,
    length_penalty: float = 0.6
) -> tuple[str, float]:
    """
    Transcrit une image de ligne avec TrOCR.

    Args:
        image_ligne    : image PIL de la ligne (dimensions variables)
        num_beams      : nombre de faisceaux pour le beam search (1 = greedy)
        max_new_tokens : longueur maximale de la transcription en tokens
        length_penalty : pénalité de longueur (< 1 = favorise les séquences courtes)

    Returns:
        transcription : texte transcrit
        confidence    : score de confiance approximatif (log-probabilité normalisée)
    """
    # Prétraitement : redimensionnement à 384×384 avec padding
    pixel_values = processor(
        images=image_ligne.convert("RGB"),
        return_tensors="pt"
    ).pixel_values.to(DEVICE)

    with torch.no_grad():
        generated = model.generate(
            pixel_values,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            length_penalty=length_penalty,
            return_dict_in_generate=True,
            output_scores=True,
        )

    # Décodage du texte
    transcription = processor.batch_decode(
        generated.sequences,
        skip_special_tokens=True
    )[0]

    # Score de confiance approximatif (somme des log-probas normalisée)
    if generated.scores:
        scores = torch.stack(generated.scores, dim=1)  # (1, T, vocab)
        log_probs = torch.log_softmax(scores, dim=-1)
        # Sélectionner les log-probas des tokens générés
        tokens = generated.sequences[:, 1:]   # Sans le token de début
        token_log_probs = log_probs.gather(
            2, tokens.unsqueeze(-1)
        ).squeeze(-1)
        confidence = float(token_log_probs.mean().exp())
    else:
        confidence = 0.0

    return transcription, confidence
```

### 1.5 La fonction de perte : entropie croisée autorégrressive

Pendant l'entraînement, TrOCR utilise le **teacher forcing** : à chaque étape $t$, le décodeur reçoit le token *réel* $y_{t-1}$ (pas sa prédiction), et produit une distribution sur le vocabulaire pour $y_t$.

La loss est l'entropie croisée sur tous les tokens de la séquence cible :

$$\mathcal{L} = -\frac{1}{T} \sum_{t=1}^{T} \log P(y_t | y_1, \ldots, y_{t-1}, E_I; \theta)$$

où $T$ est la longueur de la transcription cible et $\theta$ les paramètres du modèle.

**L'exposure bias** (mentionné dans le glossaire de 2.4) est une limite importante : le modèle entraîné avec des tokens corrects ($y_{t-1}$ réel) ne voit jamais ses propres erreurs pendant l'entraînement, mais les rencontre en inférence (où $y_{t-1}$ est sa propre prédiction précédente). Des techniques comme le **scheduled sampling** (remplacer progressivement les tokens réels par les prédictions du modèle) atténuent ce biais, mais sont rarement utilisées en pratique pour l'HTR.

---

## 2. Comparaison TrOCR vs Kraken

Avant de choisir une stratégie de fine-tuning, il faut comprendre comment TrOCR se positionne par rapport à Kraken — l'autre modèle HTR majeur de notre pipeline.

| Critère | TrOCR-base | Kraken (LSTM-CTC) |
|---------|-----------|-------------------|
| **Architecture** | ViT encoder + GPT-2 decoder | CNN + BiLSTM + CTC |
| **Paramètres** | 334 M | ~7 M |
| **Loss d'entraînement** | Entropie croisée AR | CTC |
| **Contexte** | Global (self + cross-attention) | Bidirectionnel limité (BiLSTM) |
| **Modèle de langue implicite** | Fort (GPT-2) | Faible (BiLSTM) |
| **Vitesse inférence GPU** | ~50 lignes/s | ~200 lignes/s |
| **Vitesse inférence CPU** | ~2 lignes/s | ~20 lignes/s |
| **Fine-tuning (GPU) nécessaire** | Oui (fortement recommandé) | Oui (modéré) |
| **Données fine-tuning** | 500–5 000 lignes | 300–2 000 lignes |
| **Modèles médiévaux disponibles** | Non (à fine-tuner) | Oui (HTR-United) |
| **CER moyen (médical/moderne)** | 2–5% | 5–12% |
| **CER estimé (médiéval, fine-tuné)** | 8–15% | 10–20% |

**Conclusion opérationnelle :** TrOCR offre la meilleure qualité finale sur les textes difficiles, mais demande plus de ressources et de données. Kraken est plus accessible et dispose de modèles pré-entraînés sur des données médiévales francophones — idéal comme baseline et comme modèle complémentaire dans l'ensemble. Les deux modèles sont utilisés en parallèle dans notre pipeline, et leurs prédictions sont combinées par vote pondéré (voir Jour 4).

---

## 3. Préparation des données de fine-tuning

### 3.1 Format des données HTR

Un dataset HTR pour TrOCR est constitué de paires **(image de ligne, transcription)**.

```python
from datasets import Dataset, DatasetDict
from PIL import Image
import pandas as pd
from pathlib import Path

def charger_dataset_htr(
    dossier_lignes: str,
    fichier_transcriptions: str,
    split_ratio: tuple[float, float, float] = (0.80, 0.10, 0.10)
) -> DatasetDict:
    """
    Charge un dataset HTR depuis un dossier d'images de lignes
    et un fichier de transcriptions.

    Args:
        dossier_lignes        : dossier contenant les images de lignes (.png, .jpg)
        fichier_transcriptions : fichier CSV ou TSV avec colonnes 'image' et 'texte'
        split_ratio           : proportion train / validation / test

    Format attendu du CSV :
        image,texte
        ligne_001.png,"En cel tems que li rois"
        ligne_002.png,"Artus regnoit en la grant"
        ...

    Returns:
        DatasetDict avec splits 'train', 'validation', 'test'
    """
    df = pd.read_csv(fichier_transcriptions)
    dossier = Path(dossier_lignes)

    # Vérifier que toutes les images existent
    manquantes = [
        row["image"] for _, row in df.iterrows()
        if not (dossier / row["image"]).exists()
    ]
    if manquantes:
        print(f"  ⚠ {len(manquantes)} images manquantes : {manquantes[:5]}...")
        df = df[~df["image"].isin(manquantes)]

    print(f"  {len(df)} paires image-texte chargées")

    # Mélange aléatoire reproductible
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Split
    n = len(df)
    n_train = int(n * split_ratio[0])
    n_val   = int(n * split_ratio[1])

    splits = {
        "train"     : df.iloc[:n_train],
        "validation": df.iloc[n_train:n_train + n_val],
        "test"      : df.iloc[n_train + n_val:],
    }

    datasets = {}
    for nom, df_split in splits.items():
        datasets[nom] = Dataset.from_dict({
            "image_path" : [str(dossier / p) for p in df_split["image"]],
            "text"       : df_split["texte"].tolist(),
        })
        print(f"  {nom:12s} : {len(datasets[nom])} exemples")

    return DatasetDict(datasets)
```

### 3.2 Prétraitement des images de lignes

Les images de lignes issues de Kraken ont des dimensions variables. TrOCR les redimensionne à 384 × 384 avec padding — mais ce padding peut être important si la ligne est très large (800 px) et peu haute (60 px).

**Stratégie recommandée :** normaliser la hauteur à une valeur fixe (64 ou 128 px) tout en conservant le ratio d'aspect, puis padder horizontalement.

```python
import torchvision.transforms as T
import numpy as np

def normaliser_image_ligne(
    image: Image.Image,
    hauteur_cible: int = 64,
    largeur_max: int = 1024,
    couleur_fond: tuple = (255, 255, 255)
) -> Image.Image:
    """
    Normalise une image de ligne pour TrOCR :
    1. Convertit en RGB
    2. Normalise la hauteur en conservant le ratio d'aspect
    3. Tronque si trop large (> largeur_max)
    4. Redimensionne à 384×384 avec padding blanc pour TrOCR

    La normalisation de hauteur préserve la géométrie des lettres
    mieux que le redimensionnement direct à 384×384.
    """
    image = image.convert("RGB")
    w, h  = image.size

    # Étape 1 : normaliser la hauteur
    ratio      = hauteur_cible / h
    new_width  = min(int(w * ratio), largeur_max)
    image_norm = image.resize((new_width, hauteur_cible), Image.LANCZOS)

    # Étape 2 : créer une image 384×384 avec padding
    canvas = Image.new("RGB", (384, 384), color=couleur_fond)
    # Centrer verticalement
    y_offset = (384 - hauteur_cible) // 2
    canvas.paste(image_norm, (0, y_offset))

    return canvas


def augmenter_image_ligne(image: Image.Image, p: float = 0.5) -> Image.Image:
    """
    Data augmentation pour les lignes de manuscrits.
    Simule les variabilités rencontrées en production.
    """
    augmentations = [
        # Légère rotation (simule l'inclinaison des lignes)
        T.RandomRotation(degrees=2, fill=255),
        # Variations d'éclairage (simule l'hétérogénéité des numérisations)
        T.ColorJitter(brightness=0.3, contrast=0.3),
        # Flou léger (simule la mise au point imparfaite)
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.3),
        # Perspective légère (simule la courbure du parchemin)
        T.RandomPerspective(distortion_scale=0.05, p=0.3, fill=255),
    ]
    for aug in augmentations:
        if np.random.random() < p:
            # Convertir PIL → Tensor → aug → PIL
            t = T.ToTensor()(image)
            t = aug(t)
            image = T.ToPILImage()(t)
    return image
```

### 3.3 Collation et DataLoader

```python
import torch
from torch.utils.data import DataLoader
from transformers import TrOCRProcessor

def creer_collate_fn(
    processor: TrOCRProcessor,
    augmenter: bool = True,
    hauteur_cible: int = 64
):
    """
    Retourne une fonction de collation pour le DataLoader TrOCR.
    Applique le prétraitement et l'augmentation à la volée.
    """
    def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
        images, textes = [], []
        for item in batch:
            img = Image.open(item["image_path"]).convert("RGB")
            img = normaliser_image_ligne(img, hauteur_cible=hauteur_cible)
            if augmenter:
                img = augmenter_image_ligne(img, p=0.5)
            images.append(img)
            textes.append(item["text"])

        # Encodage des images
        pixel_values = processor(
            images=images, return_tensors="pt"
        ).pixel_values   # (B, 3, 384, 384)

        # Encodage des textes cibles (labels)
        labels = processor.tokenizer(
            textes,
            padding="longest",
            max_length=128,
            truncation=True,
            return_tensors="pt"
        ).input_ids   # (B, T_max)

        # Remplacer le padding (token_id = 1) par -100 pour ignorer dans la loss
        labels[labels == processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels"      : labels,
        }

    return collate_fn
```

---

## 4. Stratégies de fine-tuning

### 4.1 Full fine-tuning

La stratégie la plus simple : tous les paramètres du modèle (encodeur + décodeur) sont mis à jour pendant l'entraînement. Offre les meilleures performances finales mais nécessite le plus de mémoire GPU et de données.

**Quand l'utiliser :** corpus abondant (> 5 000 lignes annotées), GPU avec au moins 16 Go de VRAM, temps d'entraînement disponible (plusieurs heures).

```python
from transformers import (
    VisionEncoderDecoderModel, TrOCRProcessor,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    default_data_collator
)
import evaluate

def fine_tuner_trocr_complet(
    dataset_dict,
    model_name: str = "microsoft/trocr-base-handwritten",
    output_dir: str = "trocr-medieval",
    n_epochs: int = 10,
    batch_size: int = 8,
    lr: float = 5e-5,
    warmup_steps: int = 200,
) -> VisionEncoderDecoderModel:
    """
    Full fine-tuning de TrOCR sur un dataset HTR médiéval.
    """
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)

    # Configuration des tokens spéciaux
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id           = processor.tokenizer.pad_token_id
    model.config.eos_token_id           = processor.tokenizer.sep_token_id
    model.config.vocab_size             = model.config.decoder.vocab_size
    model.config.max_length             = 128
    model.config.num_beams              = 4
    model.config.length_penalty         = 0.6

    # Métrique CER
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids   = pred.predictions
        # Remplacer -100 par le token de padding pour le décodage
        labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
        pred_str   = processor.batch_decode(pred_ids,   skip_special_tokens=True)
        label_str  = processor.batch_decode(labels_ids, skip_special_tokens=True)
        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        return {"cer": cer}

    collate_fn = creer_collate_fn(processor, augmenter=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=n_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        predict_with_generate=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,           # CER : minimiser
        learning_rate=lr,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),    # Mixed precision si GPU disponible
        logging_steps=50,
        save_total_limit=3,               # Conserver les 3 meilleurs checkpoints
        report_to="none",                 # Désactiver wandb/tensorboard par défaut
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["validation"],
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"  Modèle sauvegardé dans {output_dir}")
    return model
```

### 4.2 LoRA : Low-Rank Adaptation

**LoRA** (*Low-Rank Adaptation*, Hu et al., 2022) est une technique de fine-tuning efficace qui n'entraîne qu'une fraction des paramètres du modèle original. Pour chaque matrice de poids $W \in \mathbb{R}^{d \times k}$ à adapter, on introduit deux matrices de rang réduit $A \in \mathbb{R}^{d \times r}$ et $B \in \mathbb{R}^{r \times k}$ avec $r \ll \min(d, k)$ :

$$W' = W + \Delta W = W + BA$$

Seules $A$ et $B$ sont entraînées — $W$ reste gelé. Le nombre de paramètres entraînables est :

$$|\theta_{\text{LoRA}}| = 2 \sum_l r_l (d_l + k_l) \ll |\theta_{\text{full}}|$$

**Initialisation :** $B$ est initialisée à zéro (donc $\Delta W = 0$ en début d'entraînement — le modèle pré-entraîné est préservé) ; $A$ est initialisée avec une distribution gaussienne.

**Scaling :** la mise à jour est mise à l'échelle par $\alpha / r$ pour contrôler l'amplitude des modifications :

$$h = Wx + \frac{\alpha}{r} BAx$$

**Paramètres clés :**
- $r$ : rang des matrices de décomposition. $r \in [4, 64]$ typiquement. Un rang plus élevé = plus de paramètres entraînés = adaptation plus forte.
- $\alpha$ : facteur de scaling. Souvent fixé à $r$ (ce qui revient à $\alpha/r = 1$) ou à $2r$.
- Modules ciblés : dans TrOCR, LoRA est typiquement appliqué aux matrices Q, K, V, et O de la cross-attention et de la self-attention.

```python
from peft import get_peft_model, LoraConfig, TaskType

def fine_tuner_trocr_lora(
    dataset_dict,
    model_name: str = "microsoft/trocr-base-handwritten",
    output_dir: str = "trocr-medieval-lora",
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    n_epochs: int = 15,
    batch_size: int = 16,
    lr: float = 1e-4,
) -> VisionEncoderDecoderModel:
    """
    Fine-tuning LoRA de TrOCR.
    Entraîne ~1-3% des paramètres — adaptée aux GPU < 8 Go VRAM.
    """
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)

    # Configuration LoRA
    # On cible les matrices d'attention du décodeur (cross-attention + self-attention)
    # et les matrices de projection de l'encodeur
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=[
            # Cross-attention du décodeur
            "crossattention.k_proj",
            "crossattention.v_proj",
            "crossattention.q_proj",
            "crossattention.out_proj",
            # Self-attention du décodeur
            "attn.k_proj",
            "attn.v_proj",
            "attn.q_proj",
            "attn.out_proj",
        ],
        bias="none",
        inference_mode=False,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # Exemple : trainable params: 2,097,152 || all params: 336,197,632 || (0.62%)

    # Configuration des tokens (identique au full fine-tuning)
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id           = processor.tokenizer.pad_token_id
    model.config.eos_token_id           = processor.tokenizer.sep_token_id
    model.config.max_length             = 128
    model.config.num_beams              = 4

    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids   = pred.predictions
        labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
        pred_str   = processor.batch_decode(pred_ids,   skip_special_tokens=True)
        label_str  = processor.batch_decode(labels_ids, skip_special_tokens=True)
        return {"cer": cer_metric.compute(predictions=pred_str, references=label_str)}

    collate_fn = creer_collate_fn(processor, augmenter=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=n_epochs,
        per_device_train_batch_size=batch_size,   # Plus grand batch : LoRA < mémoire
        per_device_eval_batch_size=batch_size,
        predict_with_generate=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        learning_rate=lr,          # LR plus élevé possible avec LoRA
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        save_total_limit=3,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["validation"],
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Sauvegarder le modèle fusionné (LoRA intégré dans les poids originaux)
    model = model.merge_and_unload()
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    return model
```

**Avantages de LoRA pour notre projet :**
- 10 à 50× moins de mémoire GPU en entraînement (les gradients ne sont calculés que sur A et B).
- Entraînement possible sur GPU 6–8 Go (T4, RTX 3060) — les ressources accessibles aux étudiants.
- Convergence souvent plus rapide (moins de paramètres = moins d'overfitting sur petits datasets).
- Performances proches du full fine-tuning avec suffisamment d'epochs.

### 4.3 Adapter layers

Les **adapters** (Houlsby et al., 2019) sont de petits modules résiduels insérés entre les couches existantes du modèle. Chaque adapter est composé d'une projection descendante ($d \to r$), d'une non-linéarité, et d'une projection montante ($r \to d$) :

$$h' = h + W_{\text{up}} \cdot \text{GELU}(W_{\text{down}} \cdot h)$$

Seuls les paramètres des adapters sont entraînés — le modèle pré-entraîné est gelé. Cela correspond à ~0,5–3% des paramètres selon $r$.

**Comparaison LoRA vs Adapters pour TrOCR**

| Critère | LoRA | Adapters |
|---------|------|---------|
| Position des paramètres | Parallèle aux matrices existantes | Séquentiel (dans le flux) |
| Latence inférence | Nulle (fusionné après entraînement) | Légère (calcul supplémentaire) |
| Mémoire entraînement | Très faible | Faible |
| Implémentation | PEFT (HuggingFace) | PEFT ou custom |
| Performances | Légèrement supérieures | Comparables |

Pour TrOCR, LoRA est généralement préféré aux adapters car il peut être fusionné après entraînement (aucun surcoût à l'inférence) et son intégration dans la bibliothèque PEFT est plus mature.

### 4.4 Gel partiel : encodeur gelé, décodeur fine-tuné

Une stratégie intermédiaire entre le full fine-tuning et LoRA : geler entièrement l'encodeur (ses représentations visuelles sont déjà bonnes grâce à BEiT) et n'entraîner que le décodeur.

```python
def geler_encodeur(model: VisionEncoderDecoderModel) -> None:
    """
    Gèle tous les paramètres de l'encodeur.
    Seul le décodeur est mis à jour pendant l'entraînement.
    """
    for param in model.encoder.parameters():
        param.requires_grad_(False)

    n_encoder = sum(p.numel() for p in model.encoder.parameters())
    n_decoder = sum(p.numel() for p in model.decoder.parameters()
                    if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())

    print(f"  Encodeur gelé   : {n_encoder:,} paramètres")
    print(f"  Décodeur actif  : {n_decoder:,} paramètres")
    print(f"  Ratio actif     : {n_decoder/n_total:.1%}")
```

**Quand utiliser le gel partiel :**
- Dataset très petit (< 500 lignes) : entraîner l'encodeur risque l'overfitting.
- GPU très limité (< 6 Go) : les gradients de l'encodeur sont supprimés.
- Texte proche du manuscrit moderne (l'encodeur BEiT est suffisant sans adaptation).

### 4.5 Choix de la stratégie selon les contraintes

| Ressources | Données | Stratégie recommandée |
|-----------|---------|----------------------|
| GPU ≥ 24 Go | > 5 000 lignes | Full fine-tuning |
| GPU 8–16 Go | 1 000–5 000 lignes | LoRA ($r = 16$) |
| GPU 4–8 Go | 500–2 000 lignes | LoRA ($r = 8$) + encodeur gelé |
| CPU uniquement | Toute taille | Kraken (LSTM-CTC) — TrOCR impraticable |

---

## 5. Évaluation des performances HTR

### 5.1 Métriques : rappel et calcul pratique

```python
import editdistance  # pip install editdistance

def calculer_cer(
    predictions: list[str],
    references: list[str]
) -> dict[str, float]:
    """
    Calcule CER et WER sur un ensemble de paires (prédiction, référence).

    Returns:
        Dict avec 'cer', 'wer', et statistiques détaillées.
    """
    total_chars = total_edits_chars = 0
    total_words = total_edits_words = 0
    cers_par_ligne = []

    for pred, ref in zip(predictions, references):
        # CER
        edit_chars  = editdistance.eval(pred, ref)
        n_chars     = max(len(ref), 1)
        cer_ligne   = edit_chars / n_chars
        cers_par_ligne.append(cer_ligne)
        total_chars       += n_chars
        total_edits_chars += edit_chars

        # WER
        pred_words = pred.split()
        ref_words  = ref.split()
        edit_words = editdistance.eval(pred_words, ref_words)
        total_words       += max(len(ref_words), 1)
        total_edits_words += edit_words

    cer_global = total_edits_chars / max(total_chars, 1)
    wer_global = total_edits_words / max(total_words, 1)

    return {
        "cer"            : cer_global,
        "wer"            : wer_global,
        "cer_median"     : float(np.median(cers_par_ligne)),
        "cer_p90"        : float(np.percentile(cers_par_ligne, 90)),
        "cer_pct_parfait": float(np.mean(np.array(cers_par_ligne) == 0)),
        "n_lignes"       : len(predictions),
    }
```

### 5.2 Analyse des erreurs par type

Une métrique globale masque des patterns d'erreur qui guident les corrections à apporter.

```python
from collections import Counter
import difflib

def analyser_erreurs(
    predictions: list[str],
    references: list[str],
    n_exemples: int = 20
) -> dict:
    """
    Analyse les types d'erreurs HTR les plus fréquents.
    Catégorise en : substitutions, insertions, suppressions.
    """
    substitutions = Counter()
    insertions    = Counter()
    suppressions  = Counter()
    pires_cas     = []

    for pred, ref in zip(predictions, references):
        cer_ligne = editdistance.eval(pred, ref) / max(len(ref), 1)
        if cer_ligne > 0:
            pires_cas.append((cer_ligne, pred, ref))

        # Alignement caractère par caractère (approximatif)
        matcher = difflib.SequenceMatcher(None, ref, pred)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "replace":
                # Substitution : ref[i1:i2] → pred[j1:j2]
                for r, p in zip(ref[i1:i2], pred[j1:j2]):
                    substitutions[(r, p)] += 1
            elif tag == "insert":
                # Insertion : pred[j1:j2] ajouté
                for p in pred[j1:j2]:
                    insertions[p] += 1
            elif tag == "delete":
                # Suppression : ref[i1:i2] manquant
                for r in ref[i1:i2]:
                    suppressions[r] += 1

    # Trier les pires cas par CER décroissant
    pires_cas.sort(reverse=True)

    print("=== Analyse des erreurs ===\n")
    print("Top 10 substitutions (référence → prédiction) :")
    for (r, p), n in substitutions.most_common(10):
        print(f"  '{r}' → '{p}' : {n} fois")

    print("\nTop 10 caractères les plus supprimés :")
    for c, n in suppressions.most_common(10):
        print(f"  '{c}' : {n} fois")

    print(f"\n{n_exemples} pires cas par CER :")
    for cer_val, pred, ref in pires_cas[:n_exemples]:
        print(f"  CER={cer_val:.2f} | Réf : {ref[:60]!r}")
        print(f"           | Préd : {pred[:60]!r}")

    return {
        "top_substitutions": substitutions.most_common(20),
        "top_suppressions" : suppressions.most_common(20),
        "top_insertions"   : insertions.most_common(20),
        "pires_cas"        : pires_cas[:50],
    }
```

### 5.3 Courbes d'apprentissage et early stopping

```python
import matplotlib.pyplot as plt

def tracer_courbes_apprentissage(
    historique_train: list[float],
    historique_val: list[float],
    metrique: str = "CER",
    nom_figure: str = "courbes"
) -> None:
    """
    Trace les courbes d'apprentissage et identifie le meilleur checkpoint.
    """
    epochs = range(1, len(historique_train) + 1)
    best_epoch = int(np.argmin(historique_val)) + 1
    best_val   = min(historique_val)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, historique_train, "b-o", label=f"{metrique} entraînement",
            markersize=4, linewidth=1.5)
    ax.plot(epochs, historique_val, "r-o", label=f"{metrique} validation",
            markersize=4, linewidth=1.5)
    ax.axvline(best_epoch, color="green", linestyle="--", alpha=0.7,
               label=f"Meilleur checkpoint (epoch {best_epoch}, {metrique}={best_val:.3f})")
    ax.set_xlabel("Époque")
    ax.set_ylabel(metrique)
    ax.set_title(f"Courbes d'apprentissage — TrOCR fine-tuné")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"figures/{nom_figure}_learning_curves.png", dpi=150)
    plt.show()

    # Détecter l'overfitting
    if len(historique_val) > 5:
        ecart_final = historique_val[-1] - best_val
        if ecart_final > 0.02:
            print(f"\n  ⚠ Overfitting détecté : la validation s'est dégradée de "
                  f"{ecart_final:.3f} depuis le meilleur checkpoint.")
            print(f"  Recommandation : utiliser le checkpoint de l'époque {best_epoch}.")
```

### 5.4 Évaluation comparative : TrOCR vs Kraken vs baseline

```python
def evaluer_pipeline_complet(
    dataset_test: Dataset,
    modeles: dict[str, callable],
    processor: TrOCRProcessor
) -> dict[str, dict]:
    """
    Évalue et compare plusieurs modèles HTR sur le dataset de test.

    Args:
        dataset_test : dataset HuggingFace avec 'image_path' et 'text'
        modeles      : dict {nom_modele: fonction_transcription(image) → str}
    """
    resultats_globaux = {}

    for nom_modele, fn_transcrire in modeles.items():
        print(f"\n=== Évaluation : {nom_modele} ===")
        predictions, references = [], []

        for item in dataset_test:
            img = Image.open(item["image_path"]).convert("RGB")
            img = normaliser_image_ligne(img)
            try:
                pred, _ = fn_transcrire(img)
            except Exception as e:
                pred = ""
                print(f"  Erreur sur {item['image_path']} : {e}")
            predictions.append(pred)
            references.append(item["text"])

        metriques = calculer_cer(predictions, references)
        resultats_globaux[nom_modele] = {
            "metriques"   : metriques,
            "predictions" : predictions,
            "references"  : references,
        }

        print(f"  CER global  : {metriques['cer']:.3f} ({metriques['cer']:.1%})")
        print(f"  WER global  : {metriques['wer']:.3f} ({metriques['wer']:.1%})")
        print(f"  CER médian  : {metriques['cer_median']:.3f}")
        print(f"  Lignes parfaites : {metriques['cer_pct_parfait']:.1%}")

    # Résumé comparatif
    print("\n=== Résumé comparatif ===")
    print(f"{'Modèle':30s} | {'CER':>8} | {'WER':>8} | {'Parfait':>8}")
    print("-" * 65)
    for nom, res in sorted(
        resultats_globaux.items(),
        key=lambda x: x[1]["metriques"]["cer"]
    ):
        m = res["metriques"]
        print(f"{nom:30s} | {m['cer']:>7.1%} | {m['wer']:>7.1%} "
              f"| {m['cer_pct_parfait']:>7.1%}")

    return resultats_globaux
```

---

## 6. Inférence à grande échelle et optimisation

### 6.1 Inférence par batch

Pour traiter un corpus de milliers de lignes efficacement :

```python
def transcrire_batch(
    images_lignes: list[Image.Image],
    model: VisionEncoderDecoderModel,
    processor: TrOCRProcessor,
    batch_size: int = 32,
    num_beams: int = 4,
    device: str = "cuda"
) -> list[tuple[str, float]]:
    """
    Transcrit une liste d'images de lignes par batches.
    Retourne (transcription, confiance) pour chaque ligne.
    """
    model.eval()
    resultats = []

    for i in range(0, len(images_lignes), batch_size):
        batch = images_lignes[i:i + batch_size]
        batch_normalise = [normaliser_image_ligne(img) for img in batch]

        pixel_values = processor(
            images=batch_normalise, return_tensors="pt"
        ).pixel_values.to(device)

        with torch.no_grad():
            generated = model.generate(
                pixel_values,
                num_beams=num_beams,
                max_new_tokens=128,
                return_dict_in_generate=True,
                output_scores=True,
            )

        transcriptions = processor.batch_decode(
            generated.sequences, skip_special_tokens=True
        )

        # Score de confiance approximatif par séquence
        for j, transcrip in enumerate(transcriptions):
            resultats.append((transcrip, 0.0))   # Score à raffiner si nécessaire

        if (i // batch_size) % 10 == 0:
            print(f"  Traité : {min(i + batch_size, len(images_lignes))}"
                  f"/{len(images_lignes)}")

    return resultats
```

### 6.2 Quantification pour l'inférence CPU

Pour rendre TrOCR praticable sans GPU en production :

```python
import torch.quantization as quant

def quantifier_modele(
    model: VisionEncoderDecoderModel,
    chemin_sortie: str = "trocr-medieval-quantized"
) -> VisionEncoderDecoderModel:
    """
    Applique la quantification dynamique INT8 au décodeur de TrOCR.
    Réduit la taille du modèle de ~50% et accélère l'inférence CPU de 2-4×.
    L'encodeur (convolutions) bénéficie moins de la quantification dynamique.
    """
    model.eval()

    # Quantification dynamique du décodeur (Linear layers → INT8)
    model_quantized = quant.quantize_dynamic(
        model.decoder,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    model.decoder = model_quantized

    torch.save(model.state_dict(), f"{chemin_sortie}/pytorch_model_quantized.bin")
    print(f"  Modèle quantifié sauvegardé dans {chemin_sortie}")
    print(f"  Taille originale  : {sum(p.numel() for p in model.parameters()):,} params")
    return model
```

---

## 7. Connexions avec le pipeline global

TrOCR occupe l'**étape 4** du pipeline — après la segmentation de lignes par Kraken :

```
SAM (layout)
    ↓
Kraken (segmentation de lignes)
    ↓
TrOCR fine-tuné ← Cette section
    +
Kraken HTR (modèle médiéval)
    ↓
Vote pondéré TrOCR + Kraken  ← Jour 4, section 4.2
    ↓
Export JSON + CER estimé
```

**Ce que TrOCR apporte :** le meilleur CER absolu sur les textes difficiles, grâce au modèle de langue implicite du décodeur GPT-2. Particulièrement efficace sur les passages avec abréviations denses et ligatures complexes.

**Ce que TrOCR ne fait pas seul :** sa lenteur relative (vs Kraken) et sa dépendance au GPU le rendent moins adapté aux contraintes de déploiement légères. La combinaison des deux modèles par vote pondéré donne la meilleure robustesse.

---

## Glossaire des termes avancés

**Adapter layers**
Petits modules résiduels insérés entre les couches d'un modèle pré-entraîné. Composés d'une projection descendante, d'une non-linéarité et d'une projection montante. Seuls les adapters sont entraînés. Alternative à LoRA avec légère surcharge à l'inférence.

**BEiT (*BERT pre-training for Image Transformers*)**
Méthode de pré-entraînement du ViT par prédiction de visual tokens masqués. Confère des représentations alignées avec la modélisation de séquences discrètes — d'où l'adéquation avec le décodeur GPT-2 de TrOCR.

**Beam search**
Algorithme de décodage maintenant les $B$ hypothèses les plus probables à chaque étape (au lieu d'une seule en greedy). Améliore la qualité au prix d'un facteur $B$ en temps de calcul. $B \in [4, 8]$ est le standard pour l'HTR.

**Cross-attention**
Dans le décodeur de TrOCR : mécanisme d'attention où les requêtes proviennent du décodeur et les clés/valeurs de l'encodeur. Permet au décodeur de « lire » les représentations visuelles pour chaque token généré.

**dVAE (*discrete Variational AutoEncoder*)**
Autoencodeur variationnel produisant des représentations discrètes (visual tokens). Utilisé dans BEiT comme tokenizer visuel — cible de prédiction pour les patches masqués.

**Exposure bias**
Biais d'entraînement des modèles autoregressifs : pendant l'entraînement (teacher forcing), le décodeur reçoit toujours les tokens corrects ; en inférence, il reçoit ses propres prédictions potentiellement erronées. L'écart entre ces deux régimes dégrade les performances en inférence.

**LoRA (*Low-Rank Adaptation*)**
Technique de fine-tuning paramétrique : approximation des mises à jour de poids $\Delta W = BA$ par des matrices de bas rang ($r \ll d, k$). Réduit de 10 à 50× la mémoire GPU nécessaire. Les matrices A et B peuvent être fusionnées dans W après entraînement (aucune surcharge à l'inférence).

**Mixed precision (FP16/BF16)**
Entraînement utilisant des représentations flottantes 16 bits plutôt que 32 bits pour la majorité des calculs, avec accumulation de gradients en 32 bits. Réduit la mémoire GPU de ~50% et accélère l'entraînement de 1,5 à 3×.

**PEFT (*Parameter-Efficient Fine-Tuning*)**
Bibliothèque HuggingFace implémentant les principales techniques de fine-tuning paramétrique : LoRA, adapters, prefix tuning, prompt tuning. Référence pour les implémentations.

**Quantification dynamique**
Conversion des poids d'un modèle de FP32 vers INT8 après l'entraînement. Réduit la taille du modèle de ~50% et accélère l'inférence CPU de 2 à 4×, avec une légère dégradation des performances. Adapté à TrOCR pour le déploiement sans GPU.

**Rank ($r$) dans LoRA**
Dimension des matrices de décomposition A et B. $r$ faible (4–8) = peu de paramètres, adaptation légère, risque de sous-adaptation. $r$ élevé (32–64) = plus de paramètres, adaptation forte, proche du full fine-tuning. Typiquement $r \in [8, 32]$ pour l'HTR.

**Scheduled sampling**
Technique d'entraînement pour atténuer l'exposure bias : remplacer progressivement les tokens de référence (teacher forcing) par les propres prédictions du modèle, avec une probabilité croissante au fil de l'entraînement. Rarement utilisé en pratique pour l'HTR.

**Teacher forcing**
Stratégie d'entraînement des modèles autoregressifs : à chaque étape $t$, fournir au décodeur le token *réel* $y_{t-1}$ (pas sa prédiction) comme contexte. Accélère la convergence mais introduit l'exposure bias.

**TrOCR (*Transformer-based OCR*)**
Modèle HTR développé par Microsoft Research (Li et al., 2021). Architecture encodeur-décodeur : BEiT pré-entraîné (encodeur image) + GPT-2 (décodeur de texte avec cross-attention). Disponible via HuggingFace Transformers.

**Warmup (échauffement du taux d'apprentissage)**
Phase initiale d'entraînement où le taux d'apprentissage augmente linéairement depuis 0 jusqu'à sa valeur cible, sur quelques centaines de steps. Stabilise l'entraînement en évitant les mises à jour trop grandes au début, quand le modèle n'est pas encore adapté au nouveau domaine.

---

## Bibliographie de référence

### TrOCR et architectures HTR Transformer

- **Li, M., Lyu, T., Yao, T., Ye, J., Lu, T., Zheng, Y., … Huang, F.** (2021). *TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models*. AAAI 2023. [arXiv:2109.10282] — **L'article fondateur. Sections 3 (architecture), 4 (pré-entraînement) et 5 (évaluation) à lire impérativement.**

- **Bao, H., Dong, L., Piao, S., Wei, F.** (2022). *BEiT: BERT Pre-Training of Image Transformers*. ICLR 2022. [arXiv:2106.08254] — L'encodeur de TrOCR.

- **Radford, A. et al.** (2019). *Language Models are Unsupervised Multitask Learners (GPT-2)*. OpenAI Blog. — Le décodeur de TrOCR.

### Fine-tuning paramétrique

- **Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., … Chen, W.** (2022). *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR 2022. [arXiv:2106.09685] — **L'article fondateur de LoRA. À lire pour comprendre la justification théorique du rang faible.**

- **Houlsby, N. et al.** (2019). *Parameter-Efficient Transfer Learning for NLP*. ICML 2019. [arXiv:1902.00751] — Adapter layers : la méthode de référence avant LoRA.

- **Ding, N. et al.** (2023). *Parameter-Efficient Fine-Tuning of Large-Scale Pre-Trained Language Models*. Nature Machine Intelligence, 5, 220–235. — Revue complète des méthodes PEFT, avec comparaisons empiriques.

### Évaluation HTR et métriques

- **Romero, V., Toselli, A. H., Vidal, E.** (2012). *Multimodal Interactive Handwritten Text Transcription*. World Scientific. — Définition rigoureuse du CER et WER pour l'HTR.

- **Kahle, P., Colutto, S., Hackl, G., Mühlberger, G.** (2017). *Transkribus — A Service Platform for Transcription, Recognition and Retrieval of Historical Documents*. ICDAR 2017. — Référence sur les métriques d'évaluation HTR dans les projets patrimoniaux.

### HTR sur manuscrits médiévaux

- **Kiessling, B., Stökl Ben Ezra, D., Miller, M. T.** (2019). *BADAM: A Public Dataset for Baseline Detection in Arabic-Script Manuscripts*. GREC 2019. — Développeurs de Kraken. Contexte comparatif.

- **Pinche, A., Camps, J.-B., Clérice, T.** (2022). *Evaluating Data Augmentation for Historical Handwritten Text Recognition*. ICDAR 2022. — Stratégies d'augmentation pour l'HTR médiéval. Directement applicable à la section 3.2 de ce cours.

- **Constum, T. et al.** (2022). *HTR-United: Mutualisons la vérité de terrain !* DHd 2022. [hal-03693079] — HTR-United, la source de données annotées pour notre fine-tuning.

### Beam search et décodage

- **Graves, A., Fernández, S., Gomez, F., Schmidhuber, J.** (2006). *Connectionist Temporal Classification*. ICML 2006. — CTC, alternative au décodeur autorégressif pour la comparaison avec Kraken.

- **Wiseman, S., Rush, A. M.** (2016). *Sequence-to-Sequence Learning as Beam-Search Optimization*. EMNLP 2016. — Analyse des limites du beam search et alternatives.

---

*Support de cours rédigé pour le module Computer Vision · Promotion MD5 · Mai 2026*
*Ce document accompagne la séance 3.4 du Jour 3 et conclut le Module 3.*
*Il prépare directement le Jour 4 (pipeline end-to-end et agrégation des transcriptions).*
*Durée estimée de lecture : 80–90 minutes.*
