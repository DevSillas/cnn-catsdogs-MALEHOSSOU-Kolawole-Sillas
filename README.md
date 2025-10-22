# CNN from Scratch vs Transfer Learning (Cats vs Dogs)

## Auteur
**Nom :** MALEHOSSOU Kolawolé Sillas Shallum
**Master 1 Intelligence Artificielle**

---

## Objectif du projet
Ce projet a pour but de comparer deux approches de Deep Learning appliquées à la classification d’images du dataset Cats vs Dogs :

1. **Expérience A — CNN “from scratch”**
   - Réseau convolutionnel construit manuellement avec 3 blocs conv → BatchNorm → Dropout.
   - Entraîné depuis zéro sur le dataset.

2. **Expérience B — Transfert Learning (ResNet18)**
   - Réseau pré-entraîné sur ImageNet.
   - Couches finales adaptées pour 2 classes (cat/dog).
   - Objectif : démontrer la rapidité de convergence et la robustesse du transfert learning.

L’enjeu est d’observer l’impact du transfert learning sur la performance, la vitesse de convergence et la généralisation du modèle.

---

## Environnement et installation

### Option 1 — Utilisation sur **Google Colab**
Le projet est conçu pour fonctionner entièrement sur Google Colab (GPU activé).
Aucune installation supplémentaire n’est nécessaire.

### Option 2 — Utilisation locale
Clonez le projet et installez les dépendances :

```bash
git clone https://github.com/<ton_nom_utilisateur>/cnn-catsdogs-MALEHOSSOU-Kolawole-Sillas.git
cd cnn-catsdogs-MALEHOSSOU-Kolawole-Sillas
pip install -r requirements.txt

 Environnement Python minimal

Python ≥ 3.10

PyTorch ≥ 2.0

torchvision ≥ 0.15

matplotlib, numpy, pandas, seaborn, scikit-learn
```


## Organisation du projet
 ```bash
cnn-catsdogs-MALEHOSSOU-Kolawole-Sillas/
│
├─ notebook.ipynb                → Notebook principal du TP
├─ README.md                     → Présent fichier
├─ requirements.txt              → Librairies nécessaires
├─ .gitignore                    → Fichiers exclus du push
└─ checkpoints/                  → Sauvegarde des modèles (non poussés sur GitHub)
```

## Jeu de données

Le dataset Cats vs Dogs provient de Kaggle :
🔗 https://www.kaggle.com/datasets/salader/dogs-vs-cats

Structure attendue :
```bash
Cat_Dog_data/
 ├─ train/
 │   ├─ cats/
 │   └─ dogs/
 └─ test/
     ├─ cats/
     └─ dogs/
```


 Les données ne doivent pas être poussées sur GitHub.
Elles doivent être placées localement dans /content/drive/MyDrive/Exam_deep/Cat_Dog_data/.

## Commandes d’entraînement
 ```bash
 CNN from scratch
python train.py --model scratch --epochs 10 --batch-size 32 --optimizer adam --dropout 0.5

Transfer Learning (ResNet18)
python train.py --model resnet18 --epochs 8 --batch-size 32 --optimizer adam --lr 1e-4 --freeze True
```

Les modèles sauvegardés localement :
checkpoints/best_model_scratch.pth
checkpoints/best_model_transfer.pth

## Résultats expérimentaux
Modèle	Optimiseur	Accuracy	Precision	Recall	F1-score
CNN (from scratch)	Adam	0.53	0.57	0.55	0.56
CNN (SGD + StepLR)	SGD	0.51	0.54	0.52	0.53
ResNet18 (Transfer Learning)	Adam	0.89	0.90	0.88	0.89



## Analyse et interprétation

Le CNN from scratch converge lentement et plafonne autour de 50–55 % d’accuracy.

Le ResNet18 atteint rapidement près de 90 % d’accuracy grâce au transfert des poids pré-entraînés.

Le transfert learning améliore nettement la vitesse de convergence et la capacité de généralisation.

L’ajout de BatchNorm et Dropout stabilise l’entraînement du modèle “from scratch”.


## Conclusions

Le transfert learning est clairement supérieur pour un dataset limité.

Le CNN simple reste essentiel pour la compréhension des bases du Deep Learning.

ResNet18 + Adam + Scheduler offre le meilleur compromis entre rapidité et performance.


## Améliorations possibles

Tester d’autres architectures : MobileNet, EfficientNet, DenseNet.

Étendre la data augmentation pour réduire l’overfitting.

Ajouter TensorBoard ou W&B pour le suivi des métriques.

Débloquer progressivement certaines couches du modèle pré-entraîné (fine-tuning).


## Reproductibilité et sauvegarde

Tous les modèles sont sauvegardés localement (.pth), non poussés sur GitHub.

Un seed fixe assure la reproductibilité des résultats.

Le notebook vérifie automatiquement la présence du GPU.

Les métriques (Loss, Accuracy, Precision, Recall) sont tracées à chaque époque.


## Fichiers exclus du dépôt

Le fichier .gitignore contient :
```bash

data/
*.pt
*.pth
__pycache__/
runs/
checkpoints/
.ipynb_checkpoints/
```


MALEHOSSOU Kolawolé Sillas Schallum

Étudiant en Master 1 Intelligence Artificielle

Email : sillfreelance@gmail.com

Projet réalisé dans le cadre du TP : “Deep Learning — CNN vs Transfer Learning”,

Université: Dakar Institut of Technology, Octobre 2025.


---
