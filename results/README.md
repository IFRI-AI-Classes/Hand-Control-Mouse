# Résultats : règles contre modèles appris

Mesures du 22 novembre 2026, sur les vidéos de 2 personnes (Freud, Prince) et les photos d'une 3e (Patrick).

**En bref.** Sur des personnes déjà vues à l'entraînement, les modèles appris reconnaissent mieux les postures que les règles (F1 de 0,93 à 0,98, contre 0,85). Sur une personne jamais vue, c'est l'inverse : les règles tiennent (0,77 à 0,84) et les modèles s'effondrent (0,17 à 0,46). Avec seulement deux personnes, un modèle appris ne généralise pas à une main qu'il ne connaît pas. Les règles, écrites à la main, n'ont pas ce problème.

## 1. Données

| Personne | Source | Images avec une main | Taux de détection |
|---|---|---|---|
| Freud | 10 vidéos webcam, environ 15 images/s | 1 101 | 89 à 100 % selon la vidéo |
| Prince | 7 vidéos webcam (une par geste), environ 17 images/s | 886 | 98 à 100 % |
| Patrick | 34 photos | **5** | **15 %** |

Images par geste (vidéos) :

| | deplacement | scroll | clic_gauche | clic_droit | drag | neutre |
|---|---|---|---|---|---|---|
| Freud | 169 | 238 | 203 | 202 | 193 | 96 |
| Prince | 129 | 114 | 234 | 110 | 157 | 142 |

Choix faits sur les données :
- **6 classes.** Le double-clic n'est plus une posture dans l'appli (deux clics gauches rapprochés). Les vidéos et photos `double_clic` (pouce + auriculaire) sont réétiquetées en `clic_gauche`, puisque les règles ignorent l'auriculaire pour ce geste.
- **Le drag a deux variantes** : majeur seul, ou signe OK. Freud a filmé les deux. Le drag de Prince est mal formé (doigts à peine repliés), ce qui pèse sur les résultats « une personne de côté ».
- 0,5 s retirée au début et à la fin de chaque vidéo pour enlever les transitions. Images retournées en miroir, comme dans l'appli.

## 2. Méthode

**Entrée des modèles** : les 21 points de la main (x, y, z), centrés sur le poignet et mis à l'échelle par la distance poignet → base du majeur.

**Méthodes comparées** :
- **règles** : la décision de posture de `src/gestures.py` (version avec angles des doigts et pouce mesuré par rapport à l'index) ;
- **régression logistique**, **k plus proches voisins** (k = 5), **random forest** (200 arbres), **MLP** (64 puis 32 neurones), tous avec scikit-learn.

**Augmentation (en option)** : chaque image d'entraînement est ajoutée en miroir (main gauche) et avec deux rotations aléatoires de ±15° autour du poignet. Jamais appliquée au test. Idée reprise du module de robustesse de l'équipe (section 7).

**Trois évaluations** :
1. **Découpage dans le temps (résultat principal).** Dans chaque vidéo, les premiers 70 % des images servent à l'entraînement, les 30 % suivants au test (1 385 images d'entraînement, 602 de test). Les deux personnes sont vues à l'entraînement, mais jamais les mêmes instants. Un mélange au hasard donnerait un score gonflé (environ 0,98), parce que deux images voisines d'une vidéo sont presque identiques.
2. **Une personne de côté.** On apprend sur une personne et on teste sur l'autre. C'est le cas d'un nouvel utilisateur.
3. **Photos de Patrick.** On apprend sur toutes les vidéos et on teste sur les photos où une main est détectée.

**Mesure** : F1 macro, la moyenne du F1 des 6 gestes. Un geste mal reconnu fait baisser le score autant qu'un geste fréquent.

**Attention, les règles sont jugées image par image.** On mesure seulement si la posture est bien reconnue sur chaque image. Dans l'appli, les règles s'appuient en plus sur une confirmation sur plusieurs images (7 pour un clic), une hystérésis et un délai après une main ouverte. Le comportement réel peut donc être meilleur que ces chiffres.

## 3. Résultats

| Méthode | Temps (F1) | Test sur Freud (F1) | Test sur Prince (F1) | Photos Patrick |
|---|---|---|---|---|
| **Règles** | 0,848 | **0,842** | **0,768** | **4/5** |
| Régression logistique | 0,931 | 0,318 | 0,447 | 1/5 |
| Régression logistique + aug. | 0,937 | 0,300 | 0,542 | 0/5 |
| k plus proches voisins | 0,961 | 0,248 | 0,454 | 1/5 |
| k plus proches voisins + aug. | 0,975 | 0,325 | 0,458 | 0/5 |
| **Random forest** | **0,979** | 0,388 | 0,456 | 0/5 |
| Random forest + aug. | 0,971 | 0,462 | 0,517 | 0/5 |
| MLP | 0,945 | 0,167 | 0,434 | 0/5 |
| MLP + aug. | 0,978 | 0,306 | 0,487 | 0/5 |

« Test sur Freud » = entraîné sur Prince seul, testé sur Freud. Et inversement. « + aug. » = entraîné avec augmentation (section 3.2).

Détail par geste : `rapports_par_classe.md`. Matrices de confusion (découpage dans le temps) : `confusion_regles.png`, `confusion_random_forest.png`, `confusion_mlp.png`.

### 3.1 Où se trompent les règles (découpage dans le temps)

| Vrai geste | Bien reconnu | Principale erreur |
|---|---|---|
| deplacement | 90/90 | aucune |
| scroll | 106/107 | aucune notable |
| clic_gauche | 119/132 | 13 pris pour neutre |
| **clic_droit** | **51/94** | **40 pris pour clic_gauche** |
| drag | 83/107 | 22 pris pour scroll |
| neutre | 64/72 | 5 pris pour clic_droit |

Le **clic droit** est le point faible : près d'une fois sur deux, il est lu comme un clic gauche. Le pouce est probablement encore jugé tendu alors qu'il est replié. Le drag est parfois lu comme un scroll, quand l'index ne paraît pas assez replié.

Le MLP, sur le même test, reconnaît 74/94 clics droits et 104/107 drags. C'est sur ces deux gestes que l'apprentissage apporte le plus.

### 3.2 Ce qu'apporte l'augmentation

Sur une nouvelle personne, l'augmentation aide un peu les modèles appris, de 0,05 à 0,14 de F1 selon les cas (MLP : 0,17 → 0,31 et 0,43 → 0,49 ; random forest : 0,39 → 0,46 et 0,46 → 0,52). C'est trop peu pour rattraper les règles (0,77 à 0,84). Le manque vient du nombre de personnes, pas de la façon d'entraîner.

### 3.3 Détection et précision de bout en bout

Une main que MediaPipe ne trouve pas est une erreur pour l'utilisateur, même si les règles auraient reconnu le geste. Ces tableaux comptent donc toutes les images, détectées ou non (idée reprise du module de robustesse de l'équipe, voir section 7). Sortie de `python -m mlp.detection`.

Par personne :

| Personne | Images | Main détectée | Règles justes (mains détectées) | Règles justes (bout en bout) |
|---|---|---|---|---|
| Freud (vidéos) | 1 138 | 96,7 % | 85,7 % | 83,0 % |
| Prince (vidéos) | 888 | 99,8 % | 83,5 % | 83,3 % |
| Patrick (photos) | 34 | **14,7 %** | 80,0 % (4/5) | **11,8 %** |

Par geste (vidéos) :

| Geste | Main détectée | Règles justes (mains détectées) | Règles justes (bout en bout) |
|---|---|---|---|
| deplacement | 100 % | 97,7 % | 97,7 % |
| scroll | 97,8 % | 99,7 % | 97,5 % |
| neutre | 99,2 % | 95,8 % | 95,0 % |
| drag | 96,4 % | 84,9 % | 81,8 % |
| clic_gauche | 98,4 % | 81,9 % | 80,6 % |
| **clic_droit** | 97,2 % | **51,0 %** | **49,5 %** |

Sur les vidéos, la détection n'est presque jamais le problème (plus de 96 %) : les erreurs viennent des règles, surtout sur le clic droit. Sur les photos de Patrick, c'est l'inverse : la détection seule fait tomber la précision de bout en bout à 12 %.

## 4. Lecture

1. **Pour une personne déjà vue, apprendre marche mieux que les règles.** Tous les modèles dépassent les règles, et la random forest atteint 0,98. L'apprentissage corrige justement les gestes où les règles échouent (clic droit, drag).
2. **Pour une nouvelle personne, les règles gagnent nettement.** Entraîné sur une seule main, un modèle apprend la façon de faire de cette personne, pas le geste. Le drag mal formé de Prince le montre bien : il n'est jamais reconnu par un modèle entraîné sur Freud. Les règles, qui décrivent le geste en termes de doigts tendus ou repliés, ne dépendent pas de la personne.
3. **Le détecteur de main lui-même peut lâcher.** Sur les photos de Patrick, MediaPipe ne trouve une main que sur 5 photos sur 34 : presque uniquement la main ouverte. Baisser le seuil de confiance (9/34 au mieux, à 0,1) ou corriger le contraste (CLAHE, gamma) n'aide pas. Hypothèses, non vérifiées : poing fermé et dos de la main face à la caméra, main coupée par le bas de l'image, contre-jour (mur clair derrière). Sur les vidéos de Freud et Prince, la détection est au contraire de 89 à 100 %. **Avant même la reconnaissance du geste, la façon de se placer devant la caméra compte.**

## 5. Limites

- **Deux personnes seulement**, et une seule vidéo par geste pour Prince. Les chiffres « une personne de côté » reposent sur un seul essai chacun.
- **Le découpage dans le temps reste optimiste** : entraînement et test viennent de la même séance, avec la même lumière et la même position.
- **5 photos exploitables** pour Patrick : c'est une indication, pas une mesure.
- **Règles jugées image par image**, sans la logique temporelle de l'appli (voir section 2).
- Vidéos à environ 15 images/s, plus bas que les 30 prévus.

## 6. Suite

- Plus de personnes : c'est la seule façon de savoir si un modèle appris peut battre les règles sur un nouvel utilisateur.
- Combiner les deux approches : garder les règles et utiliser le modèle seulement là où elles échouent (clic droit, drag).
- Tester la détection en conditions difficiles (contre-jour, poing fermé, main près du bord), puisque c'est là que tout commence.

## 7. À propos du module de robustesse de l'équipe

Un module d'évaluation écrit par l'équipe (`robutness/`, retiré du dépôt pour éviter deux pipelines en parallèle) n'a pas été utilisé tel quel, pour trois raisons : il réécrit les règles de la v1 (distance au poignet, double-clic en posture) au lieu d'appeler `src/gestures.py`, il garde 7 classes, et ses fichiers ne sont pas à l'emplacement attendu par ses imports. Ses deux apports les plus utiles sont repris ici : l'augmentation (`mlp/augment.py`) et la mesure de la détection et de la précision de bout en bout (`mlp/detection.py`).

## 8. Reproduire

Depuis la racine du dépôt :

```
python src/extract_landmarks.py --input data/videos --merge
python -m mlp.extract_photos
python -m mlp.train
python -m mlp.detection
python -m mlp.export
```

Le modèle exporté dans `models/mlp.joblib` est le MLP entraîné sur toutes les vidéos. Le choix du modèle à brancher dans l'appli reste à faire.
