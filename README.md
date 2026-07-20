# -Cat-Breed-Classifie_Oliva-Citterio

Soluzione alla challenge di classificazione per predire la razza di gatti domestici a partire da caratteristiche fisiche e comportamentali.

## Modello scelto

GradientBoostingClassifier (scikit-learn) con 100 alberi e max_depth=4.
Scelto perché funziona molto bene su dati tabulari con feature miste

## Risultati

- Accuracy validation set: 95.26%
- Accuracy cross-validation (5 fold): 96.63% 

Il dataset contiene un record con razza "Alien" che abbasa l'accuracy
Il modello lo classifica come outlier senza allucinare razze inesistenti


## Librerie utilizzate

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
