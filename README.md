# -Cat-Breed-Classifie_Oliva-Citterio

Soluzione alla challenge di classificazione per predire la razza di gatti domestici a partire da caratteristiche fisiche e comportamentali.

## Modello scelto

GradientBoostingClassifier (scikit-learn) con 100 alberi e max_depth=4.
Scelto perché funziona molto bene su dati tabulari con feature miste

## Risultati

- Accuracy validation set: 95.26%
- Accuracy cross-validation (5 fold): 96.63% 

Il dataset contiene un record con razza "Alien" (1 solo esemplare).
Il modello lo classifica correttamente come outlier senza allucinare razze inesistenti.

## Struttura repository

├── main.py # codice completo commentato
├── cats_dataset.csv # training set
├── test_set.csv # test set
├── predictions.csv # predizioni finali
└── grafici/
├── razze.png # distribuzione razze nel training set
├── correlazione.png # heatmap correlazione features numeriche
├── confusione.png # matrice di confusione sul validation set
├── importanza.png # feature importance del modello
└── crossval.png # accuracy per fold della cross-validation


## Librerie utilizzate

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
