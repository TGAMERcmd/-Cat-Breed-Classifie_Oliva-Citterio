"""
pandas per caricare i dati nelle tabelle
numpy per calcoli e ordinamento delle liste
matplot per fare grafici
seaborn per fare grafici
sklearn per allenamento del modello
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# costante per migliorare leggibilità
new_line = "\n"

train = pd.read_csv("cats_dataset.csv", dtype=str)
test = pd.read_csv("test_set.csv", dtype=str)

# per caricare i dati di training e di test
train.shape
test.shape

# pulizia del dataset (i dati che non contengono una razza)
train = train.dropna(subset=["razza"])
train = train[train["razza"].str.lower() != "nan"]

n = len(train)

# DataFrame, una struttura bidimensionale simile ad una tabella
# temporaneamente uniamo train e test, in modo che la stessa stringa verrà 
# convertita nello stesso numero sia nel train che nel test
df = pd.concat([train, test], ignore_index=True)

# pulizia dei dati - alcuni pesi hanno la virgola, altri i punti. Così li rendiamo tutti uguali
df["peso_kg"] = df["peso_kg"].str.replace(",", ".", regex=False)
df["peso_kg"] = pd.to_numeric(df["peso_kg"], errors="coerce").fillna(0)
df["eta_anni"] = pd.to_numeric(df["eta_anni"], errors="coerce").fillna(0)

# definisce le colonne del DataFrame, e le converte in numeri (.cat.codes)
cols = ["sesso", "lunghezza_pelo", "colore_mantello", "livello_attivita",
        "frequenza_miagolio", "sterilizzato", "patologia"]

for c in cols:
    df[c] = df[c].fillna("vuoto").astype("category").cat.codes
df["razza_num"] = df["razza"].fillna("sconosciuta").astype("category").cat.codes

# dizionario con numero razza --> nome razza
razze_map = dict(enumerate(df["razza"].fillna("sconosciuta").astype("category").cat.categories))

"""
Noi utilizziamo una tecnica particolare. Praticamente il valore x rappresenta tutti i parametri da considerare, mentre
y rappresenta il nostro risultato, o predizione. All'inizio il modello studia i parametri e come trasformano y, 
poi togliamo y e osserviamo come il modello effettua le predizioni sul suo valore
"""

# creiamo una copia del DataFrame di allenamento e di test
train_df = df.iloc[:n].copy()
test_df = df.iloc[n:].copy()

feat = ["eta_anni", "peso_kg"] + cols

X = train_df[feat].values
y = train_df["razza_num"].values
X_test_final = test_df[feat].values

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
clf.fit(X_tr, y_tr)

y_pred_val = clf.predict(X_val)

acc = accuracy_score(y_val, y_pred_val)
print(f"{new_line}accuracy validation: {acc * 100:.2f}%")

cv_scores = cross_val_score(clf, X, y, cv=5)
print(f"cross-val media: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")
print(f"fold scores: {[round(s*100,1) for s in cv_scores]}")

classi_val = sorted(set(y_val))
nomi_val = [razze_map[c] for c in classi_val]
print(new_line, classification_report(y_val, y_pred_val, labels=classi_val, target_names=nomi_val), sep="")

modello_finale = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
modello_finale.fit(X, y)

preds = modello_finale.predict(X_test_final)
pred_nomi = [razze_map[p] for p in preds]

out = pd.DataFrame({
    "ID": test_df["ID"],
    "razza_prevista": pred_nomi
})
out.to_csv("predictions.csv", index=False)
print(f"{new_line}predictions.csv salvato")


# grafici 

# grafico delle razze
plt.figure(figsize=(8, 4))
train_df["razza"].value_counts().plot(kind="bar", color="skyblue", edgecolor="black")
plt.title("razze nel dataset")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("grafici/razze.png")
plt.close()

#! duplicato (vedi linea 126): confronto età-peso
plt.figure(figsize=(5, 4))
sns.heatmap(train_df[["eta_anni", "peso_kg"]].corr(), annot=True, cmap="coolwarm")
plt.title("eta vs peso")
plt.tight_layout()
plt.savefig("grafici/correlazione.png")
plt.close()

# matrice della confusione
mat = confusion_matrix(y_val, y_pred_val, labels=classi_val)
plt.figure(figsize=(7, 5))
sns.heatmap(mat, annot=True, fmt="d", cmap="Blues",
            xticklabels=nomi_val, yticklabels=nomi_val)
plt.title("matrice confusione")
plt.ylabel("reale")
plt.xlabel("predetto")
plt.tight_layout()
plt.savefig("grafici/confusione.png")
plt.close()

#! duplicato (vedi linea 106): confronto peso-età 
plt.figure(figsize=(8, 5))
sns.scatterplot(data=train_df, x="eta_anni", y="peso_kg", hue="razza", alpha=0.7)
plt.title("peso vs eta")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig("grafici/scatter.png")
plt.close()

# importanza dei parametri
plt.figure(figsize=(8, 4))
imp = modello_finale.feature_importances_
idx = np.argsort(imp)
plt.barh([feat[i] for i in idx], imp[idx], color="teal")
plt.title("feature importance")
plt.tight_layout()
plt.savefig("grafici/importanza.png")
plt.close()

# accuratezza
plt.figure(figsize=(6, 3))
plt.bar(range(1, 6), cv_scores * 100, color="steelblue", edgecolor="black")
plt.axhline(cv_scores.mean() * 100, color="red", linestyle="--", label=f"media: {cv_scores.mean()*100:.1f}%")
plt.title("accuracy per fold")
plt.xlabel("fold")
plt.ylabel("accuracy %")
plt.legend()
plt.tight_layout()
plt.savefig("grafici/crossval.png")
plt.close()

print("grafici fatti")
