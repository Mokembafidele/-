# -
Fidèle Mokemba Mabate

#𝗺𝗼𝗱𝗲̀𝗹𝗲 𝗱𝗲 𝗖𝗿𝗲𝗱𝗶𝘁 𝗦𝗰𝗼𝗿𝗶𝗻𝗴 à l’aide de la 𝗿𝗲́𝗴𝗿𝗲𝘀𝘀𝗶𝗼𝗻 𝗹𝗼𝗴𝗶𝘀𝘁𝗶𝗾𝘂𝗲 en utilisant Python sous Visual Studio Code.


import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, roc_auc_score,
    accuracy_score, precision_score, recall_score
)
import matplotlib.pyplot as plt
import joblib
import os

# utilitaire d'affichage fourni par l'environnement
try:
    from caas_jupyter_tools import display_dataframe_to_user
except Exception as e:
    display_dataframe_to_user = None

# --- 1) Jeu de données synthétique "plausible" ---
np.random.seed(42)
n = 5000

# variables socio-démographiques et financières simulées
age = np.random.randint(21, 71, size=n)  # âge en années
income = np.round(np.random.normal(45000, 20000, size=n)).astype(int)  # revenu annuel
income = np.clip(income, 5000, 300000)
loan_amount = np.round(np.random.normal(15000, 10000, size=n)).astype(int)
loan_amount = np.clip(loan_amount, 500, 200000)
loan_term_months = np.random.choice([12, 24, 36, 48, 60], size=n, p=[0.1,0.2,0.4,0.2,0.1])
credit_history_score = np.round(np.random.normal(650, 70, size=n)).astype(int)  # 300-850 style
credit_history_score = np.clip(credit_history_score, 300, 850)
employment_years = np.round(np.abs(np.random.normal(5, 4, size=n)),1)  # années d'emploi

# combinaaison des variables pour créer une probabilité de défaut (target)
# plus élevé si : loan_amount élevé par rapport au revenu, faible credit_history_score, jeune, faible emploi
risk_raw = (
    0.6 * (loan_amount / (income + 1)) + 
    0.003 * (700 - credit_history_score) + 
    0.01 * (35 - (age - 21)) +
    0.05 * (2 - np.minimum(employment_years, 10))
)
# transformer en probabilités via logistic
prob_default = 1 / (1 + np.exp(- ( (risk_raw - risk_raw.mean())*4 )))
# tirage binaire
y = np.random.binomial(1, prob_default)

df = pd.DataFrame({
    "age": age,
    "income": income,
    "loan_amount": loan_amount,
    "loan_term_months": loan_term_months,
    "credit_history_score": credit_history_score,
    "employment_years": employment_years,
    "default": y
})

# afficher un aperçu
print("Aperçu des 5 premières lignes du dataset synthétique :")
display(df.head())

# Afficher la distribution de la target
print("\nDistribution de la variable cible (default):")
print(df['default'].value_counts(normalize=True).rename('proportion'))

# --- 2) Préparation des données pour le modèle ---
X = df.drop(columns=['default'])
y = df['default']

# One-hot pour loan_term_months (catégorie), standardisation pour features numériques
X = pd.get_dummies(X, columns=['loan_term_months'], prefix='term')
num_cols = ['age','income','loan_amount','credit_history_score','employment_years']

scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[num_cols] = scaler.fit_transform(X[num_cols])

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42, stratify=y)

# --- 3) Entraînement d'une régression logistique ---
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Prédictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:,1]

# --- 4) Évaluation ---
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print("\n--- Metrics ---")
print(f"Accuracy : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall (Sensibilité) : {rec:.4f}")
print(f"AUC ROC : {auc:.4f}")

print("\nClassification report :")
print(classification_report(y_test, y_pred, digits=4))

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
print("Matrice de confusion (format [[TN, FP],[FN, TP]]):")
print(cm)

# --- 5) Tableaux de coefficients (interprétation) ---
coef = pd.Series(model.coef_[0], index=X_train.columns).sort_values()
coef_table = pd.DataFrame({
    "feature": coef.index,
    "coefficient": coef.values,
    "odds_ratio": np.exp(coef.values)
}).reset_index(drop=True)

print("\nTop variables (coefficients) - signification: coefficient positif => augmente la probabilité de défaut")
display(coef_table.tail(10))
display(coef_table.head(10))

# --- 6) Graphiques ---
os.makedirs('/mnt/data/credit_scoring_outputs', exist_ok=True)

# 6.1 ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.figure(figsize=(7,5))
plt.plot(fpr, tpr)
plt.plot([0,1],[0,1], linestyle='--')
plt.title("ROC curve - Modèle de régression logistique")
plt.xlabel("False Positive Rate (1 - Spécificité)")
plt.ylabel("True Positive Rate (Sensibilité)")
plt.grid(True)
plt.savefig('/mnt/data/credit_scoring_outputs/roc_curve.png', bbox_inches='tight')
plt.show()

# 6.2 Confusion matrix affichée en image (sans couleurs personnalisés)
plt.figure(figsize=(5,4))
plt.imshow(cm, interpolation='nearest')
plt.title('Matrice de confusion')
plt.xlabel('Prédiction')
plt.ylabel('Vérité terrain')
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")
plt.savefig('/mnt/data/credit_scoring_outputs/confusion_matrix.png', bbox_inches='tight')
plt.show()

# 6.3 Coefficients (bar chart)
plt.figure(figsize=(8,6))
coef_table_sorted = coef_table.sort_values(by='coefficient')
plt.barh(coef_table_sorted['feature'].astype(str), coef_table_sorted['coefficient'])
plt.title("Coefficients du modèle (logits) - signe et magnitude")
plt.xlabel("Coefficient (log-odds)")
plt.savefig('/mnt/data/credit_scoring_outputs/coefficients.png', bbox_inches='tight')
plt.show()

# 6.4 Distribution des probabilités prédites par classe réelle
plt.figure(figsize=(7,5))
plt.hist(y_proba[y_test==0], bins=30, alpha=0.6)
plt.hist(y_proba[y_test==1], bins=30, alpha=0.6)
plt.title("Distribution des probabilités prédites selon la classe réelle")
plt.xlabel("Probabilité prédite de défaut")
plt.ylabel("Nombre d'observations")
plt.legend(['Non-default (0)','Default (1)'])
plt.savefig('/mnt/data/credit_scoring_outputs/pred_proba_distribution.png', bbox_inches='tight')
plt.show()

# --- 7) Sauvegarde du modèle, scaler et jeu de données échantillon ---
joblib.dump(model, '/mnt/data/credit_scoring_outputs/logistic_model.joblib')
joblib.dump(scaler, '/mnt/data/credit_scoring_outputs/scaler.joblib')
df.sample(100).to_csv('/mnt/data/credit_scoring_outputs/sample_dataset.csv', index=False)
X_test.to_csv('/mnt/data/credit_scoring_outputs/X_test_scaled.csv', index=False)
y_test.to_csv('/mnt/data/credit_scoring_outputs/y_test.csv', index=False)

# --- 8) Sauvegarde d'un script .py prêt à exécuter dans VS Code 
script_content = r'''
# credit_scoring_model.py
# Script prêt à exécuter dans VS Code
# Instructions: installer les dépendances: pip install scikit-learn pandas matplotlib joblib
# Exécuter: python credit_scoring_model.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import joblib

# (Pour la simplicité du script reprend la génération de données synthétiques.

np.random.seed(42)
n = 5000
age = np.random.randint(21, 71, size=n)
income = np.round(np.random.normal(45000, 20000, size=n)).astype(int)
income = np.clip(income, 5000, 300000)
loan_amount = np.round(np.random.normal(15000, 10000, size=n)).astype(int)
loan_amount = np.clip(loan_amount, 500, 200000)
loan_term_months = np.random.choice([12,24,36,48,60], size=n, p=[0.1,0.2,0.4,0.2,0.1])
credit_history_score = np.round(np.random.normal(650, 70, size=n)).astype(int)
credit_history_score = np.clip(credit_history_score, 300, 850)
employment_years = np.round(np.abs(np.random.normal(5, 4, size=n)),1)
risk_raw = (
    0.6 * (loan_amount / (income + 1)) + 
    0.003 * (700 - credit_history_score) + 
    0.01 * (35 - (age - 21)) +
    0.05 * (2 - np.minimum(employment_years, 10))
)
prob_default = 1 / (1 + np.exp(- ( (risk_raw - risk_raw.mean())*4 )))
y = np.random.binomial(1, prob_default)
df = pd.DataFrame({
    "age": age,
    "income": income,
    "loan_amount": loan_amount,
    "loan_term_months": loan_term_months,
    "credit_history_score": credit_history_score,
    "employment_years": employment_years,
    "default": y
})

X = pd.get_dummies(df.drop(columns=['default']), columns=['loan_term_months'], prefix='term')
num_cols = ['age','income','loan_amount','credit_history_score','employment_years']
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])
X_train, X_test, y_train, y_test = train_test_split(X, df['default'], test_size=0.25, random_state=42, stratify=df['default'])

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:,1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_proba))
print("Classification report:")
print(classification_report(y_test, y_pred, digits=4))

cm = confusion_matrix(y_test, y_pred)
print("Matrice de confusion:")
print(cm)

# Sauvegarde
joblib.dump(model, 'logistic_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
df.to_csv('full_synthetic_dataset.csv', index=False)
print("Fichiers sauvegardés: logistic_model.joblib, scaler.joblib, full_synthetic_dataset.csv")

'''

with open('/mnt/data/credit_scoring_model.py', 'w', encoding='utf-8') as f:
    f.write(script_content)

print("\nLes fichiers et graphiques ont été sauvegardés dans /mnt/data/credit_scoring_outputs/")
print("Un script prêt à exécuter a été sauvegardé : /mnt/data/credit_scoring_model.py")
print("Téléchargez-le depuis le lien fourni dans la réponse principale.")

# afficher quelques fichiers sauvegardés
saved = os.listdir('/mnt/data/credit_scoring_outputs')
print("\nContenu du dossier /mnt/data/credit_scoring_outputs :")
print(saved)

# afficher tableau des coefficients final pour l'utilisateur
print("\nTableau complet des coefficients (extrait) :")
display(coef_table.head(15))

# si l'environnement propose une fonction d'affichage interactive, l'utiliser
if display_dataframe_to_user is not None:
    display_dataframe_to_user("Échantillon dataset", df.sample(200))
else:
    print("Affichage interactif non disponible dans cet environnement.")
