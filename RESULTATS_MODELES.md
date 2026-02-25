# Synthèse comparative des modèles – Air Paradis

## Objectif

Comparer plusieurs approches de classification de sentiments afin d’identifier le meilleur compromis entre performance, complexité et coût de déploiement.

---

## Modèles testés

1. Baseline TF-IDF + Logistic Regression  
2. Keras simple (Embedding + GlobalAveragePooling)  
3. Keras avancé (Embedding + BiLSTM)  
4. BERT (à implémenter)

---

## Résultats validation

| Modèle | AUC validation | Complexité | Temps entraînement | Remarques |
|--------|---------------|------------|--------------------|-----------|
| TF-IDF + LR | XX | Faible | Très rapide | Baseline robuste |
| Keras simple | XX | Moyenne | Rapide | Meilleur compromis |
| Keras BiLSTM | XX | Élevée | Plus long | Capte mieux le contexte |
| BERT | TBD | Très élevée | Long | Transfer learning |

---

## Analyse

Le modèle simple Keras offre un bon équilibre performance / complexité.  
Le BiLSTM améliore légèrement l’AUC mais augmente le coût.  
L’intégration d’un modèle pré-entraîné type BERT permettra de valider l’apport du transfer learning.

---

## Conclusion

Pour un déploiement rapide et scalable, le modèle Keras simple est recommandé.  
Pour maximiser la performance, un modèle basé sur BERT sera privilégié.

