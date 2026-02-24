# Stratégie d’élaboration du modèle – AirParadise Sentiment

L’objectif du projet est de détecter automatiquement les tweets négatifs afin d’anticiper un potentiel bad buzz et permettre une réaction rapide du service communication.  
Dans ce contexte, le coût d’un faux négatif est plus élevé qu’un faux positif : ne pas détecter un signal réellement négatif peut avoir un impact direct sur l’e-réputation.

La démarche adoptée repose sur une montée progressive en complexité.

La première étape consiste à établir une baseline simple à l’aide d’un modèle TF-IDF couplé à une régression logistique. Cette approche permet de vérifier rapidement si le signal lexical est exploitable et d’obtenir une référence de performance facilement interprétable.

Dans un second temps, un modèle deep learning simple est introduit. Il repose sur une couche d’embedding suivie d’un GlobalAveragePooling. L’objectif est d’apprendre une représentation dense des mots tout en conservant une architecture légère, afin de comparer l’apport réel du deep learning par rapport à la baseline classique.

Une architecture plus avancée est ensuite testée, basée sur un embedding suivi d’un BiLSTM. Ce type de modèle permet de capturer les dépendances séquentielles et de mieux modéliser le contexte des phrases, ce qui est particulièrement utile pour des textes courts et ambigus comme les tweets.

Enfin, une approche par transfer learning est étudiée via un modèle de type BERT pré-entraîné. Cette étape vise à exploiter des représentations linguistiques riches apprises sur de larges corpus. Elle permet de comparer le gain potentiel en performance face à l’augmentation du coût de calcul et de la complexité de déploiement.

Le choix du modèle final repose sur plusieurs critères combinés : performance (F1-score et rappel sur la classe négative), robustesse, latence d’inférence, taille du modèle et facilité d’intégration dans une architecture API déployée.

L’ensemble des expérimentations est suivi via MLflow afin d’assurer la traçabilité des paramètres, des métriques et des artefacts. Le projet est versionné avec Git, testé automatiquement via GitHub Actions et déployé sur Hugging Face Spaces. Un endpoint de feedback est prévu afin de collecter les prédictions incorrectes et préparer une stratégie de monitoring et d’amélioration continue.

À terme, des améliorations sont envisagées, notamment un fine-tuning spécifique au domaine de l’aviation, une meilleure gestion du sarcasme et la mise en place d’un mécanisme de détection de drift linguistique.