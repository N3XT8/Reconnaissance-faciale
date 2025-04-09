import app as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import io
import time

# === Configuration principale ===
st.set_page_config(page_title="ML Dashboard", layout="centered")
st.title("🧠 Tableau de Bord - Données & Machine Learning")

# === Navigation dans la barre latérale ===
st.sidebar.title("📌 Menu")
menu = st.sidebar.radio("Choisissez une section :", ["📂 Importer un Fichier", "🔍 Analyse EDA", "🛠️ Modélisation ML", "🔮 Prédictions"])

# === Variables de session ===
dataset = st.session_state.get("dataset", None)
trained_models = st.session_state.get("trained_models", None)
model_options = st.session_state.get("model_options", None)
features_df = st.session_state.get("features_df", None)

# === Section 1 : Chargement de données ===
if menu == "📂 Importer un Fichier":
    st.header("📥 Chargement du Fichier CSV")
    file = st.file_uploader("Sélectionnez un fichier .csv", type="csv")

    if file:
        with st.spinner("Lecture du fichier en cours..."):
            try:
                time.sleep(1)
                dataset = pd.read_csv(file, on_bad_lines='skip')
                st.session_state["dataset"] = dataset
                st.success("Fichier chargé avec succès ✅")
                st.dataframe(dataset.head())
            except Exception as err:
                st.error(f"Erreur de chargement : {err}")

# === Section 2 : Analyse exploratoire ===
elif menu == "🔍 Analyse EDA":
    st.header("🔍 Exploration des Données")
    if dataset is not None:
        tab_stats, tab_graphs, tab_nulls = st.tabs(["📊 Statistiques", "📉 Graphiques", "🧩 Valeurs Manquantes"])

        with tab_stats:
            st.subheader("Résumé Statistique")
            st.dataframe(dataset.describe())
            st.text("\nStructure des Données :")
            buf = io.StringIO()
            dataset.info(buf=buf)
            st.text(buf.getvalue())

        with tab_graphs:
            st.subheader("Visualisation")
            num_col = st.selectbox("Colonne numérique à visualiser", dataset.select_dtypes(include=['number']).columns)
            if num_col:
                graph = px.histogram(dataset, x=num_col)
                st.plotly_chart(graph)

        with tab_nulls:
            st.subheader("Valeurs Manquantes")
            nulls = dataset.isnull().sum()
            if nulls.sum() > 0:
                st.write(nulls)
                if st.button("Remplir avec la moyenne"):
                    dataset.fillna(dataset.mean(numeric_only=True), inplace=True)
                    st.session_state["dataset"] = dataset
                    st.success("Valeurs nulles remplacées ✅")
            else:
                st.info("Aucune valeur manquante détectée.")
    else:
        st.warning("Veuillez d'abord importer un fichier.")

# === Section 3 : Entraînement des modèles ===
elif menu == "🛠️ Modélisation ML":
    st.header("🛠️ Machine Learning")
    if dataset is not None:
        st.sidebar.subheader("Configuration")
        task_type = st.sidebar.selectbox("Tâche ML", ["Classification", "Régression"])
        target = st.sidebar.selectbox("Variable cible", dataset.columns)

        if target:
            features = dataset.drop(columns=[target])
            labels = dataset[target]
            st.session_state["features_df"] = features

            # Encodage des colonnes non numériques
            cat_columns = features.select_dtypes(include=['object']).columns
            if len(cat_columns) > 0:
                encoder = OneHotEncoder()
                enc_data = pd.DataFrame(encoder.fit_transform(features[cat_columns]).toarray())
                features = features.drop(columns=cat_columns).reset_index(drop=True).join(enc_data)

            X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=0)

            if task_type == "Classification":
                model_options = {
                    "Forêt Aléatoire": RandomForestClassifier(),
                    "Boosting": GradientBoostingClassifier(),
                    "Régression Logistique": LogisticRegression(max_iter=1000),
                    "SVC": SVC()
                }
            else:
                model_options = {
                    "Forêt Aléatoire (Reg)": RandomForestRegressor(),
                    "Boosting (Reg)": GradientBoostingRegressor(),
                    "Régression Linéaire": LinearRegression(),
                    "SVR": SVR()
                }
            st.session_state["model_options"] = model_options

            performance = []
            with st.spinner("Apprentissage en cours..."):
                for mdl_name, mdl in model_options.items():
                    mdl.fit(X_train, y_train)
                    y_pred = mdl.predict(X_test)

                    if task_type == "Classification":
                        rep = classification_report(y_test, y_pred, output_dict=True)
                        performance.append({
                            "Modèle": mdl_name,
                            "Précision": rep['weighted avg']['precision'],
                            "Rappel": rep['weighted avg']['recall'],
                            "F1": rep['weighted avg']['f1-score']
                        })
                    else:
                        performance.append({
                            "Modèle": mdl_name,
                            "MAE": mean_absolute_error(y_test, y_pred),
                            "MSE": mean_squared_error(y_test, y_pred),
                            "R²": r2_score(y_test, y_pred)
                        })

                st.session_state["trained_models"] = model_options
                st.success("Modèles entraînés !")
                st.dataframe(pd.DataFrame(performance))
    else:
        st.error("Aucun fichier n'a été chargé.")

# === Section 4 : Prédiction ===
elif menu == "🔮 Prédictions":
    st.header("🔮 Prédictions en Direct")
    if trained_models and features_df is not None:
        model_choice = st.selectbox("Modèle à utiliser", list(model_options.keys()))
        chosen_model = trained_models[model_choice]

        inputs_dict = {}
        for column in features_df.columns:
            inputs_dict[column] = st.slider(f"{column}", float(features_df[column].min()), float(features_df[column].max()), float(features_df[column].mean()))

        prediction_input = pd.DataFrame([inputs_dict])

        if st.button("Lancer la prédiction"):
            with st.spinner("Calcul en cours..."):
                result = chosen_model.predict(prediction_input)
                st.success(f"✅ Résultat : {result[0]}")
    else:
        st.info("Veuillez d'abord entraîner les modèles.")
