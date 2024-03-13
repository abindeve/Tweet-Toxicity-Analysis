import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Define text cleaning function
def clean_text(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation and special characters
    tokens = word_tokenize(text)  # Tokenize text
    stop_words = set(stopwords.words('english'))  # Get English stopwords
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    lemmatizer = WordNetLemmatizer()  # Initialize lemmatizer
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization
    return ' '.join(tokens)  


# Setup Streamlit layout
st.title("Tweet Toxicity Analysis")
st.sidebar.header("Settings")

# File upload and data loading
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("## Dataset Preview")
    st.dataframe(df.head())
    
    # Generate word clouds
    st.sidebar.subheader("Word Clouds")
    if st.sidebar.checkbox("Show Word Clouds"):
        st.write("## Word Clouds")
        toxic_words = ' '.join(df[df['Toxicity'] == 1]['tweet'])
        non_toxic_words = ' '.join(df[df['Toxicity'] == 0]['tweet'])
        
        toxic_wordcloud = WordCloud(width=800, height=400).generate(toxic_words)
        non_toxic_wordcloud = WordCloud(width=800, height=400).generate(non_toxic_words)

        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        ax[0].imshow(toxic_wordcloud)
        ax[0].set_title('Toxic Tweets')
        ax[0].axis('off')
        ax[1].imshow(non_toxic_wordcloud)
        ax[1].set_title('Non-Toxic Tweets')
        ax[1].axis('off')
        st.pyplot(fig)

    # Clean the tweets and display them
    st.sidebar.subheader("Text Cleaning")
    if st.sidebar.checkbox("Clean and Display Tweets"):
        with st.spinner("Cleaning text data..."):
            df['cleaned_tweet'] = df['tweet'].apply(clean_text)
        st.write("## Cleaned Tweets")
        st.dataframe(df[['tweet', 'cleaned_tweet']].head())

    # Select models
    st.sidebar.subheader("Select Models")
    selected_models = st.sidebar.multiselect("Choose models to evaluate", ["Decision Tree", "Random Forest", "Naive Bayes", "K-NN", "SVM"])

    # Train models and display metrics
    vectorization_method = st.sidebar.radio("Choose Vectorization Method", ["Bag of Words", "TF-IDF"])
    train_button = st.sidebar.button("Train Models")
    
    # Placeholder for displaying metrics
    metrics_placeholder = st.empty()

    # Train and evaluate models
    if train_button:
        if not selected_models:
            st.warning("Please select at least one model.")
        else:
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(df['cleaned_tweet'], df['Toxicity'], test_size=0.2, random_state=42)
            
            # Vectorization
            if vectorization_method == "Bag of Words":
                vectorizer = CountVectorizer()
            else:
                vectorizer = TfidfVectorizer()
            X_train_vectorized = vectorizer.fit_transform(X_train)
            X_test_vectorized = vectorizer.transform(X_test)

            # Placeholder for storing metrics
            model_metrics = []

            # Model training and evaluation
            for selected_model in selected_models:
                if selected_model == "Decision Tree":
                    model = DecisionTreeClassifier()
                elif selected_model == "Random Forest":
                    model = RandomForestClassifier()
                elif selected_model == "Naive Bayes":
                    model = MultinomialNB()
                elif selected_model == "K-NN":
                    model = KNeighborsClassifier()
                elif selected_model == "SVM":
                    model = SVC(probability=True)

                with st.spinner(f"Training {selected_model} model..."):
                    model.fit(X_train_vectorized, y_train)
                    y_pred = model.predict(X_test_vectorized)
                    y_pred_prob = model.predict_proba(X_test_vectorized)[:, 1] if selected_model != "SVM" else model.decision_function(X_test_vectorized)
                    
                    # Calculate metrics
                    precision = precision_score(y_test, y_pred)
                    recall = recall_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)
                    roc_auc = roc_auc_score(y_test, y_pred_prob)

                    # Store metrics
                    model_metrics.append({
                        "Model": selected_model,
                        "Precision": precision,
                        "Recall": recall,
                        "F1 Score": f1,
                        "ROC AUC Score": roc_auc
                    })

                    # Display metrics
                    metrics_placeholder.subheader(f"{selected_model} Model Metrics")
                    metrics_placeholder.write(f"Precision: {precision:.2f}")
                    metrics_placeholder.write(f"Recall: {recall:.2f}")
                    metrics_placeholder.write(f"F1 Score: {f1:.2f}")
                    metrics_placeholder.write(f"ROC AUC Score: {roc_auc:.2f}")

                    # Confusion matrix
                    st.subheader(f"{selected_model} Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    plt.figure()
                    sns.heatmap(cm, annot=True, fmt='d')
                    plt.title(f"{selected_model} Confusion Matrix")
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    st.pyplot(plt.gcf())  # Passing the current figure to st.pyplot()

                    # ROC Curve
                    st.subheader(f"{selected_model} ROC Curve")
                    roc_fig, roc_ax = plt.subplots()
                    RocCurveDisplay.from_predictions(y_test, y_pred_prob, ax=roc_ax)
                    roc_ax.set_title(f"{selected_model} ROC Curve")
                    st.pyplot(roc_fig)

            # Display comparison table
            st.subheader("Model Comparison")
            metrics_df = pd.DataFrame(model_metrics)
            st.write(metrics_df)
else:
    st.warning("Please upload a CSV file to get started.")
