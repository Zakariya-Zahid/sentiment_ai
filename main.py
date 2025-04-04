import streamlit as st
import joblib

# --- Load Model and Vectorizer ---
print(os.listdir("./models")) #add this line.
model = joblib.load("models/random_forest_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# --- Streamlit Page Config ---
st.set_page_config(page_title="Sentiment Analyzer", page_icon="https://cdn-icons-png.flaticon.com/512/9850/9850865.png", layout="centered")

# --- Responsive CSS and Hiding Streamlit UI ---
st.markdown("""
    <style>
    #MainMenu, footer, header {display: none;}

    /* Make title section responsive */
    .responsive-header {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 12px;
        flex-wrap: wrap;
        text-align: center;
    }

    .responsive-header h1 {
        color: #4A6EE0;
        font-size: 2rem;
        margin: 0;
        padding: 0;
    }

    .responsive-paragraph {
        text-align: center;
        font-size: 1.1rem;
        margin-top: -8px;
        padding: 0 10px;
    }

    @media screen and (max-width: 600px) {
        .responsive-header h1 {
            font-size: 1.5rem;
        }

        .responsive-paragraph {
            font-size: 1rem;
        }
    }
    </style>
""", unsafe_allow_html=True)

# --- Header Section ---
st.markdown("""
    <div class="responsive-header" style="margin-bottom:1.8em">
        <img src = "https://cdn-icons-png.flaticon.com/512/9850/9850865.png" width="50">
        <h1>Sentiment Analysis App</h1>
    </div>
""", unsafe_allow_html=True)

# --- Input Form ---
with st.form(key="sentiment_form"):
    user_input = st.text_area("📝 Enter your text below:", height=150, placeholder="Type your sentence here...")
    submit_btn = st.form_submit_button("🔍 Analyze")

# --- Prediction Logic ---
if submit_btn:
    if user_input.strip():
        with st.spinner("Analyzing sentiment..."):
            vectorized_input = vectorizer.transform([user_input])
            prediction = model.predict(vectorized_input)[0]
            prediction_str = str(prediction)

            sentiment_map = {
                "-1": "fa-solid fa-face-angry",     # Negative
                "0": "fa-solid fa-face-meh",        # Neutral
                "1": "fa-solid fa-face-smile"       # Positive
            }

            result_icon = sentiment_map.get(prediction_str, "fa-solid fa-question-circle")
            result_text = {
                "-1": "Negative",
                "0": "Neutral",
                "1": "Positive"
            }.get(prediction_str, "Unknown")

            color = {
                "-1": "red",
                "0": "gray",
                "1": "green"
            }.get(prediction_str, "black")

            # Display Result
            st.markdown(f"""
                <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
                <div style="text-align: center;">
                    <h3 style='color: {color}; font-size: 1.5rem;'>
                        <i class="{result_icon}" style="font-size: 1.8rem;"></i> Sentiment: {result_text}
                    </h3>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter some text to analyze.")

# --- Footer ---
st.markdown("""
    <hr>
    <div style='text-align: center; font-size: 14px;'>
        Built with ❤️ by <a href="https://github.com/Zakariya-Zahid" target="_blank">Zakariya Bukhari</a>
    </div>
""", unsafe_allow_html=True)
