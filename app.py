import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. PAGE CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="Global Spotify Recommendation Engine",
    page_icon="🎵",
    layout="wide"
)  

# Apply Spotify Branding Palette using Custom CSS
st.markdown("""
    <style>
    .main { background-color: #121212; color: #FFFFFF; }
    h1, h2, h3 { color: #1DB954 !important; }
    .stButton>button {
        background-color: #1DB954 !important;
        color: white !important;
        border-radius: 20px !important;
        border: none !important;
    }
    .stSelectbox, .stTextInput { color: black !important; }
    </style>
    """, unsafe_allow_html=True)


# --- 2. RECOMMENDATION ENGINE CLASS ---
class SpotifyRecommendationEngine:
    def __init__(self, df):
        self.df = df.copy().reset_index(drop=True)
        self.feature_columns = [
            'danceability', 'energy', 'valence', 
            'acousticness', 'speechiness', 'instrumentalness', 'tempo'
        ]
        self.scaler = MinMaxScaler()
        self.normalized_matrix = None
        self._prepare_engine()

    def _prepare_engine(self):
        # Data cleaning for bulletproof matching
        self.df['name_clean'] = self.df['name'].astype(str).str.lower().str.strip()
        
        # Scale continuous sonic spaces
        numerical_features = self.df[self.feature_columns]
        self.normalized_matrix = self.scaler.fit_transform(numerical_features)

    def recommend(self, target_idx, top_n=5):
        target_vector = self.normalized_matrix[target_idx].reshape(1, -1)
        
        # Calculate global Cosine Similarity
        similarity_scores = cosine_similarity(target_vector, self.normalized_matrix).flatten()
        
        # Build working copy of data with scores
        results_df = self.df.copy()
        results_df['similarity_score'] = similarity_scores
        
        # Remove the input track itself from potential outputs
        results_df = results_df.drop(target_idx)
            
        # Get top matches sorted by similarity score
        top_matches = results_df.sort_values(by='similarity_score', ascending=False).head(top_n)
        
        return top_matches[['id', 'name', 'artists', 'year', 'similarity_score']]


# --- 3. DATA LOADING PIPELINE ---
@st.cache_data # Caches data in memory so it doesn't re-run on every user click
def load_data():
    # Replace 'spotify_songs.csv' with your local dataset path
    # Example mock structure to ensure compatibility with your localized metadata
    try:
        file=st.file_uploader("Upload your csv file :",type='csv')
        df = pd.read_csv(file)
    except FileNotFoundError:
        # Fallback dummy generator for development demo purposes
        st.warning("⚠️ 'spotify_songs.csv' not found. Generating standardized multilingual demo dataset.")
    return df

df = load_data()
engine = SpotifyRecommendationEngine(df)

# --- 4. STREAMLIT UI DESIGN ---
st.title("🎵 Vector-Based Multilingual Spotify Recommendation Engine")
st.write("Discover music across borders based on mathematical audio geometry.")

# Sidebar Controls
st.sidebar.header("🔍 Search & Filter Controls")

search_query=st.sidebar.text_input("Type song name:","")
if search_query:
    filtered_df=df[df['name'].astype(str).str.contains(search_query,case=False,na=False)]
    song_list=sorted(filtered_df['name'].unique())
else:
    if 'popularity' in df.columns:
        top_100=df.sort_values(by='popularity',ascending=False).head(100)
        song_list=sorted(top_100['name'].unique())
    else:
        song_list=sorted(df['name'].head(100).unique())

    
selected_song_name = st.sidebar.selectbox("search song",song_list)

# Find matching track profiles to populate the sidebar telemetry data
matching_tracks = df[df['name'] == selected_song_name]
if not matching_tracks.empty:
    selected_track_idx = matching_tracks.index[0]
    selected_track_row = matching_tracks.iloc[0]
    
    st.sidebar.markdown(f"**Current Selection:** *{selected_track_row['artists']}*")
    
    st.sidebar.divider()

    # Dynamic Regional Discovery Filters
    st.sidebar.subheader("Cross-Border Constraints")
    num_recommendations = st.sidebar.slider("Number of songs to generate", 3, 10, 5)

    # --- 5. EXECUTE VECTOR MATCHING ACTION ---
    if st.sidebar.button("Generate Sound-Alikes"):
        st.subheader(f"🎧 Closest Audio Matches for '{selected_song_name}'")
        
        with st.spinner("Computing cosine similarities across multi-dimensional vector matrices..."):
            recs = engine.recommend(
                target_idx=selected_track_idx,
                top_n=num_recommendations
            )
        
        if isinstance(recs, str):
            st.error(recs)
        else:
            # Display results in a beautifully styled native interactive dataframe table
            st.dataframe(
                recs.style.format({'similarity_score': "{:.4f}"}),
                use_container_width=True,
                hide_index=True
            )
            
            # Metric Callout Box for the closest match
            best_match = recs.iloc[0]
            st.success(f"💡 **Top Recommendation Idea:** Try listening to **'{best_match['name']}'** by **{best_match['artists']}**. It has a sonic vector match score of **{best_match['similarity_score']:.2%}**!")
else:
    st.error("Select a valid track from your library.")