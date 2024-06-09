<h1 align="center">
  <a href="https://songmap.xyz">
    SongMap 🎵🌍
  </a>
</h1>
<h3 align="center">
Streamlit app that uses an XGBoost and Spark to predict the country of origin of songs.
</h3>

This project aims to predict the country of origin for a song based on its features.  We use a machine learning model trained on a dataset containing various song attributes.
        
### How It Works 🛠️
- **Data**: We used a [HuggingFace](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset) dataset containing song features (check out the link for information on what does each feature mean), and a [MusicBrainz](https://data.metabrainz.org/pub/musicbrainz/data/json-dumps/20240608-001001/) dataset
            containing country information of artists.
- **Feature Engineering**: We processed the data to obtain the country of origin for each song whose artist's country was known.
- **Model Training**: We trained an XGBoost model to predict the country of origin based on song features.
- **Inference**: Once the model was trained, all songs (those with known artist country and those without) were saved without the country of origin.
            When a song is searched, the model predicts the country of origin based on the song features (no matter if the artist's country is known or not) and 
            the 3 countries with the highest probability are shown.
- **Explainability**: We provide [SHAP](https://christophm.github.io/interpretable-ml-book/shap.html) values to explain the model's decisions.
- **Accuracy**: The model has an accuracy of 70% on the test set.

### How to deploy 🚀

1. Clone the repository
```bash
git clone https://github.com/FerranAD/songmap.git
cd songmap
```

2. Run docker compose 🐋
```bash
docker compose up
```