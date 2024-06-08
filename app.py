import streamlit as st
import xgboost as xgb
from streamlit_shap import st_shap
import shap
import joblib
from pyspark.sql import SparkSession
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import folium_static
import pyspark.sql.functions as sfn

st.set_page_config(page_title="Songmap", page_icon="images/icon.png")

spark = SparkSession.builder.appName("SongOriginPrediction").getOrCreate()

model = joblib.load("models/model.joblib")
country_label_encoder = joblib.load("models/country_label_encoder.joblib")
genre_label_encoder = joblib.load("models/genre_label_encoder.joblib")

data_path = 'datasets/processed/merged_df_without_country.parquet'
train_data_path = 'datasets/processed/joined_df.parquet'
spark_df = spark.read.parquet(data_path) \
    .filter(sfn.col("track_genre") != "study")

country_data = {
    'NL': ['Netherlands', 'NLD', '🇳🇱'],
    'MX': ['Mexico', 'MEX', '🇲🇽'],
    'AT': ['Austria', 'AUT', '🇦🇹'],
    'AU': ['Australia', 'AUS', '🇦🇺'],
    'CA': ['Canada', 'CAN', '🇨🇦'],
    'GB': ['United Kingdom', 'GBR', '🇬🇧'],
    'BR': ['Brazil', 'BRA', '🇧🇷'],
    'DE': ['Germany', 'DEU', '🇩🇪'],
    'ES': ['Spain', 'ESP', '🇪🇸'],
    'TR': ['Turkey', 'TUR', '🇹🇷'],
    'KR': ['South Korea', 'KOR', '🇰🇷'],
    'US': ['United States', 'USA', '🇺🇸'],
    'IN': ['India', 'IND', '🇮🇳'],
    'JM': ['Jamaica', 'JAM', '🇯🇲'],
    'FR': ['France', 'FRA', '🇫🇷'],
    'IT': ['Italy', 'ITA', '🇮🇹'],
    'SE': ['Sweden', 'SWE', '🇸🇪'], 
    'JP': ['Japan', 'JPN', '🇯🇵'],
    'CO': ['Colombia', 'COL', '🇨🇴'],
    'AR': ['Argentina', 'ARG', '🇦🇷'],
    'PR': ['Puerto Rico', 'PRI', '🇵🇷']
}

country_codes = list(country_data.keys())

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

def highlight_country(map_object, country_code):
    country = world[world.iso_a3 == country_data[country_code][1]]
    folium.GeoJson(
        country,
        style_function=lambda x: {
            'fillColor': '#1f78b4',
            'color': '#1f78b4',
            'weight': 3,
            'fillOpacity': 0.6
        }
    ).add_to(map_object)

    if not country.empty:
        bounds = country.total_bounds
        center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
        return center, country.area.iloc[0]
    return [20, 0], None

def calculate_zoom(area):
    print(area)
    if area > 600:
        return 2
    elif area > 250:
        return 3
    else:
        return 4

def show_map(country_code):
    center, area = highlight_country(folium.Map(location=[20, 0], zoom_start=2), country_code)

    zoom_level = calculate_zoom(area) if area is not None else 2

    m = folium.Map(location=center, zoom_start=zoom_level, tiles='cartodb positron')

    highlight_country(m, country_code)

    map_container = st.container()
    with map_container:
        folium_static(m)

def get_songs(input_text: str) -> list[str]:
    songs = spark_df.filter(spark_df.track_and_artist.contains(input_text.lower())).toPandas()
    return pd.concat([songs["track_name"], songs["name"]], axis=1).apply(lambda x: f"🎼 {x[0]} - {x[1]}", axis=1).tolist()

def get_song(track_and_artist: str):
    song = spark_df.filter(spark_df.track_and_artist == track_and_artist.lower()).orderBy("popularity", ascending=False).limit(1).toPandas()
    song["track_genre"] = genre_label_encoder.transform(song["track_genre"])
    return song

def get_song_country_top3(song):
    prediction = model.predict_proba(song.drop(["name", "track_name", "track_and_artist"], axis=1))
    top_3_indexes = prediction.argsort()[0][::-1][:3]
    top_3_countries = country_label_encoder.inverse_transform(top_3_indexes)
    return top_3_countries

def show_force_plot(song, top_3_countries: list[str]):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(song.drop(["name", "track_name", "track_and_artist"], axis=1))
    top_3_indexes = [country_label_encoder.transform([country])[0] for country in top_3_countries]
    for i, index in enumerate(top_3_indexes):
        st.write(f"##### #{i + 1} {country_data[country_label_encoder.inverse_transform([index])[0]][0]} {country_data[country_label_encoder.inverse_transform([index])[0]][2]}")
        st_shap(shap.force_plot(explainer.expected_value[index], shap_values[..., index], song.drop(["name", "track_name", "track_and_artist"], axis=1)))

def format_song_name(song_name: str) -> str:
    formatted_song = song_name[2:].split(" - ")
    result = formatted_song[-1]
    for item in formatted_song[:-1]:
        result += " - " + item
    return result

def show_shap_for_country(selected_country):
    st.image(f"images/shaps/{selected_country}.png")

def main():
    
    tabs = st.tabs(["Home", "More Info"])

    with tabs[0]:
        st.title("Want to know where your favorite song is from? 🌍")

        search_query = st.text_input("Enter track or artist name:")

        if search_query:
            filtered_items = get_songs(search_query)
            selected_item = st.selectbox("Select an item:", filtered_items)
            if selected_item:
                if st.button("Predict! ✨"):
                    result = format_song_name(selected_item)
                    song = get_song(result)
                    top_3_countries = get_song_country_top3(song)

                    st.write(f"### And our model says... 🥁")
                    st.write(f"#### {country_data[top_3_countries[0]][0]}! {country_data[top_3_countries[0]][2]}")

                    show_map(top_3_countries[0])

                    st.write(f"## Why? 🤔")

                    st.write(f"Lets take a look at which features contributed to the prediction of {country_data[top_3_countries[0]][0]} 📊, as well as the closest 2 candidates.")

                    st.write("The following plots show, in red, which features contributed the most to make each country the chosen one, and in blue, the features that contributed the most to make each country __NOT__ the chosen one.")

                    st.write("The number in bold is the one that will be used to make the final prediction, so the the country with the highest number will be the chosen one.")

                    show_force_plot(song, top_3_countries)

    with tabs[1]:
        st.title("More Info")
        st.write("""
        This project aims to predict the country of origin for a song based on its features. 
        We use a machine learning model trained on a dataset containing various song attributes.
        
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

        ### More explainability! 🤓

        Remember those plots we showed you? They are called SHAP values, and they help us understand how the model makes its decisions. But the ones we showed you are
                 just for a single song. Choose a country to see the SHAP values for the whole dataset! Which features are the most relevant for each country? 🤔

        > **Note**: To understand how much does a feature influence the prediction, you have to look at
        how much presence does it have on values that are away from 0. The more it is away from 0, the more it influences the prediction. Don't get confused
        by the color, in this case it denotes the value of the feature, not how much it influences the prediction.
        """)

        selected_country = st.selectbox('Select a country to highlight:', country_codes, format_func=lambda x: country_data[x][0])

        show_shap_for_country(selected_country)
        st.write("""
        ### Even more explainability!!! 🤯

        Since an XGBoost is a tree-based model, we cant literally see how it takes its decisions! Take a look at the
                 following image (click [here](https://github.com/FerranAD/songmap/blob/main/images/xgb_tree.png) for full resolution) to see the insights of the model! 🌳
        """)

        st.image("images/xgb_tree.png")

        st.write("""
        For any questions or feedback, please contact us at [oriol@agost.info](mailto:oriol@agost.info) or [ferran@ferranaran.com](mailto:ferran@ferranaran.com).
                 """)

    st.markdown(
        """
        <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #f0f0f0;
            color: black;
            padding: 10px 0;
            text-align: center;
        }
        </style>
        <div class="footer">
            <p>Made with ❤️ by Oriol and Ferran, source code available on <a style='text-decoration: none;' href="https://github.com/FerranAD/songmap" target="_blank">GitHub <img src="https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Flogos-world.net%2Fwp-content%2Fuploads%2F2020%2F11%2FGitHub-Symbol.png&f=1&nofb=1&ipt=6d0687e15eb81fa1b6f82b4fa61acb018af35b6e1335db9fede3463f4a32a985&ipo=images" height="21" width="38" style="vertical-align: text-bottom;"></a></p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == '__main__':
    main()