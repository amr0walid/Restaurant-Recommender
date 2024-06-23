# -*- coding: utf-8 -*-

from surprise import Dataset, Reader
from surprise import SVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
from surprise.model_selection import cross_validate, train_test_split
import numpy as np
import streamlit as st

user = pd.read_csv('ratings.csv')
user.rename(columns={'ID': 'RestID'}, inplace=True)
user.rename(columns={'userId': 'CustomerID'}, inplace=True)

rest = pd.read_csv('Europe_Restaurants.csv')

rest['RestID'] = rest.index+1
rest.rename_axis('ID_Old', inplace=True)

rest.rename(columns={'Cuisine Style': 'Cuisine', 'Number of Reviews': 'Count', 'Price Range': 'Price'}, inplace=True)
rest.drop(columns=['Unnamed: 0', 'URL_TA', 'ID_TA', 'Ranking', 'City'], inplace=True)
rest['Rating'].fillna(rest['Rating'].mean(), inplace=True)

price_map = {'$': 20, '$$ - $$$': 50, '$$$$':100}
rest['Price'] = rest['Price'].map(price_map)

rest['Reviews'] = rest['Reviews'].apply(lambda x: x.split('],')[0].replace('[','').replace("'", "") if isinstance(x, str) else '')


rest.reset_index(drop=True, inplace=True)
rest.drop(columns=['ID_Old'], inplace=True, errors='ignore')
rest = rest.iloc[:-100000]
rest.to_csv('cleaned_Europe_Restaurants.csv', index=False)
rest.head()

df = pd.merge(rest, user, on='RestID', how='inner')

df.head()

# Aggregate by rest
agg_ratings = df.groupby('Name').agg(mean_rating = ('rating', 'mean'),
                                                number_of_ratings = ('rating', 'count')).reset_index()

# Keep the rest with over 50 ratings
agg_ratings_GT50 = agg_ratings[agg_ratings['number_of_ratings']>50]
agg_ratings_GT50.info()

from sklearn.feature_extraction.text import TfidfVectorizer

#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
rest['Reviews'] = rest['Reviews'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(rest['Reviews'])

#Output the shape of tfidf_matrix
tfidf_matrix.shape

# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

#Construct a reverse map of indices and restauarnt names
indices = pd.Series(rest.index, index=rest['Name']).drop_duplicates()

# Function that takes in movie title as input and outputs most similar movies
def get_content_recommendations(name,top_n, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[name]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the restaurantss based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:top_n+1]

    # Get the restaurant indices
    restaurants_top = [i[0] for i in sim_scores]

    # Return the top 10 most similar
    return rest['Name'].iloc[restaurants_top]

get_content_recommendations('La Rive',10)

agg_ratings_GT50.sort_values(by='number_of_ratings', ascending=False).head()

df_GT50 = pd.merge(df, agg_ratings_GT50[['Name']], on='Name', how='inner')
df_GT50.info()

columns =['CustomerID','Name','rating']
df_GT50=df_GT50[columns]

reader = Reader(rating_scale=(1,5))
data= Dataset.load_from_df(df_GT50,reader)
trainset, testset = train_test_split(data, test_size=0.25)

model =SVD(n_factors=100)
model.fit(trainset)

def get_collaborative_recommendations(CustomerID, top_n,testset):
    testset = filter(lambda x: x[0] == CustomerID, testset)
    predictions = model.test(testset)
    predictions.sort(key=lambda x: x.est, reverse=True)
    recommendations = [prediction.iid for prediction in predictions[:top_n]]
    return recommendations

def get_hybrid_recommendations(CustomerID, RestID, top_n,testset):
    content_based_recommendations = get_content_recommendations(RestID,top_n )
    collaborative_filtering_recommendations = get_collaborative_recommendations(CustomerID,top_n, testset )
    hybrid_recommendations = list(set(content_based_recommendations + collaborative_filtering_recommendations))
    return hybrid_recommendations[:top_n]

# CustomerID = 17
# Name = 'La Rive'
# recommendations = get_hybrid_recommendations(CustomerID,Name,10,testset)


st.markdown("# hybrid Recommender systems")

with st.form(key='my_form'):

    st.text_input(label='enter customerid:',key= 'CustomerID')
    st.text_input(label='enter name of restaurant:',key= 'Name')
    
    

    submit = st.form_submit_button(label='recommend')

    if submit :
        rrecommendations = get_hybrid_recommendations(int(st.session_state.CustomerID),st.session_state.Name,10,testset)
        st.table(rrecommendations)
