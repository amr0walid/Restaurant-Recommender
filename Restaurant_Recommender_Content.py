#!/usr/bin/env python
# coding: utf-8

# In[33]:


# Data processing
import pandas as pd
import numpy as np
import scipy.stats

# Visualization
import seaborn as sns
import streamlit as st

# Similarity
from sklearn.metrics.pairwise import cosine_similarity


# In[61]:


rest = pd.read_csv('Europe_Restaurants.csv')

rest['RestID'] = rest.index+1
rest.rename_axis('ID_Old', inplace=True)

rest.rename(columns={'Cuisine Style': 'Cuisine', 'Number of Reviews': 'Count', 'Price Range': 'Price'}, inplace=True)
rest.drop(columns=['Unnamed: 0', 'URL_TA', 'ID_TA', 'Ranking'], inplace=True)
rest['Cuisine'] = rest['Cuisine'].apply(lambda x: eval(x.replace('[','').replace(']','')) if isinstance(x, str) else [])


rest['Rating'].fillna(rest['Rating'].mean(), inplace=True)

price_map = {'$': 20, '$$ - $$$': 50, '$$$$':100}
rest['Price'] = rest['Price'].map(price_map)

rest['Reviews'] = rest['Reviews'].apply(lambda x: x.split('],')[0].replace('[','').replace("'", "") if isinstance(x, str) else '')

rest.reset_index(drop=True, inplace=True)
rest.drop(columns=['ID_Old'], inplace=True, errors='ignore')

rest = rest.iloc[:-100000]

rest.to_csv('cleaned_Europe_Restaurants.csv', index=False)
rest.head()


# In[62]:


C= rest['Rating'].mean()
C


# In[63]:


m= rest['Count'].quantile(0.9) #movies having vote count greater than 90% from the list will be taken
m


# In[64]:


lists_restaurants = rest.copy().loc[rest['Count'] >= m]
lists_restaurants.shape


# In[65]:


def weighted_rating(x, m=m, C=C):
    v = x['Count']
    R = x['Rating']
    # Calculation based on the IMDB formula (m=1838, c=6.09)
    return (v/(v+m) * R) + (m/(m+v) * C)


# In[66]:


# Define a new feature 'score' and calculate its value with `weighted_rating()`
lists_restaurants['score'] = lists_restaurants.apply(weighted_rating, axis=1)


# In[67]:


lists_restaurants.head(3)


# In[68]:


lists_restaurants.shape


# In[69]:


#Sort movies based on score calculated above
lists_restaurants = lists_restaurants.sort_values('score', ascending=False)

#Print the top 10 movies
lists_restaurants[['Name', 'Count', 'Rating', 'score']].head(10)


# In[70]:


pop= rest.sort_values('Price', ascending=False)
import matplotlib.pyplot as plt
plt.figure(figsize=(6,2))

plt.barh(pop['Name'].head(6),pop['Price'].head(6), align='center',
        color='lightgreen')
plt.gca().invert_yaxis()
plt.xlabel("Price")
plt.title("Price Range" )


# In[71]:


rest.columns


# In[72]:


rest['Reviews'].head(10)


# In[73]:


from sklearn.feature_extraction.text import TfidfVectorizer

#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
rest['Reviews'] = rest['Reviews'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(rest['Reviews'])

#Output the shape of tfidf_matrix
tfidf_matrix.shape


# In[74]:


# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

from scipy.sparse import csr_matrix


# Convert tfidf_matrix to csr_matrix
tfidf_matrix_sparse = csr_matrix(tfidf_matrix)

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

#Construct a reverse map of indices and movie titles
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


# In[80]:


# print(get_content_recommendations('Viana',10))


# # In[81]:


# print(get_content_recommendations('Aleria',10))


# In[ ]:

st.markdown("# content based Recommender systems")

with st.form(key='my_form'):
    st.text_input(label='enter name of restaurant:',key= 'Name')
    submit = st.form_submit_button(label='recommend')

if submit :
    recommendation = get_content_recommendations(st.session_state.Name,top_n = 10, cosine_sim=cosine_sim)
    st.write(recommendation)


