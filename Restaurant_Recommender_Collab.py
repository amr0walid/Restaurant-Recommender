#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Data processing
import pandas as pd
import numpy as np
import scipy.stats

# Visualization
import seaborn as sns
import streamlit as st

# Similarity
from sklearn.metrics.pairwise import cosine_similarity


# In[4]:


user = pd.read_csv('ratings.csv')
user.rename(columns={'ID': 'RestID'}, inplace=True)
user.rename(columns={'userId': 'CustomerID'}, inplace=True)
user.head()


# In[5]:


# Number of Customer
print('The ratings dataset has', user['CustomerID'].nunique(), 'unique Customer')

# Number of rest
print('The ratings dataset has', user['RestID'].nunique(), 'unique Rest')

# Number of ratings
print('The ratings dataset has', user['rating'].nunique(), 'unique ratings')

# List of unique ratings
print('The unique ratings are', sorted(user['rating'].unique()))


# In[8]:


rest = pd.read_csv('Europe_Restaurants.csv')

rest['RestID'] = rest.index+1
rest.rename_axis('ID_Old', inplace=True)

rest.rename(columns={'Cuisine Style': 'Cuisine', 'Number of Reviews': 'Quantity', 'Price Range': 'Price'}, inplace=True)
rest.drop(columns=['Unnamed: 0', 'URL_TA', 'ID_TA', 'Ranking', 'Quantity', 'Cuisine', 'Price', 'Reviews', 'City'], inplace=True)


rest.reset_index(drop=True, inplace=True)
rest.drop(columns=['ID_Old'], inplace=True, errors='ignore')

rest.to_csv('cleaned_Europe_Restaurants.csv', index=False)


# In[9]:


df = pd.merge(rest, user, on='RestID', how='inner')



# In[10]:


# Aggregate by rest
agg_ratings = df.groupby('Name').agg(mean_rating = ('rating', 'mean'),
                                                number_of_ratings = ('rating', 'count')).reset_index()

# Keep the rest with over 50 ratings
agg_ratings_GT50 = agg_ratings[agg_ratings['number_of_ratings']>50]
agg_ratings_GT50.info()


# In[11]:


agg_ratings_GT50.sort_values(by='number_of_ratings', ascending=False).head()


# In[12]:


sns.jointplot(x='mean_rating', y='number_of_ratings', data=agg_ratings_GT50)


# In[13]:


df_GT50 = pd.merge(df, agg_ratings_GT50[['Name']], on='Name', how='inner')
df_GT50.info()


# In[14]:


# Number of Customer
print('The ratings dataset has', df_GT50['CustomerID'].nunique(), 'unique Customer')

# Number of rest
print('The ratings dataset has', df_GT50['RestID'].nunique(), 'unique Rest')

# Number of ratings
print('The ratings dataset has', df_GT50['rating'].nunique(), 'unique ratings')

# List of unique ratings
print('The unique ratings are', sorted(df_GT50['rating'].unique()))


# In[15]:


matrix = df_GT50.pivot_table(index='CustomerID', columns='Name', values='rating')


# In[16]:


# Normalize user-item matrix
matrix_norm = matrix.subtract(matrix.mean(axis=1), axis = 'rows')


# In[17]:


# User similarity matrix using Pearson correlation
Customer_similarity = matrix_norm.T.corr()


# In[18]:


# User similarity matrix using cosine similarity
Customer_similarity_cosine = cosine_similarity(matrix_norm.fillna(0))


# In[19]:


st.markdown("# collaborative recommender")

with st.form(key='my_form'):
    st.text_input(label='enter picked customer id:',key= 'picked_CustomerID')
    submit = st.form_submit_button(label='recommend')



# In[24]:

def get_bycollab(picked_CustomerID, matrix, top_r=10):
    # Remove picked user ID from the candidate list
    Customer_similarity.drop(index=picked_CustomerID, inplace=True)

# Reset the index after dropping rows
    Customer_similarity.reset_index(drop=True, inplace=True)

# Take a look at the data   
    Customer_similarity.head()


# In[20]:


# Number of similar Customer
    n = 10

# User similarity threashold
    user_similarity_threshold = 0.3

# Get top n similar Customer
    similar_Customer = Customer_similarity[Customer_similarity[picked_CustomerID]>user_similarity_threshold][picked_CustomerID].sort_values(ascending=False)[:n]

# Print out top n Customer
    print(f'The similar users for user {picked_CustomerID} are', similar_Customer)


# In[21]:


    picked_similar_CustomerID_visited= matrix_norm[matrix_norm.index == picked_CustomerID].dropna(axis=1, how='all')

# In[22]:


# rest that similar users visited. Remove rest that none of the similar Customer have visited
    similar_Customer_rest = matrix_norm[matrix_norm.index.isin(similar_Customer.index)].dropna(axis=1, how='all')


# In[23]:


# Remove the visited rest from the rest list
    similar_Customer_rest.drop(picked_similar_CustomerID_visited.columns,axis=1, inplace=True, errors='ignore')

# Take a look at the data


    # A dictionary to store item scores
    item_score = {}

    # Loop through rest
    for i in similar_Customer_rest.columns:
        # Get the ratings for rest i
        rest_rating = similar_Customer_rest[i]
        # Create a variable to store the score
        total = 0
        # Create a variable to store the number of scores
        count = 0
        # Loop through similar Customer
        for u in similar_Customer.index:
            # If the rest has rating
            if pd.isna(rest_rating[u]) == False:
                # Score is the sum of Customer similarity score multiply by the rest rating
                score = similar_Customer[u] * rest_rating[u]
                # Add the score to the total score for the rest so far
                total += score
                # Add 1 to the count
                count +=1
        # Get the average score for the rest
        item_score[i] = total / count

    # Convert dictionary to pandas dataframe
    item_score = pd.DataFrame(item_score.items(), columns=['rest', 'rest_score'])

    # Sort the rest by score
    ranked_item_score = item_score.sort_values(by='rest_score', ascending=False)

    # Calculate the average rating for the picked Customer
    avg_rating = matrix[matrix.index == picked_CustomerID].T.mean()[picked_CustomerID]

    # Calculate the predicted rating
    ranked_item_score['predicted_rating'] = ranked_item_score['rest_score'] + avg_rating

    # Select top r rest
    top_r_rest = ranked_item_score.head(top_r)

    return top_r_rest

# print(get_bycollab(4, similar_Customer, similar_Customer_rest, matrix, top_r=10))


if submit :
    recommendation = get_bycollab(int(st.session_state.picked_CustomerID),matrix,top_r=10)
    st.write(recommendation)