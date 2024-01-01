import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
df = pd.read_csv('"C:\\Users\\Aditya Thakur\\Dropbox\\PC\\Downloads\\Health_Recommendation_System1.csv"')

# Combine relevant features into a single column
features = ['disease_name', 'medicine', 'food_recommendation', 'diet_plan', 'exercise']
df['combined_features'] = df[features].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Transform the dataset into TF-IDF vectors
X = vectorizer.fit_transform(df['combined_features'])

# Function to get recommendations based on user symptoms
def get_recommendations(user_symptoms):
    # Vectorize user's symptoms
    user_symptoms_vector = vectorizer.transform([user_symptoms])

    # Calculate cosine similarity between user's symptoms vector and dataset
    similarities = cosine_similarity(user_symptoms_vector, X)

    # Get indices of top recommendations
    top_indices = similarities.argsort()[0][::-1][:10]

    # Extract top recommendations from the dataset
    top_recommendations = df.iloc[top_indices, :]

    return top_recommendations

# Get user input for symptoms
user_symptoms = input('Enter your symptoms: ')

# Get recommendations based on user symptoms
recommendations = get_recommendations(user_symptoms)

# Print the recommendations
print('Top 10 recommendations:')
for index, row in recommendations.iterrows():
    print(f'- Disease Name: {row["disease_name"]}')
    print(f'- Medicine: {row["medicine"]}')
    print(f'- Food Recommendation: {row["food_recommendation"]}')
    print(f'- Diet Plan: {row["diet_plan"]}')
    print(f'- Exercise: {row["exercise"]}')
    print('---')
