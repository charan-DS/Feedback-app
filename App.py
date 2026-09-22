import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

raw_data = {
    'Months_Subscribed' : [1, 2, 3, 4, 5],
    'Total_Movies_Watched' : [10, 20, 30, 40, 50],
    'Chat_Support_Feedback' : ['hated it bad', 'worst app error', 'ok decent', 'good liked it', 'best service love'],
    'Feedback_Sentiment' : [0, 0, 1, 1, 1]
}

df = pd.DataFrame(raw_data)

X = df[['Months_Subscribed']]
y = df['Total_Movies_Watched']

model = LinearRegression()
model.fit(X,y)

predicted_spend = model.predict([[3.5]])

Vectorizer = CountVectorizer()
X_nlp = Vectorizer.fit_transform(df['Chat_Support_Feedback'])

y_nlp = [0,0,1,1,1]

nlp_model = MultinomialNB()
nlp_model.fit(X_nlp,y_nlp)

new_feedback = ['Bad App i Hate it']
new_feedback_transformed = Vectorizer.transform(new_feedback)
predicted_sentiment = nlp_model.predict(new_feedback_transformed)

print(predicted_sentiment)