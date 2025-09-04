import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
import string

# Download stopwords if not already available
nltk.download('stopwords')

# Load dataset
df = pd.read_csv("3) Sentiment dataset.csv")

print("Columns in dataset:", df.columns)

# --- Text Preprocessing Function ---
def preprocess(text):
    text = text.lower()  # lowercase
    text = "".join([ch for ch in text if ch not in string.punctuation])  # remove punctuation
    stop_words = set(stopwords.words("english"))
    text = " ".join([word for word in text.split() if word not in stop_words])  # remove stopwords
    return text

# Apply preprocessing on 'Text' column
df['cleaned'] = df['Text'].astype(str).apply(preprocess)

# --- WordCloud ---
all_text = " ".join(df['cleaned'])
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("WordCloud of Sentiment Texts")
plt.savefig("wordcloud.png")   # Save image
plt.show()

# --- Sentiment Distribution ---
plt.figure(figsize=(6, 4))
df['Sentiment'].value_counts().plot(kind='bar', color=['skyblue', 'orange', 'green'])
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.savefig("sentiment_distribution.png")   # Save image
plt.show()

# --- Scatter Plot: Retweets vs Likes ---
if 'Retweets' in df.columns and 'Likes' in df.columns:
    plt.figure(figsize=(6, 4))
    plt.scatter(df['Retweets'], df['Likes'], alpha=0.5, c='blue')
    plt.title("Retweets vs Likes Scatter Plot")
    plt.xlabel("Retweets")
    plt.ylabel("Likes")
    plt.savefig("retweets_vs_likes.png")   # Save image
    plt.show()
else:
    print("Retweets or Likes columns not available in dataset.")
