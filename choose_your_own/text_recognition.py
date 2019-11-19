from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer


sentence1 = "This is a test"
sentence2 = "Hello Anto, I would like to buy a cat"
sentence3 = "Dear Mr. Soru, I think it would be important for you to buy a cat"

email = [ sentence1, sentence2, sentence3]

text = CountVectorizer()
text.fit(email)
text.analyzer

sw = stopwords.words("english")
print(len(sw))

stemmer = SnowballStemmer("english")
print(stemmer.stem("responsiveness"))
print(stemmer.stem("unresponsive"))