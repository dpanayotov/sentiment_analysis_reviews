import re
import nltk
from nltk.corpus import stopwords

lemmatizer = nltk.WordNetLemmatizer()
stemmer = nltk.PorterStemmer()

genres = ['genre', "animation", "adventure", "comedy", 'cartoon', "action", "family", "romance", "drama", "crime",
          "thriller", "fantasy", "horror", "biography", "history", "mystery", "sci-fi", "war", "sport", "music",
          "documentary", "musical", "western", "short", "film-noir", "talk-show", "news", "adult", "reality-tv",
          "game-show", "indie", 'romantic', 'comic', 'comics', 'comedic', 'fiction', 'science', 'science-fiction',
          'romantic', 'romance', 'comedy-drama']

common_words = ['performance', 'performances', 'story', 'could', 'would', 'screen', 'plot', 'narrative', 'script',
                'picture', 'writer', 'writers', 'character', 'characters', 'director', 'acting', 'actor', 'actress',
                'actresses', 'actors', 'cinema', 'cinematic', 'cinematography', 'storytelling', 'play', 'plays',
                'screenplay', 'role', 'roles', 'scene', 'scenes', 'show', 'see', 'saw', 'seen', 'scenario',
                'video', 'viewer', 'viewers', 'storyline', 'storylines', 'audiences', 'audience', 'watch', 'artist',
                'watched', 'watching', 'comedian', 'anyway', 'book', 'title', 'film', 'movie', 'movies',
                'films', 'filmmaker', 'filmmaking', 'moviegoer', 'moviemaker', 'filmgoer', 'filmgoers', 'moviemaking',
                'moviegoer', 'tale', 'screenwriter']

stop_words = stopwords.words('english')
stop_words.extend(genres)
stop_words.extend(common_words)

def pre_process(comment_text):
    comment_text = re.sub(" n't", "n't", comment_text)
    comment_text = re.sub(" 's", "", comment_text)
    comment_text = re.sub(" 'd", "'d", comment_text)
    comment_text = re.sub(" 're", "'re", comment_text)
    comment_text = re.sub("-LRB-", "(", comment_text)
    comment_text = re.sub("-RRB-", ")", comment_text)
    comment_text = re.sub('\W', ' ', comment_text)
    comment_text = re.sub('\s+', ' ', comment_text)
    comment_text = re.sub(r'[0-9]+', '', comment_text)
    tokenizer = nltk.WhitespaceTokenizer()
    tokens = tokenizer.tokenize(comment_text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    stemmed = nltk.tag.pos_tag(tokens)
    no_stopwords = []
    for token, tag in stemmed:
        token = token.lower()
        if tag != 'NNP' and tag != 'NNPS' and token not in stop_words and len(token) > 2:
            no_stopwords.append(token)

    processed = ' '.join(no_stopwords)
    processed = processed.strip(' ')
    return processed
