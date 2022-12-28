import textacy
import joblib
from textblob import TextBlob
from spacy.lang.en import English
from geotext import GeoText
from geopy.geocoders import Nominatim
from collections import defaultdict
from nltk.tokenize import word_tokenize, sent_tokenize
from spacy.lang.en.stop_words import STOP_WORDS
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from spacy_summarization import text_summarizer
import nltk
import re

nlp = English()
import en_core_web_sm
nlp = en_core_web_sm.load()

parser = English()
stopwords = ["1qfy23","eu","t","and","s","â€", "0o", "0s", "3a", "3b", "3d", "6b", "6o", "a", "a1", "a2", "a3", "a4", "ab", "able", "about", "above", "abst", "ac", "accordance", "according", "accordingly", "across", "act", "actually", "ad", "added", "adj", "ae", "af", "affected", "affecting", "affects", "after", "afterwards", "ag", "again", "against", "ah", "ain", "ain't", "aj", "al", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", "anyone", "anything", "anyway", "anyways", "anywhere", "ao", "ap", "apart", "apparently", "appear", "appreciate", "appropriate", "approximately", "ar", "are", "aren", "arent", "aren't", "arise", "around", "as", "a's", "aside", "ask", "asking", "associated", "at", "au", "auth", "av", "available", "aw", "away", "awfully", "ax", "ay", "az", "b", "b1", "b2", "b3", "ba", "back", "bc", "bd", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "begin", "beginning", "beginnings", "begins", "behind", "being", "believe", "below", "beside", "besides", "best", "better", "between", "beyond", "bi", "bill", "biol", "bj", "bk", "bl", "bn", "both", "bottom", "bp", "br", "brief", "briefly", "bs", "bt", "bu", "but", "bx", "by", "c", "c1", "c2", "c3", "ca", "call", "came", "can", "cannot", "cant", "can't", "cause", "causes", "cc", "cd", "ce", "certain", "certainly", "cf", "cg", "ch", "changes", "ci", "cit", "cj", "cl", "clearly", "cm", "c'mon", "cn", "co", "com", "come", "comes", "con", "concerning", "consequently", "consider", "considering", "contain", "containing", "contains", "corresponding", "could", "couldn", "couldnt", "couldn't", "course", "cp", "cq", "cr", "cry", "cs", "c's", "ct", "cu", "currently", "cv", "cx", "cy", "cz", "d", "d2", "da", "date", "dc", "dd", "de", "definitely", "describe", "described", "despite", "detail", "df", "di", "did", "didn", "didn't", "different", "dj", "dk", "dl", "do", "does", "doesn", "doesn't", "doing", "don", "done", "don't", "down", "downwards", "dp", "dr", "ds", "dt", "du", "due", "during", "dx", "dy", "e", "e2", "e3", "ea", "each", "ec", "ed", "edu", "ee", "ef", "effect", "eg", "ei", "eight", "eighty", "either", "ej", "el", "eleven", "else", "elsewhere", "em", "empty", "en", "end", "ending", "enough", "entirely", "eo", "ep", "eq", "er", "es", "especially", "est", "et", "et-al", "etc", "eu", "ev", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "ey", "f", "f2", "fa", "far", "fc", "few", "ff", "fi", "fifteen", "fifth", "fify", "fill", "find", "fire", "first", "five", "fix", "fj", "fl", "fn", "fo", "followed", "following", "follows", "for", "former", "formerly", "forth", "forty", "found", "four", "fr", "from", "front", "fs", "ft", "fu", "full", "further", "furthermore", "fy", "g", "ga", "gave", "ge", "get", "gets", "getting", "gi", "give", "given", "gives", "giving", "gj", "gl", "go", "goes", "going", "gone", "got", "gotten", "gr", "greetings", "gs", "gy", "h", "h2", "h3", "had", "hadn", "hadn't", "happens", "hardly", "has", "hasn", "hasnt", "hasn't", "have", "haven", "haven't", "having", "he", "hed", "he'd", "he'll", "hello", "help", "hence", "her", "here", "hereafter", "hereby", "herein", "heres", "here's", "hereupon", "hers", "herself", "hes", "he's", "hh", "hi", "hid", "him", "himself", "his", "hither", "hj", "ho", "home", "hopefully", "how", "howbeit", "however", "how's", "hr", "hs", "http", "hu", "hundred", "hy", "i", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ibid", "ic", "id", "i'd", "ie", "if", "ig", "ignored", "ih", "ii", "ij", "il", "i'll", "im", "i'm", "immediate", "immediately", "importance", "important", "in", "inasmuch", "inc", "indeed", "index", "indicate", "indicated", "indicates", "information", "inner", "insofar", "instead", "interest", "into", "invention", "inward", "io", "ip", "iq", "ir", "is", "isn", "isn't", "it", "itd", "it'd", "it'll", "its", "it's", "itself", "iv", "i've", "ix", "iy", "iz", "j", "jj", "jr", "js", "jt", "ju", "just", "k", "ke", "keep", "keeps", "kept", "kg", "kj", "km", "know", "known", "knows", "ko", "l", "l2", "la", "largely", "last", "lately", "later", "latter", "latterly", "lb", "lc", "le", "least", "les", "less", "lest", "let", "lets", "let's", "lf", "like", "liked", "likely", "line", "little", "lj", "ll", "ll", "ln", "lo", "look", "looking", "looks", "los", "lr", "ls", "lt", "ltd", "m", "m2", "ma", "made", "mainly", "make", "makes", "many", "may", "maybe", "me", "mean", "means", "meantime", "meanwhile", "merely", "mg", "might", "mightn", "mightn't", "mill", "million", "mine", "miss", "ml", "mn", "mo", "more", "moreover", "most", "mostly", "move", "mr", "mrs", "ms", "mt", "mu", "much", "mug", "must", "mustn", "mustn't", "my", "myself", "n", "n2", "na", "name",
             "namely", "nay", "nc", "nd", "ne", "near", "nearly", "necessarily", "necessary", "need", "needn", "needn't", "needs", "neither", "never", "nevertheless", "new", "next", "ng", "ni", "nine", "ninety", "nj", "nl", "nn", "no", "nobody", "non", "none", "nonetheless", "noone", "nor", "normally", "nos", "not", "noted", "nothing", "novel", "now", "nowhere", "nr", "ns", "nt", "ny", "o", "oa", "ob", "obtain", "obtained", "obviously", "oc", "od", "of", "off", "often", "og", "oh", "oi", "oj", "ok", "okay", "ol", "old", "om", "omitted", "on", "once", "one", "ones", "only", "onto", "oo", "op", "oq", "or", "ord", "os", "ot", "other", "others", "otherwise", "ou", "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "ow", "owing", "own", "ox", "oz", "p", "p1", "p2", "p3", "page", "pagecount", "pages", "par", "part", "particular", "particularly", "pas", "past", "pc", "pd", "pe", "per", "perhaps", "pf", "ph", "pi", "pj", "pk", "pl", "placed", "please", "plus", "pm", "pn", "po", "poorly", "possible", "possibly", "potentially", "pp", "pq", "pr", "predominantly", "present", "presumably", "previously", "primarily", "probably", "promptly", "proud", "provides", "ps", "pt", "pu", "put", "py", "q", "qj", "qu", "que", "quickly", "quite", "qv", "r", "r2", "ra", "ran", "rather", "rc", "rd", "re", "readily", "really", "reasonably", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", "research", "research-articl", "respectively", "resulted", "resulting", "results", "rf", "rh", "ri", "right", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "run", "rv", "ry", "s", "s2", "sa", "said", "same", "saw", "say", "saying", "says", "sc", "sd", "se", "sec", "second", "secondly", "section", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sensible", "sent", "serious", "seriously", "seven", "several", "sf", "shall", "shan", "shan't", "she", "shed", "she'd", "she'll", "shes", "she's", "should", "shouldn", "shouldn't", "should've", "show", "showed", "shown", "showns", "shows", "si", "side", "significant", "significantly", "similar", "similarly", "since", "sincere", "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so", "some", "somebody", "somehow", "someone", "somethan", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "sp", "specifically", "specified", "specify", "specifying", "sq", "sr", "ss", "st", "still", "stop", "strongly", "sub", "substantially", "successfully", "such", "sufficiently", "suggest", "sup", "sure", "sy", "system", "sz", "t", "t1", "t2", "t3", "take", "taken", "taking", "tb", "tc", "td", "te", "tell", "ten", "tends", "tf", "th", "than", "thank", "thanks", "thanx", "that", "that'll", "thats", "that's", "that've", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "thered", "therefore", "therein", "there'll", "thereof", "therere", "theres", "there's", "thereto", "thereupon", "there've", "these", "they", "theyd", "they'd", "they'll", "theyre", "they're", "they've", "thickv", "thin", "think", "third", "this", "thorough", "thoroughly", "those", "thou", "though", "thoughh", "thousand", "three", "throug", "through", "throughout", "thru", "thus", "ti", "til", "tip", "tj", "tl", "tm", "tn", "to", "together", "too", "took", "top", "toward", "towards", "tp", "tq", "tr", "tried", "tries", "truly", "try", "trying", "ts", "t's", "tt", "tv", "twelve", "twenty", "twice", "two", "tx", "u", "u201d", "ue", "ui", "uj", "uk", "um", "un", "under", "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "uo", "up", "upon", "ups", "ur", "us", "use", "used", "useful", "usefully", "usefulness", "uses", "using", "usually", "ut", "v", "va", "value", "various", "vd", "ve", "ve", "very", "via", "viz", "vj", "vo", "vol", "vols", "volumtype", "vq", "vs", "vt", "vu", "w", "wa", "want", "wants", "was", "wasn", "wasnt", "wasn't", "way", "we", "wed", "we'd", "welcome", "well", "we'll", "well-b", "went", "were", "we're", "weren", "werent", "weren't", "we've", "what", "whatever", "what'll", "whats", "what's", "when", "whence", "whenever", "when's", "where", "whereafter", "whereas", "whereby", "wherein", "wheres", "where's", "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", "whoever", "whole", "who'll", "whom", "whomever", "whos", "who's", "whose", "why", "why's", "wi", "widely", "will", "willing", "wish", "with", "within", "without", "wo", "won", "wonder", "wont", "won't", "words", "world", "would", "wouldn", "wouldnt", "wouldn't", "www", "x", "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y", "y2", "yes", "yet", "yj", "yl", "you", "youd", "you'd", "you'll", "your", "youre", "you're", "yours", "yourself", "yourselves", "you've", "yr", "ys", "yt", "z", "zero", "zi", "zz", ] + list(STOP_WORDS)

def Dup_Function(x):
    return list(dict.fromkeys(x))

def News_POS( news_text):
    news_subjectivity = Get_Subjectivity(news_text)
    news_polarity = Get_Polarity(news_text)
    news_sentiment = Get_Analysis(news_polarity)
    tokenized = sent_tokenize(news_text)

    tag = []
    for j in tokenized:
        wordsList = nltk.word_tokenize(j)
        wordsList = [w for w in wordsList if not w in stopwords]
        tagged = nltk.pos_tag(wordsList)
        tag.append(tagged)
    nouns = []
    for k in tag:
        for l in range(len(k)):
            if k[l][1] == 'NN' or k[l][1] == 'NNPS':
                nouns.append(k[l][0])
    news_nouns = Dup_Function(nouns)
    proper_noun = []
    for m in tag:
        for n in range(len(m)):
            if m[n][1] == 'NNP' or m[n][1] == 'NNPS':
                proper_noun.append(m[n][0])
    proper_noun = Dup_Function(proper_noun)
    complete_nouns = nouns+proper_noun
    news_complete_nouns = Dup_Function(complete_nouns)
    verbs = []
    for o in tag:
        for p in range(len(o)):
            if o[p][1] == 'VB' or o[p][1] == 'VBD' or o[p][1] == 'VBG' or o[p][1] == 'VBN' or o[p][1] == 'VBP' or o[p][1] == 'VBZ':
                verbs.append(o[p][0])
    news_verbs = Dup_Function(verbs)
    cardinal_digit = []
    for q in tag:
        for r in range(len(q)):
            if q[r][1] == 'CD':
                cardinal_digit.append(q[r][0])
    news_cardinal_digit = Dup_Function(cardinal_digit)

    return news_subjectivity, news_polarity, news_sentiment, news_complete_nouns, news_cardinal_digit, news_verbs

# def Event_Extraction(soup, news_short_desc, **kwargs):
#     #     pip install textacy
#     pattern = [{'POS': 'VERB'}]
#     doc = nlp(news_short_desc)
#     news_events = textacy.extract.token_matches(doc, patterns=pattern)
#     # print(" Event In News ", events)
#     news_events = ' '.join([str(elem) for elem in news_events])
#     return news_events

def Data_Cleaning(text):

    # split into sentences
    words = word_tokenize(text)
    words = [word for word in words if word.isalpha()]

    # remove stop words in sentence
    stop_words = set(stopwords)
    words = [w for w in words if not w in stop_words]
    # print(words[:100])

    # please it comment if you don't want to use Lemmatizer
    # lemmatizing of words
    lmtzr = WordNetLemmatizer()
    words = [lmtzr.lemmatize(word) for word in words]
    # print(lemmt[:100])

    # stemming of words
    porter = PorterStemmer()
    words = [porter.stem(word) for word in words]
    return (" ".join(str(x) for x in words))

def News_Classifier( news_text):
    news_content_clean = []
    news_content_clean.append(Data_Cleaning(news_text))
    news_content_clean[0]
    tf_load_vec = joblib.load(
        'news_classification_model_tf_vectorizer.pkl')
    model = joblib.load('news_classification_model.pkl')
    extract = tf_load_vec.transform(news_content_clean)
    prediction = model.predict(extract)
    prediction = prediction[0].capitalize()
    # print("News Category: ", prediction)
    return prediction

def Get_Subjectivity( text):
    """_summary_sentences_to_text( sentences):

    Args:
        text (_type_): The text to be analyzed.

    Returns:
        _type_: The subjectivity of the text.
    """
    return TextBlob(text).sentiment.subjectivity

def Get_Polarity( text):
    """_summary_ sentences_polarity( sentences, sentence_polarity):

    Args:       
        text (_type_): The text to be analyzed. 

    Returns:
        _type_: The polarity of the text in form of percentage.
    """

    return TextBlob(text).sentiment.polarity

def Get_Analysis( score, **kwargs):
    if score < -0.5 and score >= -0.7:
        return "Very Negative"
    elif score == 0 and score < 0.1:
        return "Neutral"
    elif score > 0.7:
        return "Very Postive"
    elif score > 0.5 and score <= 0.7:
        return "Positive"
    elif score > 0.3 and score <= 0.5:
        return "Slightly Positive"
    elif score > 0 and score <= 0.3:
        return "Weakly Positive"
    elif score < -0.7:
        return "Very Negative"
    elif score < -0.3 and score >= -0.5:
        return "Slightly Negative"
    elif score < 0 and score >= -0.3:
        return "Weakly Negative"

def Geographic_Details( text):
    """_summary_ : This function will return the geographic details of the news article     

    Args:
        text (string): this function will take an input of the news article text

    Returns:
        _type_: The return of this function will return the geographic details of the news article like country, city, latitude, longitude, etc.
    """
    places = GeoText(text)
    # contry = (','.join(str(a)for a in places.countries))
    # city = (','.join(str(a)for a in places.cities))
    try:
        country = places.countries
        temp = defaultdict(int)
        for sub in country:
            for wrd in sub.split():
                temp[wrd] += 1
        country = max(temp, key=temp.get)
    except:
        country = " "
    try:
        city = places.cities
        temp = defaultdict(int)
        for sub in city:
            for wrd in sub.split():
                temp[wrd] += 1
        city = max(temp, key=temp.get)
    except:
        city = " "
    try:
        geolocator = Nominatim(user_agent="abc")
        location = geolocator.geocode(city, language='en')
        address = location.address
        latitude = location.latitude
        longitude = location.longitude
    except:
        address = " "
        latitude = " "
        longitude = " "
    return country, city, address, latitude, longitude

def Get_Words( text):
    # Extract words from a text file. Clean the words by removing surrounding
    # punctuation and whitespace, and convert the word to singular.
    words = text.replace("\n", " ")
    words = parser.convert_abbreviations(words)
    words = words.split(" ")
    words = parser.remove_blanks(words)
    for i in range(0, len(words)):
        words[i] = parser.clean(words[i])
    return words

def Event_Extraction(soup, news_short_desc,  **kwargs):
    try:
        news_events = soup.find(
            "meta",  {"name": "keywords"}).attrs['content']
    except:
        #     pip install textacy
        pattern = [{'POS': 'VERB'}]
        doc = nlp(news_short_desc)
        news_events = textacy.extract.token_matches(doc, patterns=pattern)
        # print(" Event In News ", events)
        news_events = ' '.join([str(elem) for elem in news_events])
    return news_events

def News_Events(short_description):
    """
    This function will return the events of the News from the given soup object.
    """
    # Extract words from the text file.
    words = Get_Words(short_description)
    # Get the list of events from the words.
    events = parser.get_events(words)
    # Print the events.
    return events

def News_Target_Names(news_details, **kwargs):
    target_names = nlp(news_details)
    target_names = [(X.text, X.label_) for X in target_names.ents]
    target_names = [x for x in target_names if 'PERSON' in x[1]]
    target_names = ', '.join([tup[0] for tup in target_names])
    return target_names

def Remove_non_Ascii(string):
    # text = string.encode('cp1252', errors='ignore').decode('utf8')
    text = string.encode('ascii', errors='ignore').decode('utf8')
    return text

def Count_Text_Words(news_text, **kwargs):
    news_words = [i for i in news_text.lower().split() if i not in stopwords]
    news_words = Dup_Function(news_words)
    # news_words = [i for i in word_tokenize(news_text.lower()) if i not in stopwords] 
    clean_text = (" ").join(news_words)
    clean_text = Remove_non_Ascii(clean_text)
    news_words = [i for i in news_text.lower().split() if i not in stopwords]
    news_words_count = len(news_words)
    
    # news_word_cloud = WordCloud(collocations = False, background_color = 'white').generate(clean_text)
    word_cloud = news_words.copy()
    # dic = news_word_cloud.words_
    # for key, value in dic.items():
    #     word_cloud.append(key)
    return news_words_count, word_cloud

def Article_Summary(text, **kwargs):
    news_text = text.replace("\"", "")
    news_text = re.sub(r'(?!(([^"]*"){2})*[^"]*$),', '', news_text)
    news_text = news_text.replace("'", " ")
    news_text = news_text.replace("\n", "")
    news_text = news_text.replace('"', '')
    if news_text is None:
        news_text = 'No Detail Found'
    else:
        news_summary = text_summarizer(news_text).replace("\n", "")
    return news_summary































