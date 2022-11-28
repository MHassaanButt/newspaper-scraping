# import the required modules and libraries
import re
from sympy import elliptic_f
import textacy
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from collections import Counter
from spacy import displacy
from textblob import TextBlob
import en_core_web_sm
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
from wordcloud import WordCloud
from spacy.language import Language
from datetime import datetime, time
import re
import requests
from bs4 import BeautifulSoup, SoupStrainer
import pandas as pd
# nltk.download()
from nltk import word_tokenize
# from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from newspaper import Article
import newspaper
from datetime import date as d
from newspaper import Config
from nltk.tag import pos_tag
from spacy.lang.en.stop_words import STOP_WORDS
from gensim.parsing.preprocessing import remove_stopwords
import time as t
from string import punctuation
import string
from spacy.lang.en import English
from models.spacy_summarization import text_summarizer
from geotext import GeoText
from geopy.geocoders import Nominatim
from collections import Counter
from collections import defaultdict
from pathlib import Path
# from practice import news_authors, news_title
punctuations = string.punctuation
nlp = English()
nlp = en_core_web_sm.load()
# stop_words = set(stopwords.words('english'))
nlp.add_pipe('sentencizer')  # updated

parser = English()
stopwords = ["1qfy23", "eu", "t", "and", "s", "â€", "0o", "0s", "3a", "3b", "3d", "6b", "6o", "a", "a1", "a2", "a3", "a4", "ab", "able", "about", "above", "abst", "ac", "accordance", "according", "accordingly", "across", "act", "actually", "ad", "added", "adj", "ae", "af", "affected", "affecting", "affects", "after", "afterwards", "ag", "again", "against", "ah", "ain", "ain't", "aj", "al", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", "anyone", "anything", "anyway", "anyways", "anywhere", "ao", "ap", "apart", "apparently", "appear", "appreciate", "appropriate", "approximately", "ar", "are", "aren", "arent", "aren't", "arise", "around", "as", "a's", "aside", "ask", "asking", "associated", "at", "au", "auth", "av", "available", "aw", "away", "awfully", "ax", "ay", "az", "b", "b1", "b2", "b3", "ba", "back", "bc", "bd", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "begin", "beginning", "beginnings", "begins", "behind", "being", "believe", "below", "beside", "besides", "best", "better", "between", "beyond", "bi", "bill", "biol", "bj", "bk", "bl", "bn", "both", "bottom", "bp", "br", "brief", "briefly", "bs", "bt", "bu", "but", "bx", "by", "c", "c1", "c2", "c3", "ca", "call", "came", "can", "cannot", "cant", "can't", "cause", "causes", "cc", "cd", "ce", "certain", "certainly", "cf", "cg", "ch", "changes", "ci", "cit", "cj", "cl", "clearly", "cm", "c'mon", "cn", "co", "com", "come", "comes", "con", "concerning", "consequently", "consider", "considering", "contain", "containing", "contains", "corresponding", "could", "couldn", "couldnt", "couldn't", "course", "cp", "cq", "cr", "cry", "cs", "c's", "ct", "cu", "currently", "cv", "cx", "cy", "cz", "d", "d2", "da", "date", "dc", "dd", "de", "definitely", "describe", "described", "despite", "detail", "df", "di", "did", "didn", "didn't", "different", "dj", "dk", "dl", "do", "does", "doesn", "doesn't", "doing", "don", "done", "don't", "down", "downwards", "dp", "dr", "ds", "dt", "du", "due", "during", "dx", "dy", "e", "e2", "e3", "ea", "each", "ec", "ed", "edu", "ee", "ef", "effect", "eg", "ei", "eight", "eighty", "either", "ej", "el", "eleven", "else", "elsewhere", "em", "empty", "en", "end", "ending", "enough", "entirely", "eo", "ep", "eq", "er", "es", "especially", "est", "et", "et-al", "etc", "eu", "ev", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "ey", "f", "f2", "fa", "far", "fc", "few", "ff", "fi", "fifteen", "fifth", "fify", "fill", "find", "fire", "first", "five", "fix", "fj", "fl", "fn", "fo", "followed", "following", "follows", "for", "former", "formerly", "forth", "forty", "found", "four", "fr", "from", "front", "fs", "ft", "fu", "full", "further", "furthermore", "fy", "g", "ga", "gave", "ge", "get", "gets", "getting", "gi", "give", "given", "gives", "giving", "gj", "gl", "go", "goes", "going", "gone", "got", "gotten", "gr", "greetings", "gs", "gy", "h", "h2", "h3", "had", "hadn", "hadn't", "happens", "hardly", "has", "hasn", "hasnt", "hasn't", "have", "haven", "haven't", "having", "he", "hed", "he'd", "he'll", "hello", "help", "hence", "her", "here", "hereafter", "hereby", "herein", "heres", "here's", "hereupon", "hers", "herself", "hes", "he's", "hh", "hi", "hid", "him", "himself", "his", "hither", "hj", "ho", "home", "hopefully", "how", "howbeit", "however", "how's", "hr", "hs", "http", "hu", "hundred", "hy", "i", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ibid", "ic", "id", "i'd", "ie", "if", "ig", "ignored", "ih", "ii", "ij", "il", "i'll", "im", "i'm", "immediate", "immediately", "importance", "important", "in", "inasmuch", "inc", "indeed", "index", "indicate", "indicated", "indicates", "information", "inner", "insofar", "instead", "interest", "into", "invention", "inward", "io", "ip", "iq", "ir", "is", "isn", "isn't", "it", "itd", "it'd", "it'll", "its", "it's", "itself", "iv", "i've", "ix", "iy", "iz", "j", "jj", "jr", "js", "jt", "ju", "just", "k", "ke", "keep", "keeps", "kept", "kg", "kj", "km", "know", "known", "knows", "ko", "l", "l2", "la", "largely", "last", "lately", "later", "latter", "latterly", "lb", "lc", "le", "least", "les", "less", "lest", "let", "lets", "let's", "lf", "like", "liked", "likely", "line", "little", "lj", "ll", "ll", "ln", "lo", "look", "looking", "looks", "los", "lr", "ls", "lt", "ltd", "m", "m2", "ma", "made", "mainly", "make", "makes", "many", "may", "maybe", "me", "mean", "means", "meantime", "meanwhile", "merely", "mg", "might", "mightn", "mightn't", "mill", "million", "mine", "miss", "ml", "mn", "mo", "more", "moreover", "most", "mostly", "move", "mr", "mrs", "ms", "mt", "mu", "much", "mug", "must", "mustn", "mustn't", "my", "myself", "n", "n2", "na", "name",
             "namely", "nay", "nc", "nd", "ne", "near", "nearly", "necessarily", "necessary", "need", "needn", "needn't", "needs", "neither", "never", "nevertheless", "new", "next", "ng", "ni", "nine", "ninety", "nj", "nl", "nn", "no", "nobody", "non", "none", "nonetheless", "noone", "nor", "normally", "nos", "not", "noted", "nothing", "novel", "now", "nowhere", "nr", "ns", "nt", "ny", "o", "oa", "ob", "obtain", "obtained", "obviously", "oc", "od", "of", "off", "often", "og", "oh", "oi", "oj", "ok", "okay", "ol", "old", "om", "omitted", "on", "once", "one", "ones", "only", "onto", "oo", "op", "oq", "or", "ord", "os", "ot", "other", "others", "otherwise", "ou", "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "ow", "owing", "own", "ox", "oz", "p", "p1", "p2", "p3", "page", "pagecount", "pages", "par", "part", "particular", "particularly", "pas", "past", "pc", "pd", "pe", "per", "perhaps", "pf", "ph", "pi", "pj", "pk", "pl", "placed", "please", "plus", "pm", "pn", "po", "poorly", "possible", "possibly", "potentially", "pp", "pq", "pr", "predominantly", "present", "presumably", "previously", "primarily", "probably", "promptly", "proud", "provides", "ps", "pt", "pu", "put", "py", "q", "qj", "qu", "que", "quickly", "quite", "qv", "r", "r2", "ra", "ran", "rather", "rc", "rd", "re", "readily", "really", "reasonably", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", "research", "research-articl", "respectively", "resulted", "resulting", "results", "rf", "rh", "ri", "right", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "run", "rv", "ry", "s", "s2", "sa", "said", "same", "saw", "say", "saying", "says", "sc", "sd", "se", "sec", "second", "secondly", "section", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sensible", "sent", "serious", "seriously", "seven", "several", "sf", "shall", "shan", "shan't", "she", "shed", "she'd", "she'll", "shes", "she's", "should", "shouldn", "shouldn't", "should've", "show", "showed", "shown", "showns", "shows", "si", "side", "significant", "significantly", "similar", "similarly", "since", "sincere", "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so", "some", "somebody", "somehow", "someone", "somethan", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "sp", "specifically", "specified", "specify", "specifying", "sq", "sr", "ss", "st", "still", "stop", "strongly", "sub", "substantially", "successfully", "such", "sufficiently", "suggest", "sup", "sure", "sy", "system", "sz", "t", "t1", "t2", "t3", "take", "taken", "taking", "tb", "tc", "td", "te", "tell", "ten", "tends", "tf", "th", "than", "thank", "thanks", "thanx", "that", "that'll", "thats", "that's", "that've", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "thered", "therefore", "therein", "there'll", "thereof", "therere", "theres", "there's", "thereto", "thereupon", "there've", "these", "they", "theyd", "they'd", "they'll", "theyre", "they're", "they've", "thickv", "thin", "think", "third", "this", "thorough", "thoroughly", "those", "thou", "though", "thoughh", "thousand", "three", "throug", "through", "throughout", "thru", "thus", "ti", "til", "tip", "tj", "tl", "tm", "tn", "to", "together", "too", "took", "top", "toward", "towards", "tp", "tq", "tr", "tried", "tries", "truly", "try", "trying", "ts", "t's", "tt", "tv", "twelve", "twenty", "twice", "two", "tx", "u", "u201d", "ue", "ui", "uj", "uk", "um", "un", "under", "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "uo", "up", "upon", "ups", "ur", "us", "use", "used", "useful", "usefully", "usefulness", "uses", "using", "usually", "ut", "v", "va", "value", "various", "vd", "ve", "ve", "very", "via", "viz", "vj", "vo", "vol", "vols", "volumtype", "vq", "vs", "vt", "vu", "w", "wa", "want", "wants", "was", "wasn", "wasnt", "wasn't", "way", "we", "wed", "we'd", "welcome", "well", "we'll", "well-b", "went", "were", "we're", "weren", "werent", "weren't", "we've", "what", "whatever", "what'll", "whats", "what's", "when", "whence", "whenever", "when's", "where", "whereafter", "whereas", "whereby", "wherein", "wheres", "where's", "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", "whoever", "whole", "who'll", "whom", "whomever", "whos", "who's", "whose", "why", "why's", "wi", "widely", "will", "willing", "wish", "with", "within", "without", "wo", "won", "wonder", "wont", "won't", "words", "world", "would", "wouldn", "wouldnt", "wouldn't", "www", "x", "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y", "y2", "yes", "yet", "yj", "yl", "you", "youd", "you'd", "you'll", "your", "youre", "you're", "yours", "yourself", "yourselves", "you've", "yr", "ys", "yt", "z", "zero", "zi", "zz", ] + list(STOP_WORDS)


class RT():
    """
    This class will be used to scrap the news from the given url.
    """

    def __init__(self,):
        pass
        # self.url = url

    def HTML_Document(self, url):
        # request for HTML document of given url
        agent = {
            "User-Agent": 'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36'}
        # request for HTML document of given url
        try:
            response = requests.get(url, headers=agent, verify=True)
            html_document = response.text
            return html_document
        except:
            return "Internet Connection Error"

    def B_Soup(self, html_document):
        soup = BeautifulSoup(html_document, 'html.parser')
        # print("This is thewsoup", soup)
        # soup.prettify()
        return soup

    def News_Title(self, soup):
        try:
            title = soup.find("title").get_text(strip=True)
            # print("Titlw is Found",title)
            return title
            # title = soup.find("meta",  {"property": "og:title"})[
            #     'content']
            # if title:
            #     # title = title.split(' – RT – ')[0]
            #     # print(title)
            #     return title
            # else:
            #     title = soup.find('h2', class_=['story__title', 'text-7.5', 'font-bold', 'font-playfair-display', 'mt-1', 'pb-3', 'border-b', 'border-gray-300', 'border-solid',
            #                       'text-6', 'sm:text-10.5', 'text-center', 'text-black-400 hover:text-pink-default', 'leading-tight', 'mt-2', 'sm:mt-8', 'pb-4  ']).get_text(strip=True)
            #     # title = soup.find('h2').get_text(strip=True)
        except:
            title = "No Title"

        return title

    def News_Time(self, soup):
        try:
            # format = '%H:%M:%S'
            # time_now = datetime.now()
            # time_now = time_now.strftime("%H:%M:%S")
            time = soup.find('span', class_=[
                "date date_article-header"]).get_text(strip=True)
            time = time.split(" ")[3]

            # if "hour" in time_scraped:
            #     time_scraped = time_scraped.split(' ')[0]
            #     time_scraped = str(time_scraped+':00:00')
            #     time = datetime.strptime(time_now, format) - \
            #         datetime.strptime(time_scraped, format)
            #     time = time.split(',')[1]

            #     # time = time.split(' ')[1]
            #     # time = time.strftime("%H:%M:%S")
            #     return time
            # elif "min" in time_scraped:
            #     time_scraped = time_scraped.split(' ')[0]
            #     time_scraped = str('00:'+time_scraped+':00')
            # #     print(time_scraped)

            #     time = datetime.strptime(time_now, format) - \
            #         datetime.strptime(time_scraped, format)
            #     # time = time.strftime("%H:%M:%S")
            #     return time
            # else:
            #     print("Invalid Time")
            # time = time.strftime("%H:%M:%S")
            # # print("--------Time ---------------------", time)
            return time

        except:

            time_now = datetime.now()
            time_now = time_now.strftime("%H:%M:%S")
            # print("--------Time Now---------------------", time_now)
            return time_now

    def News_Date(self, soup):
        try:
            date = soup.find('span', class_=[
                "date date_article-header"]).get_text(strip=True)
            date = date[:-6]
            return date

            # cur_date = soup.find('time').text
            # print("-------------------------------------------------------------------")
            # print('----------------', cur_date,)
            # if "ago" in cur_date:
            #     today = d.today()
            #     today = today.strftime("%m/%d/%Y")
            #     return today
            # else:
            #     return cur_date

        except:
            return "No Date"

        # date = re.sub('\s+', ' ', date)[0:11]
        # date = soup.find('div', attrs={'class': "dateTime secTime"}).get_text(strip=True)[13:]

    def News_Source(self, soup):
        try:
            source = soup.find("meta",  {"name": "author"}).attrs['content']
            if source:
                return source
            else:
                source = soup.find(
                    'a', class_=['story__byline__link']).get_text(strip=True)
        except:
            source = "No Source"
        return source

    def News_Source_Link(self, soup):
        try:
            source_link = soup.find(
                "meta",  {"property": "article:author"})['content']
            if source_link:
                return source_link
            else:
                source_link = soup.find(
                    'div', class_=['article-top-author-nw-nf-left']).get('href')
            # source_link = source_link
        except:
            source_link = "No Source Link"
        return source_link

    def Short_Description(self, soup):
        try:
            short_desc = soup.find("meta",  {"property": "og:description"})[
                'content']
        except:
            short_desc = "No Short Description"
        return short_desc

    def Image_URL(self, soup):
        try:
            images = soup.find_all('div', class_=["media__item"])
            image_src = soup.find("meta",  {"property": "og:image"})['content']
            if image_src:
                return image_src
            else:
                if not images:
                    image_src = None
                else:
                    for img in images:
                        if img.find('img') is not None:
                            image_src = img.find('img')['src']
                            return image_src
                        else:
                            image_src = None
        except:
            image_src = "No Source Link"
        return image_src

    def Image_Alt_Text(self, soup):
        try:
            images_alt = soup.find(
                "div",   class_=["media__footer media__footer_bottom"]).get_text(strip=True)
            return images_alt
        except:
            images_alt = None
            return images_alt

    def News_Detail(self, soup):
        # try:
        article_text = ''
        try:
            definition = soup.find(
                'div', class_=['storyDetails', 'story-detail'])
            for p in definition.find_all('p'):
                article_text = article_text + p.get_text(strip=True)
            # article = soup.find('story-detail').find_all('p').text
            # for paragraph in article.find_all('p'):
            #     article_text += paragraph.text
            article_text = article_text.replace("\n", " ")
            article_text = article_text.replace("\xa0", " ")
            article_text = article_text.replace('?', '')
        except:
            article = soup.findAll('p')
            for element in article:
                article_text += '\n' + ''.join(element.findAll(text=True))
                article_text = article_text.replace("\xa0", " ")
                article_text = article_text.replace('?', '')
        # print(article_text)
        return article_text

    def Short_Description(self, soup):
        try:
            short_desc = soup.find("meta",  {"property": "og:description"})[
                'content']
        except:
            short_desc = soup.find(
                'h2', attrs={'class': 'summary-desc'}).get_text(strip=True)
        return short_desc

    def News_Events(self, short_description):
        """
        This function will return the events of the News from the given soup object.
        """
        # Extract words from the text file.
        words = self.Get_Words(short_description)
        # Get the list of events from the words.
        events = parser.get_events(words)
        # Print the events.
        return events

    def News_POS(self, news_text):
        news_subjectivity = self.Get_Subjectivity(news_text)
        news_polarity = self.Get_Polarity(news_text)
        news_sentiment = self.Get_Analysis(news_polarity)
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
        news_nouns = self.Dup_Function(nouns)
        proper_noun = []
        for m in tag:
            for n in range(len(m)):
                if m[n][1] == 'NNP' or m[n][1] == 'NNPS':
                    proper_noun.append(m[n][0])
        proper_noun = self.Dup_Function(proper_noun)
        complete_nouns = nouns+proper_noun
        news_complete_nouns = self.Dup_Function(complete_nouns)
        verbs = []
        for o in tag:
            for p in range(len(o)):
                if o[p][1] == 'VB' or o[p][1] == 'VBD' or o[p][1] == 'VBG' or o[p][1] == 'VBN' or o[p][1] == 'VBP' or o[p][1] == 'VBZ':
                    verbs.append(o[p][0])
        news_verbs = self.Dup_Function(verbs)
        cardinal_digit = []
        for q in tag:
            for r in range(len(q)):
                if q[r][1] == 'CD':
                    cardinal_digit.append(q[r][0])
        news_cardinal_digit = self.Dup_Function(cardinal_digit)

        return news_subjectivity, news_polarity, news_sentiment, news_complete_nouns, news_cardinal_digit, news_verbs

    def Scrap_Articles_URL(self, link):
        USER_AGENT = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:78.0) Gecko/20100101 Firefox/78.0'
        config = Config()
        config.browser_user_agent = USER_AGENT
        config.request_timeout = 10
        article_urls = set()
        marketwatch = newspaper.build(
            link, config=config, memoize_articles=False, language='en')
        for sub_article in marketwatch.articles:
            article = Article(sub_article.url, config=config,
                              memoize_articles=False, language='en')
            article.download()
            article.parse()
            if article.url not in article_urls:
                article_urls.add(article.url)
                # print(article.url, '\n')
        return article_urls

    def Get_Words(self, text):
        # Extract words from a text file. Clean the words by removing surrounding
        # punctuation and whitespace, and convert the word to singular.
        words = text.replace("\n", " ")
        words = parser.convert_abbreviations(words)
        words = words.split(" ")
        words = parser.remove_blanks(words)
        for i in range(0, len(words)):
            words[i] = parser.clean(words[i])
        return words

    def Dup_Function(self, x):

        return list(dict.fromkeys(x))

    def Get_Subjectivity(self, text):
        """_summary_sentences_to_text(self, sentences):

        Args:
            text (_type_): The text to be analyzed.

        Returns:
            _type_: The subjectivity of the text.
        """
        return TextBlob(text).sentiment.subjectivity

    def Get_Polarity(self, text):
        """_summary_ sentences_polarity(self, sentences, sentence_polarity):

        Args:       
            text (_type_): The text to be analyzed. 

        Returns:
            _type_: The polarity of the text in form of percentage.
        """

        return TextBlob(text).sentiment.polarity

    def Get_Analysis(self, score, **kwargs):
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

    def Count_Text_Words(self, news_text, **kwargs):
        news_words = [i for i in news_text.lower().split()
                      if i not in stopwords]
        news_words = self.Dup_Function(news_words)
        # news_words = [i for i in word_tokenize(news_text.lower()) if i not in stopwords]
        clean_text = (" ").join(news_words)
        # clean_text = self.Remove_non_Ascii(clean_text)
        news_words = [i for i in news_text.lower().split()
                      if i not in stopwords]
        news_words_count = len(news_words)

        # news_word_cloud = WordCloud(collocations = False, background_color = 'white').generate(clean_text)
        word_cloud = news_words.copy()
        # dic = news_word_cloud.words_
        # for key, value in dic.items():
        #     word_cloud.append(key)
        return news_words_count, word_cloud

    def News_Target_Names(self, news_details, **kwargs):
        target_names = nlp(news_details)
        target_names = [(X.text, X.label_) for X in target_names.ents]
        target_names = [x for x in target_names if 'PERSON' in x[1]]
        target_names = ', '.join([tup[0] for tup in target_names])
        return target_names

    def Article_Summary(self, soup):
        news_image_link = self.Image_URL(soup)
        news_text = self.News_Detail(soup)
        news_text = news_text.replace("\"", "")
        news_text = re.sub(r'(?!(([^"]*"){2})*[^"]*$),', '', news_text)
        news_text = news_text.replace("'", " ")
        news_text = news_text.replace("\n", "")
        news_text = news_text.replace('"', '')
        if news_text is not None:
            try:
                news_summary = text_summarizer(news_text).replace("\n", "")
                total_news_words, news_word_cloud = self.Count_Text_Words(
                    news_text)
                total_summary_words, summary_word_cloud = self.Count_Text_Words(
                    news_summary)
            except:
                news_summary = 'No Summary'
                total_news_words = 0
                news_word_cloud = []
                total_summary_words = 0
                summary_word_cloud = []
        return news_text, news_summary, total_news_words, total_summary_words, news_image_link, news_word_cloud, summary_word_cloud

    def Data_Cleaning(self, text):

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

    def Event_Extraction(self, news_short_desc, **kwargs):
        #     pip install textacy
        pattern = [{'POS': 'VERB'}]
        doc = nlp(news_short_desc)
        news_events = textacy.extract.token_matches(doc, patterns=pattern)
        # print(" Event In News ", events)
        news_events = ' '.join([str(elem) for elem in news_events])
        return news_events

    def News_Classifier(self, news_text):
        news_content_clean = []
        news_content_clean.append(self.Data_Cleaning(news_text))
        news_content_clean[0]
        tf_load_vec = joblib.load(
            'models/news_classification_model_tf_vectorizer.pkl')
        model = joblib.load('models/news_classification_model.pkl')
        extract = tf_load_vec.transform(news_content_clean)
        prediction = model.predict(extract)
        prediction = prediction[0].capitalize()
        # print("News Category: ", prediction)
        return prediction

    def News_Section(self, soup):
        try:
            news_section = soup.find_all(
                'div', attrs={'class': 'sc-bBrHrO sc-llJcti bTxTGD fJQkxy sc-gfbRpc gtRDfy kicker sc-hbyLVd jqlWTM'})
            for sec in news_section:
                for span in sec.find_all('span'):
                    print(span.text)
                    return span.text
        except:
            news_section = "No Section"
        return news_section

    def News_Classifier(self, news_text):
        news_content_clean = []
        news_content_clean.append(self.Data_Cleaning(news_text))
        news_content_clean[0]
        tf_load_vec = joblib.load(
            'models/news_classification_model_tf_vectorizer.pkl')
        model = joblib.load('models/news_classification_model.pkl')
        extract = tf_load_vec.transform(news_content_clean)
        prediction = model.predict(extract)
        prediction = prediction[0].capitalize()
        # print("News Category: ", prediction)
        return prediction

    def Geographic_Details(self, text):
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

    def Scrap_Links(self, url):
        main_url = 'https://www.rt.com/'
        html_document = self.HTML_Document(url)
        #print("-----------------------------------------url", url)
        if html_document is not None:
            soup = self.B_Soup(html_document)
            # print(soup)
            all_links = []
            all_links = []
            print("-------------------all links")
            className = 'main-promobox__item'
            tag = 'li'
            print(url)
            if 'news' in url:
                print("In russia")
                className = 'listCard-rows__content'
                tag = 'div'
            elif 'russia' in url:
                className = 'listCard-rows__content'
                tag = 'div'

            print(tag, className)
            # 'full-light-container'
            results = soup.find_all(
                tag, class_=[className, ])
            # results_links = [i for i in results if i is not None]
            for div in results:
                links = div.findAll('a', href=True)

                # a_tag.append(links)
                for a in links:
                    #print("HREF and its lenth -------------------->",a['href'], len(a['href']))
                    if a['href'] and len(a['href']) > 30:
                        link = main_url + a['href']
                        all_links.append(link)
                        # print(link)

                    else:
                        continue

            all_links = [i for i in set(all_links)]
            return all_links
        else:
            print('connection error')
            return None

    def Scrap_RT(self, url, name, **kwargs):
        df_news = {"News_URL": [], "News_Source": [], "News_Title": [],  "News_Date": [], "News_Time": [], "News_Authors": [], "News_Authors_Source": [],  "News_Image_Link": [], "News_Image_Caption": [],
                   "News_Proper_Nouns": [], "News_Verbs": [], "News_Cardinal_Digit": [], "News_Target_Names": [], "News_Total_Words": [], "News_Total_Summary_Words": [],
                   'News_Word_Cloud': [], 'News_Summary_Word_Cloud': [], "News_Short_Description": [], "News_Detail": [], "News_Summary": [], "News_Polarity_Score": [],
                   "News_Subjectivity": [], "News_Sentiments": [], "News_Classification": [], "News_Section": [], "News_Event": [], "News_Country": [], "News_City": [],
                   "News_Address": [], "News_Latitude": [], "News_Longitude": []}
        # extract all links using the above function
        print("-------------------------------Scrapping RT Russia NewsPaper-------------------------------")
        i = 0
        article_urls = self.Scrap_Links(url)
        if article_urls is not None:
            print(
                "-------------------------------Scrapping All Links from current Page----------------------------------")
            print(
                "-------------------------------Totatl URL's Scrapped from Requested Link: ", len(article_urls), '---------------------')
            for url in article_urls:  # [:5]
                print("Scraping Sub URL # ", i, "\nLink :", url)
                html = self.HTML_Document(url)
                if html is not None:
                    soup = self.B_Soup(html)
                    news_title = self.News_Title(soup)
                    news_authors = self.News_Source(soup)
                    news_authors_source = self.News_Source_Link(soup)
                    news_time = self.News_Time(soup)
                    news_date = self.News_Date(soup)
                    # try:
                    #     txt = news_date
                    #     news_date = re.search("\d\d/\d\d/\d\d\d\d", txt)[0]
                    # except:
                    #     pass
                    news_short_desc = self.Short_Description(soup)
                    news_image_text = self.Image_Alt_Text(soup)
                    news_section = self.News_Section(soup)
                    news_text, news_summary, total_news_words, total_summary_words, news_image_link, news_words, summary_words = self.Article_Summary(
                        soup)
                    news_subjectivity, news_polarity, news_sentiment, news_complete_nouns, news_cardinal_digit, news_verbs = self.News_POS(
                        news_text)
                    target_names = self.News_Target_Names(news_summary)
                    news_classification = self.News_Classifier(news_text)
                    news_event = self.Event_Extraction(news_short_desc)
                    country, city, address, latitude, longitude = self.Geographic_Details(
                        news_text)
                    print('**********************************')
                    print(f'News URL: {url}\n')
                    print(f'News Source: {name}\n')
                    print(f'News Section: {news_section}\n')
                    print(f'News Title: {news_title}\n')
                    print(f'News Author: {news_authors}\n')
                    print(
                        f'News Author Source link: {news_authors_source}\n')
                    print(f'News Publish Date: {news_date}\n')
                    print(f'News Publish Time: {news_time}\n')
                    print(f'News Short Description: {news_short_desc}\n')
                    print(f'News Image Alt Text: {news_image_text}\n')
                    print(f'News Top Image URL: {news_image_link}\n')
                    print(f'News Complete Nouns:  {news_complete_nouns}\n')
                    print(f'News Cardinal Digit:  {news_cardinal_digit}\n')
                    print(
                        f'News Targeted Persons in News:  {target_names}\n')
                    print(
                        f'News Total Words in News Details:  {total_news_words}\n')
                    print(
                        f'News Total Words in News Summary:  {total_summary_words}\n')
                    print(f'News Article:  {news_text}\n')
                    print(f'News Summary:  {news_summary}\n')
                    print(
                        f'News Sentiments WRT Subjectivity:  {news_subjectivity}\n')
                    print(f'News Sentiment Score:  {news_polarity}\n')
                    print(f'News Sentiment Anaylsis:  {news_sentiment}\n')
                    print(f'News Classification:  {news_classification}\n')
                    print(f'News Events:  {news_event}\n')
                    print(f'News Country:  {country}\n')
                    print(f'News City:  {city}\n')
                    print(f'News Address:  {address}\n')
                    print(f'News Latitude:  {latitude}\n')
                    print(f'News Longitude:  {longitude}\n')
                    print('**********************************')

                    df_news["News_URL"].append(url)
                    df_news["News_Source"].append(name)
                    df_news["News_Section"].append(news_section)
                    df_news["News_Title"].append(news_title)
                    df_news["News_Date"].append(news_date)
                    df_news["News_Time"].append(news_time)
                    df_news["News_Authors"].append(news_authors)
                    df_news["News_Authors_Source"].append(
                        news_authors_source)
                    df_news["News_Image_Link"].append(news_image_link)
                    df_news["News_Image_Caption"].append(news_image_text)
                    df_news["News_Short_Description"].append(
                        news_short_desc)
                    df_news["News_Proper_Nouns"].append(
                        news_complete_nouns)
                    df_news["News_Verbs"].append(news_verbs)
                    df_news["News_Cardinal_Digit"].append(
                        news_cardinal_digit)
                    df_news["News_Target_Names"].append(target_names)
                    df_news["News_Word_Cloud"].append(news_words)
                    df_news["News_Summary_Word_Cloud"].append(
                        summary_words)
                    df_news["News_Total_Words"].append(total_news_words)
                    df_news["News_Total_Summary_Words"].append(
                        total_summary_words)
                    df_news["News_Detail"].append(news_text)
                    df_news["News_Summary"].append(news_summary)
                    df_news["News_Subjectivity"].append(news_subjectivity)
                    df_news["News_Polarity_Score"].append(news_polarity)
                    df_news["News_Sentiments"].append(news_sentiment)
                    df_news["News_Classification"].append(
                        news_classification)
                    df_news["News_Event"].append(news_event)
                    df_news["News_Country"].append(country)
                    df_news["News_City"].append(city)
                    df_news["News_Address"].append(address)
                    df_news["News_Latitude"].append(latitude)
                    df_news["News_Longitude"].append(longitude)

                    i = i+1
                else:
                    print(
                        "No HTML Document for Current Link -----> Access Denied")
                    print("System Sleeping Mode on for New Request!!! %s" %
                          t.ctime())
                    t.sleep(50)
                    print("Cheecking for New Request!!! %s" % t.ctime())
                    continue

            # except:
            #     #     # print("Oops! authors not found", sys.exc_info()[0], "occurred.")
            #     continue
        print("Scraping Completed for the Current News Source!!!")

        df = pd.DataFrame.from_dict(df_news)
        return df_news

    def scrap_latest_news(self, article_url,  **kwargs):
        print("-------------------------------Scraping Latest Tab News-------------------------------")
        df_latest = self.Scrap_RT(article_url, 'Latest')
        # df_latest = pd.DataFrame.from_dict(df_latest)
        return df_latest

    def scrap_national_news(self,  article_url, **kwargs):
        print("-------------------------------Scraping National Tab News-------------------------------")
        df_national = self.Scrap_RT(article_url, 'National')
        # df_national = pd.DataFrame.from_dict(df_national)
        return df_national

    def scrap_world_news(self,  article_url, **kwargs):
        print("-------------------------------Scraping International Tab News-------------------------------")
        df_national = self.Scrap_RT(article_url, 'International')
        # df_national = pd.DataFrame.from_dict(df_news)
        return df_national


if __name__ == "__main__":
    print("-------------------------------Scraping RT Russia News -------------------------------")
    scrap = RT()
    latest = 'https://www.rt.com/'
    national = 'https://www.rt.com/russia/'

    world = 'https://www.rt.com/news/'
#
    scrap_latest = scrap.scrap_latest_news(latest)
    scrap_national = scrap.scrap_national_news(national)
    scrap_world = scrap.scrap_world_news(world)

    df_latest = pd.DataFrame.from_dict(scrap_latest)
    df_national = pd.DataFrame.from_dict(scrap_national)
    df_world = pd.DataFrame.from_dict(scrap_world)
    Path("RT").mkdir(parents=True, exist_ok=True)
    df_latest.to_csv('RT/RT_latest.csv', index=False)
    df_national.to_csv('RT/RT_national.csv', index=False)
    df_world.to_csv('RT/RT_world.csv', index=False)
    # df_news.head()
