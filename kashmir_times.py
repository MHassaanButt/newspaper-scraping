"""
In this script, we will be extracting the news from the given url. The news will be stored in a dataframe file.
The News are from the given url.
    -TheNews
    -Hindustan Times
    -BBC News
    -Economic Times
    -The Hindu
    -The Indian Express
    -The Times of India
    -The Tribune
    -The Wire
    -Eurasian Times, India
    -The New Indian Express
    -The Print, India



Input:
    url: The url of the news website
Output:
    news_df: The dataframe file of the news


"""

# import the required modules and libraries
import en_core_web_sm
from nltk.tokenize import word_tokenize, sent_tokenize
from datetime import datetime, date, time
import time
from time import ctime
import re
import requests
from bs4 import BeautifulSoup, SoupStrainer
import pandas as pd
# nltk.download()
from nltk import word_tokenize
# from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from spacy.lang.en.stop_words import STOP_WORDS
from gensim.parsing.preprocessing import remove_stopwords
import string
from spacy.lang.en import English
import utils as utils
# from practice import news_authors, news_title
punctuations = string.punctuation
nlp = English()
nlp = en_core_web_sm.load()
# stop_words = set(stopwords.words('english'))
nlp.add_pipe('sentencizer')  # updated
parser = English()
stopwords = ["1qfy23", "eu", "t", "and", "s", "â€", "0o", "0s", "3a", "3b", "3d", "6b", "6o", "a", "a1", "a2", "a3",
             "a4", "ab", "able", "about", "above", "abst", "ac", "accordance", "according", "accordingly", "across",
             "act", "actually", "ad", "added", "adj", "ae", "af", "affected", "affecting", "affects", "after",
             "afterwards", "ag", "again", "against", "ah", "ain", "ain't", "aj", "al", "all", "allow", "allows",
             "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst",
             "amount", "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", "anyone", "anything",
             "anyway", "anyways", "anywhere", "ao", "ap", "apart", "apparently", "appear", "appreciate", "appropriate",
             "approximately", "ar", "are", "aren", "arent", "aren't", "arise", "around", "as", "a's", "aside", "ask",
             "asking", "associated", "at", "au", "auth", "av", "available", "aw", "away", "awfully", "ax", "ay", "az",
             "b", "b1", "b2", "b3", "ba", "back", "bc", "bd", "be", "became", "because", "become", "becomes",
             "becoming", "been", "before", "beforehand", "begin", "beginning", "beginnings", "begins", "behind",
             "being", "believe", "below", "beside", "besides", "best", "better", "between", "beyond", "bi", "bill",
             "biol", "bj", "bk", "bl", "bn", "both", "bottom", "bp", "br", "brief", "briefly", "bs", "bt", "bu", "but",
             "bx", "by", "c", "c1", "c2", "c3", "ca", "call", "came", "can", "cannot", "cant", "can't", "cause",
             "causes", "cc", "cd", "ce", "certain", "certainly", "cf", "cg", "ch", "changes", "ci", "cit", "cj", "cl",
             "clearly", "cm", "c'mon", "cn", "co", "com", "come", "comes", "con", "concerning", "consequently",
             "consider", "considering", "contain", "containing", "contains", "corresponding", "could", "couldn",
             "couldnt", "couldn't", "course", "cp", "cq", "cr", "cry", "cs", "c's", "ct", "cu", "currently", "cv", "cx",
             "cy", "cz", "d", "d2", "da", "date", "dc", "dd", "de", "definitely", "describe", "described", "despite",
             "detail", "df", "di", "did", "didn", "didn't", "different", "dj", "dk", "dl", "do", "does", "doesn",
             "doesn't", "doing", "don", "done", "don't", "down", "downwards", "dp", "dr", "ds", "dt", "du", "due",
             "during", "dx", "dy", "e", "e2", "e3", "ea", "each", "ec", "ed", "edu", "ee", "ef", "effect", "eg", "ei",
             "eight", "eighty", "either", "ej", "el", "eleven", "else", "elsewhere", "em", "empty", "en", "end",
             "ending", "enough", "entirely", "eo", "ep", "eq", "er", "es", "especially", "est", "et", "et-al", "etc",
             "eu", "ev", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly",
             "example", "except", "ey", "f", "f2", "fa", "far", "fc", "few", "ff", "fi", "fifteen", "fifth", "fify",
             "fill", "find", "fire", "first", "five", "fix", "fj", "fl", "fn", "fo", "followed", "following", "follows",
             "for", "former", "formerly", "forth", "forty", "found", "four", "fr", "from", "front", "fs", "ft", "fu",
             "full", "further", "furthermore", "fy", "g", "ga", "gave", "ge", "get", "gets", "getting", "gi", "give",
             "given", "gives", "giving", "gj", "gl", "go", "goes", "going", "gone", "got", "gotten", "gr", "greetings",
             "gs", "gy", "h", "h2", "h3", "had", "hadn", "hadn't", "happens", "hardly", "has", "hasn", "hasnt",
             "hasn't", "have", "haven", "haven't", "having", "he", "hed", "he'd", "he'll", "hello", "help", "hence",
             "her", "here", "hereafter", "hereby", "herein", "heres", "here's", "hereupon", "hers", "herself", "hes",
             "he's", "hh", "hi", "hid", "him", "himself", "his", "hither", "hj", "ho", "home", "hopefully", "how",
             "howbeit", "however", "how's", "hr", "hs", "http", "hu", "hundred", "hy", "i", "i2", "i3", "i4", "i6",
             "i7", "i8", "ia", "ib", "ibid", "ic", "id", "i'd", "ie", "if", "ig", "ignored", "ih", "ii", "ij", "il",
             "i'll", "im", "i'm", "immediate", "immediately", "importance", "important", "in", "inasmuch", "inc",
             "indeed", "index", "indicate", "indicated", "indicates", "information", "inner", "insofar", "instead",
             "interest", "into", "invention", "inward", "io", "ip", "iq", "ir", "is", "isn", "isn't", "it", "itd",
             "it'd", "it'll", "its", "it's", "itself", "iv", "i've", "ix", "iy", "iz", "j", "jj", "jr", "js", "jt",
             "ju", "just", "k", "ke", "keep", "keeps", "kept", "kg", "kj", "km", "know", "known", "knows", "ko", "l",
             "l2", "la", "largely", "last", "lately", "later", "latter", "latterly", "lb", "lc", "le", "least", "les",
             "less", "lest", "let", "lets", "let's", "lf", "like", "liked", "likely", "line", "little", "lj", "ll",
             "ll", "ln", "lo", "look", "looking", "looks", "los", "lr", "ls", "lt", "ltd", "m", "m2", "ma", "made",
             "mainly", "make", "makes", "many", "may", "maybe", "me", "mean", "means", "meantime", "meanwhile",
             "merely", "mg", "might", "mightn", "mightn't", "mill", "million", "mine", "miss", "ml", "mn", "mo", "more",
             "moreover", "most", "mostly", "move", "mr", "mrs", "ms", "mt", "mu", "much", "mug", "must", "mustn",
             "mustn't", "my", "myself", "n", "n2", "na", "name",
             "namely", "nay", "nc", "nd", "ne", "near", "nearly", "necessarily", "necessary", "need", "needn",
             "needn't", "needs", "neither", "never", "nevertheless", "new", "next", "ng", "ni", "nine", "ninety", "nj",
             "nl", "nn", "no", "nobody", "non", "none", "nonetheless", "noone", "nor", "normally", "nos", "not",
             "noted", "nothing", "novel", "now", "nowhere", "nr", "ns", "nt", "ny", "o", "oa", "ob", "obtain",
             "obtained", "obviously", "oc", "od", "of", "off", "often", "og", "oh", "oi", "oj", "ok", "okay", "ol",
             "old", "om", "omitted", "on", "once", "one", "ones", "only", "onto", "oo", "op", "oq", "or", "ord", "os",
             "ot", "other", "others", "otherwise", "ou", "ought", "our", "ours", "ourselves", "out", "outside", "over",
             "overall", "ow", "owing", "own", "ox", "oz", "p", "p1", "p2", "p3", "page", "pagecount", "pages", "par",
             "part", "particular", "particularly", "pas", "past", "pc", "pd", "pe", "per", "perhaps", "pf", "ph", "pi",
             "pj", "pk", "pl", "placed", "please", "plus", "pm", "pn", "po", "poorly", "possible", "possibly",
             "potentially", "pp", "pq", "pr", "predominantly", "present", "presumably", "previously", "primarily",
             "probably", "promptly", "proud", "provides", "ps", "pt", "pu", "put", "py", "q", "qj", "qu", "que",
             "quickly", "quite", "qv", "r", "r2", "ra", "ran", "rather", "rc", "rd", "re", "readily", "really",
             "reasonably", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related",
             "relatively", "research", "research-articl", "respectively", "resulted", "resulting", "results", "rf",
             "rh", "ri", "right", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "run", "rv", "ry", "s",
             "s2", "sa", "said", "same", "saw", "say", "saying", "says", "sc", "sd", "se", "sec", "second", "secondly",
             "section", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sensible",
             "sent", "serious", "seriously", "seven", "several", "sf", "shall", "shan", "shan't", "she", "shed",
             "she'd", "she'll", "shes", "she's", "should", "shouldn", "shouldn't", "should've", "show", "showed",
             "shown", "showns", "shows", "si", "side", "significant", "significantly", "similar", "similarly", "since",
             "sincere", "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so", "some", "somebody", "somehow",
             "someone", "somethan", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry",
             "sp", "specifically", "specified", "specify", "specifying", "sq", "sr", "ss", "st", "still", "stop",
             "strongly", "sub", "substantially", "successfully", "such", "sufficiently", "suggest", "sup", "sure", "sy",
             "system", "sz", "t", "t1", "t2", "t3", "take", "taken", "taking", "tb", "tc", "td", "te", "tell", "ten",
             "tends", "tf", "th", "than", "thank", "thanks", "thanx", "that", "that'll", "thats", "that's", "that've",
             "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby",
             "thered", "therefore", "therein", "there'll", "thereof", "therere", "theres", "there's", "thereto",
             "thereupon", "there've", "these", "they", "theyd", "they'd", "they'll", "theyre", "they're", "they've",
             "thickv", "thin", "think", "third", "this", "thorough", "thoroughly", "those", "thou", "though", "thoughh",
             "thousand", "three", "throug", "through", "throughout", "thru", "thus", "ti", "til", "tip", "tj", "tl",
             "tm", "tn", "to", "together", "too", "took", "top", "toward", "towards", "tp", "tq", "tr", "tried",
             "tries", "truly", "try", "trying", "ts", "t's", "tt", "tv", "twelve", "twenty", "twice", "two", "tx", "u",
             "u201d", "ue", "ui", "uj", "uk", "um", "un", "under", "unfortunately", "unless", "unlike", "unlikely",
             "until", "unto", "uo", "up", "upon", "ups", "ur", "us", "use", "used", "useful", "usefully", "usefulness",
             "uses", "using", "usually", "ut", "v", "va", "value", "various", "vd", "ve", "ve", "very", "via", "viz",
             "vj", "vo", "vol", "vols", "volumtype", "vq", "vs", "vt", "vu", "w", "wa", "want", "wants", "was", "wasn",
             "wasnt", "wasn't", "way", "we", "wed", "we'd", "welcome", "well", "we'll", "well-b", "went", "were",
             "we're", "weren", "werent", "weren't", "we've", "what", "whatever", "what'll", "whats", "what's", "when",
             "whence", "whenever", "when's", "where", "whereafter", "whereas", "whereby", "wherein", "wheres",
             "where's", "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod",
             "whoever", "whole", "who'll", "whom", "whomever", "whos", "who's", "whose", "why", "why's", "wi", "widely",
             "will", "willing", "wish", "with", "within", "without", "wo", "won", "wonder", "wont", "won't", "words",
             "world", "would", "wouldn", "wouldnt", "wouldn't", "www", "x", "x1", "x2", "x3", "xf", "xi", "xj", "xk",
             "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y", "y2", "yes", "yet", "yj", "yl", "you", "youd", "you'd",
             "you'll", "your", "youre", "you're", "yours", "yourself", "yourselves", "you've", "yr", "ys", "yt", "z",
             "zero", "zi", "zz", ] + list(STOP_WORDS)


main_url = 'http://www.kashmirtimes.com/'
class KashmirTimes_News():
    """
    This class will be used to scrap the news from the given url.
    """

    # KashmirTimes
    def __init__(self, ):
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
        # soup.prettify()
        return soup

    def News_Title(self, soup):
        try:
            # title = soup.find("title", ).text
            title  = soup.title.get_text(strip=True)
            # if title:
            #     return title
            # else:
            #     title = soup.find('h2').get_text(strip=True)
        except:
            title = "No Title"
        return title

    def News_Time(self, soup):
        try:
            time = soup.find('p', class_=["newsdate"]).get_text()
            time = time.split(' ')[-2]
        except:
            now = datetime.now()
            time = now.strftime("%H:%M:%S")
        return time

    def News_Date(self, soup):
        try:
            news_date = soup.find('p', class_=["newsdate"]).get_text()
            news_date = news_date.split(' ')[-3]
        except:
            news_date = date.today()
        return news_date

    def News_Authors(self, soup):
        try:
            source = soup.find("meta", {"name": "author"}).attrs['content']
            if source:
                return source
            else:
                source = soup.find(
                    'a', class_=['story__byline__link']).get_text(strip=True)
        except:
            source = "No Source"
        return source

    def News_Author_Link(self, soup):
        try:
            source_link = soup.find(
                "meta", {"property": "article:author"})['content']
            if source_link:
                return source_link
            else:
                source_link = soup.find(
                    'div', class_=['article-top-author-nw-nf-left']).get('href')
            # source_link = source_link
        except:
            source_link = "No Source Link"
        return source_link

    def Image_URL(self, soup):
        try:
            image_src = soup.find('img', class_=["img-responsive", "pic"])
            print(image_src['src'])
        except:
            image_src = None
        return image_src

    def Image_Alt_Text(self, soup):

        try:
            image_src = soup.find('img', class_=["img-responsive", "pic"])
            print(image_src['src'])
        except:
            images_alt = None

        return images_alt


    def News_Detail(self, soup):
        # try:
        article_text = ''
        try:
            definition = soup.find('div', class_=['content count-br', 'story-detail'])
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
            short_desc = soup.find("meta", {"property": "og:description"})['content'].strip()
            if short_desc is not None:
                return short_desc
            else:
                short_desc = soup.find(
                    'h3', attrs={'class': 'preamble-nf'}).get_text(strip=True)
        except:
            short_desc = "No Short Description"
        return short_desc


    def News_Section(self, soup):
        try:
            news_section = soup.find(
                'div', attrs={'class': 'secName'}).get_text(strip=True).title()
        except:
            news_section = "No Section"
        return news_section


    def Scrap_World_Links(self, url):
        html_document = self.HTML_Document(url)
        if html_document is not None:
            soup = self.B_Soup(html_document)
            all_links = []

            results = soup.find_all('div', class_=['wpb_wrapper', ])  # 'full-light-container'
            # results = soup.find_all('div',
            #                         class_=['col-md-9 col-sm-12 col-xs-12 left-block', ])  # 'full-light-container'
            # results_links = [i for i in results if i is not None]
            for div in results:
                links = div.findAll('a', href=True)
                # a_tag.append(links)
                for a in links:
                    if a['href'] and len(a['href']) > 55:
                        link = a['href']
                        all_links.append(link)
                        print(link)
                    else:
                        continue
            all_links = [i for i in set(all_links)]
            print("all Links here:", all_links)
            return all_links

        else:
            print('connection error')
            return None

    def Scrap_Latest_Links(self, url):
        html_document = self.HTML_Document(url)
        if html_document is not None:
            soup = self.B_Soup(html_document)
            all_links = []

            # results = soup.find_all('div', class_=['col-md-8 col-sm-8 col-xs-12 left-block', ])#'full-light-container'
            results = soup.find_all('div',
                                    class_=[
                                        'col-lg-4 no1 topnews', ])  # 'col-lg-8 no right-sep main'
            # results_links = [i for i in results if i is not None]
            for div in results:
                links = div.findAll('a', href=True)
                # a_tag.append(links)
                for a in links:
                    if a['href'] and len(a['href']) > 15:
                        link = main_url + a['href']
                        all_links.append(link)
                        print(link)
                    else:
                        continue
            all_links = [i for i in set(all_links)]
            print("all Links here:", all_links)
            return all_links

        else:
            print('connection error')
            return None

    def Scrap_National_Links(self, url):
        html_document = self.HTML_Document(url)
        if html_document is not None:
            soup = self.B_Soup(html_document)
            all_links = []

            results = soup.find_all('div', class_=['wpb_wrapper', ])  # 'full-light-container'
            # results = soup.find_all('div',
            #                         class_=['col-md-9 col-sm-12 col-xs-12 left-block', ])  # 'full-light-container'
            # results_links = [i for i in results if i is not None]
            for div in results:
                links = div.findAll('a', href=True)
                # a_tag.append(links)
                for a in links:
                    if a['href'] and len(a['href']) > 60:
                        link = a['href']
                        all_links.append(link)
                        print(link)
                    else:
                        continue
            all_links = [i for i in set(all_links)]
            print("all Links here:", all_links)
            return all_links

        else:
            print('connection error')
            return None

    def Scrap_KashmirTimes_News(self, article_urls, name, **kwargs):
        df_news = {"News_URL": [], "News_Source": [], "News_Title": [], "News_Date": [], "News_Time": [],
                   "News_Authors": [], "News_Authors_Source": [], "News_Image_Link": [], "News_Image_Caption": [],
                   "News_Proper_Nouns": [], "News_Verbs": [], "News_Cardinal_Digit": [], "News_Target_Names": [],
                   "News_Total_Words": [], "News_Total_Summary_Words": [],
                   'News_Word_Cloud': [], 'News_Summary_Word_Cloud': [], "News_Short_Description": [],
                   "News_Detail": [], "News_Summary": [], "News_Polarity_Score": [],
                   "News_Subjectivity": [], "News_Sentiments": [], "News_Classification": [], "News_Section": [],
                   "News_Event": [], "News_Country": [], "News_City": [],
                   "News_Address": [], "News_Latitude": [], "News_Longitude": []}
        # extract all links using the above function
        # article_urls = self.Scrap_Links(link)
        print("-------------------------------URL Scrapped from Requested Link-------------------------------")
        print("-------------------------------Totatl URL's Scrapped from Requested Link: ", len(article_urls))
        i = 0
        for url in article_urls:
            print("Scraping URL # ", i, "\nLink :", url)
            i += 1
            html = self.HTML_Document(url)
            soup = self.B_Soup(html)
            news_title = self.News_Title(soup)
            news_authors = self.News_Authors(soup)
            news_authors_link = self.News_Author_Link(soup)
            news_time = self.News_Time(soup)
            news_date = self.News_Date(soup)
            news_short_desc = self.Short_Description(soup)
            news_image_link = self.Image_URL(soup)
            news_image_text = self.Image_Alt_Text(soup)
            news_section = self.News_Section(soup)
            news_text = self.News_Detail(soup)
            news_event = utils.Event_Extraction(soup, news_short_desc)
            news_summary = utils.Article_Summary(news_text)
            total_news_words, news_words = utils.Count_Text_Words(news_text)
            total_summary_words, summary_words = utils.Count_Text_Words(
                news_summary)
            news_subjectivity, news_polarity, news_sentiment, news_complete_nouns, news_cardinal_digit, news_verbs = utils.News_POS(
                news_text)
            target_names = utils.News_Target_Names(news_summary)
            news_classification = utils.News_Classifier(news_text)
            country, city, address, latitude, longitude = utils.Geographic_Details(
                news_text)

            print('**********************************')
            print(f'News URL: {url}\n')
            print(f'News Source: {name}\n')
            print(f'News Section: {news_section}\n')
            print(f'News Title: {news_title}\n')
            print(f'News Author: {news_authors}\n')
            print(
                f'News Author Source link: {news_authors_link}\n')
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
                f'News Total Words in News Summary:  {total_news_words}\n')
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
                news_authors_link)
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

            # except:
            #     #     # print("Oops! authors not found", sys.exc_info()[0], "occurred.")
            #     continue
        df = pd.DataFrame.from_dict(
            df_news, orient='index').transpose()
        df.to_csv('eurasian_scraped.csv', index=False)
        print("Scraping Completed for ", name, " Tab ")
        # dff = pd.read_csv('C:/Users/ASDF/Desktop/print_media/dawn_national_scraped.csv')
        # df = pd.DataFrame.from_dict(df_news, orient='index').transpose()
        # frames = [dff, df]
        # result = pd.concat(frames)
        # result.to_csv('C:/Users/ASDF/Desktop/print_media/dawn_national_scraped.csv', index=False)
        return df_news

    def scrap_latest_news(self, latest_url, **kwargs):
        print("-------------------------------Scraping Latest Tab News-------------------------------")
        article_urls = self.Scrap_Latest_Links(latest_url)
        df_latest = self.Scrap_KashmirTimes_News(article_urls, 'KashmirTimes Latest')
        # df_latest = pd.DataFrame.from_dict(df_latest)
        return df_latest

    def scrap_national_news(self, latest_url, **kwargs):
        print("-------------------------------Scraping National Tab News-------------------------------")
        article_urls = self.Scrap_National_Links(latest_url)
        df_national = self.Scrap_KashmirTimes_News(article_urls, 'KashmirTimes India')
        # df_national = pd.DataFrame.from_dict(df_national)
        return df_national

    def scrap_world_news(self, latest_url, **kwargs):
        print("-------------------------------Scraping International Tab News-------------------------------")
        article_urls = self.Scrap_World_Links(latest_url)
        df_world = self.Scrap_KashmirTimes_News(article_urls, 'KashmirTimes International')
        # df_national = pd.DataFrame.from_dict(df_news)
        return df_world


if __name__ == "__main__":
    print("-------------------------------Scraping KashmirTimes News -------------------------------")
    scrap = KashmirTimes_News()
    latest = 'http://www.kashmirtimes.com/index.aspx'
    # national = 'http://www.kashmirtimes.com/news.aspx?q=India'
    # world = 'http://www.kashmirtimes.com/news.aspx?q=World'

    scrap_latest = scrap.scrap_latest_news(latest)
    # scrap_national = scrap.scrap_national_news(national)
    # scrap_world = scrap.scrap_world_news(world)

    # df_news = pd.DataFrame.from_dict(national)
    # df_news.to_csv('all_Eurasian_News_wrold_scraped.csv', index=False)
    # df_news.head()
