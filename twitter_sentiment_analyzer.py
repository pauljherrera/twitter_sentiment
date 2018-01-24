import tweepy
import time
import datetime as dt
import re

from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.gridspec as gridspec


def percentages(x, pos):
    """
    Function used for formatting the Y axis of the chart.
    """
    return '%0.0f %%' % (x * 100)


class TwitterAnalyzer():
    """
    The Twitter Analyzer makes a search on Twitter, stores the tweets
    for future searches on the same topic, and
    makes an analysis of the sentiment. Finally, plots the results of
    the sentiment analysis.
    """

    def __init__(self, consumer_key, consumer_secret, access_token, 
                 access_token_secret, waiting_mode=False, *args, **kwargs):

        self.auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        self.auth.set_access_token(access_token, access_token_secret)

        self.api = tweepy.API(self.auth)
        self.stop_words = set(stopwords.words("english"))

        self.waiting_mode = waiting_mode

    def analyze_sentiment(self):
        """
        Classifies each tweet in positive, negative or neutral.
        Returns two DataFrames with the total number of tweets 
        on each classification and the percentage of positive tweets 
        compared with the percentage of the negative tweets.
        For classifying, we take Sentiment Intensity Analyzer from
        the Natural Language Toolkit library (NLTK).
        """
        # Instantiating the sentiment intensity analyzer.
        sia = SIA()
        
        # Creating scores dataframe.
        tweets = self.tweets.dropna()
        tweets.index = tweets.index.date
        index = tweets.index.unique()
        columns = ['Positives', 'Neutral', 'Negatives']
        scores = pd.DataFrame(columns=columns, index=index).fillna(0).sort_index()
        
        # Looping through the tweets lists.
        for i in range(len(tweets)):
            index = tweets.iloc[i, :].name
            print(tweets.iloc[i, 1])
            score = sia.polarity_scores(tweets.iloc[i, 1])
            if score['compound'] > 0.10:
                scores.loc[index, 'Positives'] += 1
            elif score['compound'] < -0.10:
                scores.loc[index, 'Negatives'] += 1
            else:
                scores.loc[index, 'Neutral'] += 1
                     
        # Creating scores percentage dataframe.
        scores_pct = pd.DataFrame(columns=columns)    
        for column in columns:
            scores_pct[column] = scores[column] / scores.sum(axis=1)
        scores_pct.fillna(0, inplace=True)
        
        self.scores = scores
        self.scores_pct = scores_pct
        
        return scores, scores_pct


    def search_feed(self, search_words):
        # Retrieving the tweets file or creating it.
        try:
            df = pd.read_csv('tweets/' + search_words + '.csv', index_col=0, 
                             parse_dates=True)
        except FileNotFoundError:
            columns = ['Tweet', 'ID']
            df = pd.DataFrame(columns=columns)
        
        # Instantiating cursor for retrieving the tweets from the API.
        if self.waiting_mode:
            tw_cursor = self.limit_handled(tweepy.Cursor(self.api.search, 
                                                         q=search_words, lang='en').pages())
        else:
            tw_cursor = tweepy.Cursor(self.api.search, q=search_words, 
                                      lang='en').pages(50)

        # Iterating through the tweets.
        print('\nSearching for tweets')
        for page in tw_cursor:
            for status in page:
                # Fetching tweet's complete info.
                tweet = status._json
                # Extracting the data.
                date = dt.datetime.strptime(tweet['created_at'],'%a %b %d %H:%M:%S +0000 %Y')
                # Extracting the data and removing unnecesary symbols.
                tweet_text = self.clean_content(tweet['text'])
                # Preprocessing tweet for sentiment analysis.
                tweet_text = self.pre_process(tweet_text)
                # Adding new tweet to the existent Dataframe.
                new_df = pd.DataFrame({'Tweet': tweet_text,                                             
                                       'ID': tweet['id']},
                                       index = [date])
                df = pd.concat([df, new_df])
                
        # Storing tweets removing duplicates.
        df = df.drop_duplicates()
        df = df.dropna()
        df.to_csv('tweets/' + search_words + '.csv')
        self.tweets = df
        
        return df

    def clean_content(self, tweet):
        """ 
        Replace URLS, @username, special chars, white spaces, hashtags, etc 
        """
        tweet = tweet.lower()
        tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',tweet)
        tweet = re.sub('@[^\s]+','',tweet)
        tweet = re.sub('[\s]+', ' ', tweet)
        tweet = re.sub('[\n]+', ' ', tweet)
        tweet = re.sub(r'[^\w]', ' ', tweet)
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
        tweet = re.sub('rt','',tweet)
        tweet = tweet.replace(':)','')
        tweet = tweet.replace(':(','')
        tweet = tweet.strip('\'"')

        return tweet
    
    def pre_process(self,tweet):
        """
        The preprocessing process has two main steps: deleting stop words 
        and stemming.
        Stopwords are words that doesn't add anything to the sentiment 
        of the test such as 'to, and, or, etc". NLTK comes from
        a complete library of stopwords for the english language.
        Stemming is reducing each word to its stem. It makes it easier
        to the trained machine learning model built in NLTK to identify
        the sentiment of each word.
        """
        snow_s = SnowballStemmer("english")
        token_word = word_tokenize(tweet)
        tokenized = []
        for word in token_word:
            if word not in self.stop_words:
                tokenized.append(snow_s.stem(word))

        return " ".join(tokenized)
    
    def limit_handled(self, cursor):
        """
        Helper funcion for the waiting_mode
        """
        while True:
            try:
                yield cursor.next()
            except tweepy.error.TweepError:
                print('\nWaiting until the Twitter API allows us to search more tweets...')
                time.sleep(15 * 60)



if __name__ == "__main__":
    # User variables.
    search_string = "Pope Francis"  
    waiting_mode = False
    plot = True
    
    consumer_key = 'z4OMjR0lY1PGYojysSZK4bLtf'
    consumer_secret = 'aBBWdQK1aT4mKLIz2zbD42fLJJhQUaKqfQkUSAkEEwxcKPDWlk'
    access_token = '74024384-WGOxmxPOaAApRFLTvhETyqcoXErZ7Gn2021rjtnwG'
    access_token_secret = '3N5JEE32n9GcQJhqps8VbzONNVHvWYHWYcwZOqnrPnyaU'

    # Analyzing tweets.
    tw = TwitterAnalyzer(consumer_key, consumer_secret, access_token, 
                         access_token_secret, 
                         waiting_mode=waiting_mode)
    Tweets_lists = tw.search_feed(search_string)
    print("\nAnalyzing the sentiment of the tweets")
    scores, scores_pct = tw.analyze_sentiment()
    
    # Making plots
    if plot:
        print("\nCreating charts")
        gs = gridspec.GridSpec(5, 5,
                           width_ratios=[2, 14, 3, 14, 2],
                           height_ratios=[1, 2, 1, 8, 1],
                           wspace=0, hspace=0
                           )
    
        ax1 = plt.subplot(gs[6:10])
        ax2 = plt.subplot(gs[16:17])
        ax3 = plt.subplot(gs[18:19])
        fig = plt.gcf()
        fig.set_size_inches(18.5, 10)
    
        ax1.text(0.01, 0.5, 
                 "Sentiment analysis of the search:\n '" + search_string + "'", 
                 fontdict={'fontname':'DejaVu Sans'}, 
                 color='#6194BC',
                 fontsize=29)
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.tick_params(axis=u'both', which=u'both',length=0)
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        
        ax2 = scores.plot(ax=ax2, kind='bar', color=['#A5D1F3', '#E49D67', '#E4001B'])
        ax2.set_title('Number of tweets', fontsize=22)
        for tick in ax2.get_xticklabels():
            tick.set_rotation(45)
        
        ax3 = scores_pct[['Positives', 'Negatives']].plot(kind='bar', 
                                                          color=['#A5D1F3', '#E4001B'],
                                                          ax=ax3)
        ax3.set_title('Percentage of positive and negative tweets',
                      fontsize=22)
        for tick in ax3.get_xticklabels():
            tick.set_rotation(45)
        formatter = FuncFormatter(percentages)
        ax3.yaxis.set_major_formatter(formatter)
        
        fig.savefig('charts/' + search_string + '.png')

                
                
    
    