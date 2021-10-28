# -*- coding: utf-8 -*-
"""

@author: PrateekKumar
"""

# General:
import tweepy           # To consume Twitter's API
import pandas as pd     # To handle data
import numpy as np      # For number computing
import json
from elasticsearch import Elasticsearch

# create instance of elasticsearch
es = Elasticsearch()

CONSUMER_KEY    = 
CONSUMER_SECRET = 

# Access:
ACCESS_TOKEN  = 
ACCESS_SECRET = 

# API's setup:
def twitter_setup():
    """
    Utility function to setup the Twitter's API
    with our access keys provided.
    """
    # Authentication and access using keys:
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)

    # Return API with authentication:
    api = tweepy.API(auth)
    return api
# We create an extractor object:
extractor = twitter_setup()

# We create a tweet list as follows:
tweets = extractor.search(q="bareMinerals", count=500)
print("Number of tweets extracted: {}.\n".format(len(tweets)))

data = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['Tweets'])

# We add relevant data:
data['len']  = np.array([len(tweet.text) for tweet in tweets])
data['ID']   = np.array([tweet.id for tweet in tweets])
data['Date'] = np.array([tweet.created_at for tweet in tweets])
data['Source'] = np.array([tweet.source for tweet in tweets])
data['Likes']  = np.array([tweet.favorite_count for tweet in tweets])
data['RTs']    = np.array([tweet.retweet_count for tweet in tweets])
data['Screen_Name']   = np.array([tweet.user.screen_name for tweet in tweets])
data['Language']  = np.array([tweet.user.lang for tweet in tweets])
data['Location']  = np.array([tweet.user.location for tweet in tweets])
data['Geo_Enabled']  = np.array([tweet.user.geo_enabled for tweet in tweets])
data['Followers_Count']  = np.array([tweet.user.followers_count for tweet in tweets])
data['Time_Zone']  = np.array([tweet.user.time_zone for tweet in tweets])
data['Verified']  = np.array([tweet.user.verified for tweet in tweets])
#data['RT Status']  = np.array([tweet.retweeted_status for tweet in tweets])

data['Date'] = pd.to_datetime(data['Date']).dt.strftime('%Y/%m/%d %H:%M:%S') 

from textblob import TextBlob
import re

def clean_tweet(tweet):
    '''
    Utility function to clean the text in a tweet by removing 
    links and special characters using regex.
    '''
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

def analize_sentiment(tweet):
    '''
    Utility function to classify the polarity of a tweet
    using textblob.
    '''
    analysis = TextBlob(clean_tweet(tweet))
    if analysis.sentiment.polarity > 0.6:
        return analysis.sentiment.polarity
    elif analysis.sentiment.polarity > 0.2:
        return analysis.sentiment.polarity
    elif analysis.sentiment.polarity > -0.2:
        return analysis.sentiment.polarity
    elif analysis.sentiment.polarity > -0.6:
        return analysis.sentiment.polarity
    else:
        return analysis.sentiment.polarity
    
# We create a column with the result of the analysis:
data['SA'] = np.array([ analize_sentiment(tweet) for tweet in data['Tweets'] ])

def sentiment(tweet):
    '''
    Utility function to classify the polarity of a tweet
    using textblob.
    '''
    analysis = TextBlob(clean_tweet(tweet))
    if analysis.sentiment.polarity > 0.6:
        return "very positive"
    elif analysis.sentiment.polarity > 0.2:
        return "positive"
    elif analysis.sentiment.polarity > -0.21:
        return "neutral"
    elif analysis.sentiment.polarity > -0.6:
        return "negative"
    else:
        return "very negative"
    
# We create a column with the result of the analysis:
data['Sentiment'] = np.array([ sentiment(tweet) for tweet in data['Tweets'] ])
data['Sentiment'] = data['Sentiment'].astype(str)

data['Screen_Name']  = np.where(data['Verified'], data['Screen_Name']+'(v)', data['Screen_Name'])
data['Verified Account']  = np.where(data['Verified'], 'Y', 'N')
data['RTs+Likes']=data['RTs']+data['Likes']

data["no_index"] = [x+1 for x in range(len(data["Tweets"]))]
# Convert into json
tmp = data.to_json(orient = "records")
# Load each record into json format before bulk
data_json= json.loads(tmp)
print data_json[0]

for doc in data_json:
    es.index (index="twitter-sa", doc_type='doc', id=doc['no_index'], body=doc)
