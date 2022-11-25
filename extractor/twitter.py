import tweepy as tw
import pandas as pd
from openpyxl import load_workbook
import configparser


def read_config():
    config = configparser.ConfigParser()
    config.read("config.ini", encoding='utf-8')
    return config


def get_tweets(query, count: int = 100):
    config = read_config()
    auth = tw.OAuthHandler(
        config['Twitter']['consumer_key'],
        consumer_secret=config['Twitter']['consumer_secret']
    )
    auth.set_access_token(
        config['Twitter']['access_token'],
        config['Twitter']['access_token_secret']
    )
    api = tw.API(auth, wait_on_rate_limit=True)
    tweets = api.search_tweets(
        q=query, lang="en", count=count, tweet_mode="extended")

    users_locs = [[
        tweet.created_at,
        tweet.user.location,
        tweet.user.screen_name,
        tweet.retweeted_status.full_text if tweet.full_text.startswith(
            "RT") else tweet.full_text,
        [hashs['text'] for hashs in tweet.entities['hashtags']]
    ]for tweet in tweets]
    twitterDf = pd.DataFrame(data=users_locs, columns=[
                             'datetime', "location", "user", "tweet", "hashtags"])

    FilePath = "Topics/" + \
        query.replace(" ", "")+"/Data/"+query.replace(" ", "")+".xlsx"
    ExcelWorkbook = load_workbook(FilePath)
    writer = pd.ExcelWriter(FilePath, engine='openpyxl')
    writer.book = ExcelWorkbook
    twitterDf.to_excel(writer, index=False, sheet_name="Twitter")
    writer.save()
    writer.close()
