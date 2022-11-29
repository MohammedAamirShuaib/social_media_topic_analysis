import tweepy as tw
import os
import pandas as pd
from openpyxl import load_workbook
import configparser


def read_config():
    config = configparser.ConfigParser()
    config.read("config.ini", encoding='utf-8')
    return config


def get_tweets(query, count: int = 100):
    print("Extracting Data from Twitter on " + query)
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
        str(tweet.created_at),
        tweet.user.location,
        tweet.user.screen_name,
        tweet.retweeted_status.full_text if tweet.full_text.startswith(
            "RT") else tweet.full_text,
        [hashs['text'] for hashs in tweet.entities['hashtags']]
    ]for tweet in tweets]
    twitterDf = pd.DataFrame(data=users_locs, columns=[
                             'datetime', "location", "user", "tweet", "hashtags"])

    main_file_name = "Topics/" + \
        query.replace(" ", "")+"/Data/"+query.replace(" ", "")+".xlsx"
    if os.path.exists(main_file_name):
        ExcelWorkbook = load_workbook(main_file_name)
        writer = pd.ExcelWriter(main_file_name, engine='openpyxl')
        writer.book = ExcelWorkbook
    else:
        writer = pd.ExcelWriter(main_file_name, engine='xlsxwriter')
    twitterDf.to_excel(writer, index=False, sheet_name="Twitter")
    writer.save()
    writer.close()
    print("Twitter Extraction on "+query+" is complete")
