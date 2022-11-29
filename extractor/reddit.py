import praw
import os
import pandas as pd
from praw.models import MoreComments
from openpyxl import load_workbook
import configparser


def read_config():
    config = configparser.ConfigParser()
    config.read("config.ini", encoding='utf-8')
    return config


def get_comments(url):
    config = read_config()
    post_comments = []
    try:
        reddit = praw.Reddit(
            client_id=config['Reddit']['client_id'],
            client_secret=config['Reddit']['client_secret'],
            user_agent=config['Reddit']['user_agent'])
        submission = reddit.submission(url=url)
        for comment in submission.comments:
            if type(comment) == MoreComments:
                continue
            post_comments.append(comment.body)
        return post_comments
    except:
        return post_comments


def get_reddit(query, count: int = 100):
    config = read_config()
    print("Extracting Data from Reddit Posts for "+query)
    reddit = praw.Reddit(client_id=config['Reddit']['client_id'],
                         client_secret=config['Reddit']['client_secret'],
                         user_agent=config['Reddit']['user_agent'])
    subreddit = reddit.subreddit(query)
    posts = subreddit.hot(limit=count)

    posts_dict = {"Title": [], "Post Text": [], "ID": [],
                  "Score": [], "Total Comments": [], "Post URL": []}

    for post in posts:
        posts_dict["Title"].append(post.title)
        posts_dict["Post Text"].append(post.selftext)
        posts_dict["ID"].append(post.id)
        posts_dict["Score"].append(post.score)
        posts_dict["Total Comments"].append(post.num_comments)
        posts_dict["Post URL"].append(post.url)

    top_posts = pd.DataFrame(posts_dict)
    top_posts['comments'] = top_posts['Post URL'].apply(
        lambda x: get_comments(x))

    main_file_name = "Topics/"+query+"/Data/"+query+".xlsx"
    if os.path.exists(main_file_name):
        ExcelWorkbook = load_workbook(main_file_name)
        writer = pd.ExcelWriter(main_file_name, engine='openpyxl')
        writer.book = ExcelWorkbook
    else:
        writer = pd.ExcelWriter(main_file_name, engine='xlsxwriter')
    top_posts.to_excel(writer, index=False, sheet_name="Reddit")
    writer.save()
    writer.close()
    print("Reddit Extraction for "+query+" is complete")
