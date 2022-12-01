from pytube import YouTube
import requests
import pandas as pd
import os
from openpyxl import load_workbook
import configparser
import time
from extractor.ser_functions import *
from extractor.ser_functions import Get_Emotions, emotion_results


def read_config():
    config = configparser.ConfigParser()
    config.read("config.ini", encoding='utf-8')
    return config

# ------------------------------------------Downloading Youtube Videos----------------------------------------


def get_video(query, links):
    SAVE_PATH = "Topics/"+query+"/videos/"
    for link in links:
        try:
            yt = YouTube(link)
        except:
            print("Connection Error")
        yt.streams.filter(progressive=True, file_extension='mp4').order_by(
            'resolution').asc().first().download(SAVE_PATH)
        time.sleep(5)


# -------------------------------------------Transcribing Youtube Videos--------------------------------------
def read_file(filename, chunk_size=5242880):
    with open(filename, 'rb') as _file:
        while True:
            data = _file.read(chunk_size)
            if not data:
                break
            yield data


def get_url(token, fname):
    headers = {'authorization': token}
    response = requests.post('https://api.assemblyai.com/v2/upload',
                             headers=headers,
                             data=read_file(fname))
    url = response.json()["upload_url"]
    return url


def get_transcribe_id(token, url):
    endpoint = "https://api.assemblyai.com/v2/transcript"
    json = {
        "audio_url": url,
        "speaker_labels": True
    }
    headers = {
        "authorization": token,
        "content-type": "application/json"
    }
    response = requests.post(endpoint, json=json, headers=headers)
    id = response.json()['id']
    return id


def upload_file(token, fname):
    file_url = get_url(token, fname)
    transcribe_id = get_transcribe_id(token, file_url)
    return transcribe_id


def get_text(token, transcribe_id):
    endpoint = f"https://api.assemblyai.com/v2/transcript/{transcribe_id}"
    headers = {
        "authorization": token
    }
    result = requests.get(endpoint, headers=headers).json()
    return result


def json_data_extraction(result, fname):
    audindex = pd.json_normalize(result['words'])
    audindex['file_name'] = fname
    speakers = list(audindex.speaker)
    previous_speaker = 'A'
    l = len(speakers)
    i = 0
    speaker_seq_list = list()
    for index, new_speaker in enumerate(speakers):
        if index > 0:
            previous_speaker = speakers[index - 1]
        if new_speaker != previous_speaker:
            i += 1
        speaker_seq_list.append(i)
    audindex['sequence'] = speaker_seq_list

    group = ['file_name', 'speaker', 'sequence']
    df = pd.DataFrame(audindex.groupby(group).agg(
        utter=('text', ' '.join), start_time=('start', 'min'), end_time=('end', 'max')))
    df.reset_index(inplace=True)
    df.sort_values(by=['start_time'], inplace=True)
    return df

# ---------------------------------------------Main Function---------------------------------------


def get_youtube(query, links, download: bool = True, get_emotions: bool = False, model: str = 'BERT_PT'):
    if download:
        print("Downloading YouTube Videos")
        get_video(query, links)
        print("Download Complete")
    config = read_config()
    files = os.listdir("Topics/"+query+"/videos/")
    for file in files:
        fname = "Topics/"+query+"/videos/"+file
        tid = upload_file(config['YouTube']['token'], fname)
        result = {}
        print('Transcribing file: [ {} ]'.format(fname))
        while result.get("status") != 'completed':
            result = get_text(config['YouTube']['token'], tid)

        df = json_data_extraction(result, fname)
        if get_emotions:
            print('Extracting Emotions')
            df = generate_emotions(df, model)

        main_file_name = "Topics/"+query+"/Data/"+query+".xlsx"
        if os.path.exists(main_file_name):
            ExcelWorkbook = load_workbook(main_file_name)
            writer = pd.ExcelWriter(main_file_name, engine='openpyxl')
            writer.book = ExcelWorkbook
        else:
            writer = pd.ExcelWriter(main_file_name, engine='xlsxwriter')
        df.to_excel(writer, index=False, sheet_name="YouTube_video_" +
                    str(files.index(file)+1))
        writer.save()
        writer.close()
        print('Data Extraction Complete: [ {} ]'.format(fname))


def generate_emotions(df, model):
    df['model'] = model
    df['Emotion'] = df.apply(lambda x: Get_Emotions(
        x['file_name'], x['start_time'], x['end_time'], x['utter'], x['model']), axis=1)
    if model == "BERT_PT":
        df['angry'] = df['Emotion'].apply(lambda x: emotion_results(x, "ang"))
        df['happy'] = df['Emotion'].apply(lambda x: emotion_results(x, "hap"))
        df['sad'] = df['Emotion'].apply(lambda x: emotion_results(x, "sad"))
        df['neutral'] = df['Emotion'].apply(
            lambda x: emotion_results(x, "neu"))
        df['excited'] = df['Emotion'].apply(
            lambda x: emotion_results(x, "exc"))
        df.drop(['Emotion'], axis=1, inplace=True)
    if model == "HuggingFace":
        df['angry'] = df['Emotion'].apply(lambda x: emotion_results(x, "ang"))
        df['happy'] = df['Emotion'].apply(lambda x: emotion_results(x, "hap"))
        df['sad'] = df['Emotion'].apply(lambda x: emotion_results(x, "sad"))
        df['neutral'] = df['Emotion'].apply(
            lambda x: emotion_results(x, "neu"))
        df.drop(['Emotion'], axis=1, inplace=True)
    return df
