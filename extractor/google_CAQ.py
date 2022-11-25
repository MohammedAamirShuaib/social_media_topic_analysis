import pandas as pd
import people_also_ask
from openpyxl import load_workbook


def get_CAQ(topic):
    folder_name = topic.replace(" ", "")
    name_of_sheet = "CAQ"
    questions = people_also_ask.get_related_questions(topic, 20)
    df = pd.DataFrame(questions, columns=["Questions"])
    FilePath = "Topics/"+folder_name+"/Data/"+folder_name+".xlsx"
    ExcelWorkbook = load_workbook(FilePath)
    writer = pd.ExcelWriter(FilePath, engine='openpyxl')
    writer.book = ExcelWorkbook
    df.to_excel(writer, index=False, sheet_name=name_of_sheet)
    writer.save()
    writer.close()
    return True
