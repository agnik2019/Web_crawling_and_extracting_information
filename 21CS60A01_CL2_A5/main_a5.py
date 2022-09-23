# Agnik Saha
# Roll: 21CS60A01
# ASSIGNMENT 5
# python3 main_a5.py

# importing necessary modules
import os
import re
import requests
import datetime
import sys

# modules for qn2
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import io
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import numpy as np
import pandas as pd
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter
from collections import namedtuple # for checking overlapping dates

import ply.lex as lex
import ply.yacc as yacc

# variables for assignment 4
total_cases = []
active_cases = []
total_deaths = []
total_recovered = []



# header for request module
headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36'
    }


# ======================== helper functions for qn 1 ===============================

# cleanhtml takes input as raw_html and outputs the only texts by removing html tags
def cleanhtml(raw_html):
    CLEANR = re.compile('<.*?>') 
    cleantext = re.sub(CLEANR, '', raw_html)
    pattern = r'.&#'
    modified_text = re.sub(pattern, '', cleantext )
    modified_text = re.sub("\n+"," ",modified_text)
    modified_text = re.sub("\t+"," ",modified_text)
    modified_text = re.sub("^"," ",modified_text)   # new
    #modified_text = re.sub("[edit]"," ",modified_text)   # new
    CLEANR2 = re.compile('[.*?]') 
    modified_text = re.sub(CLEANR2, '', modified_text)
    CLEANR3 = re.compile('<!-- .*? -->') 
    modified_text = re.sub(CLEANR3, '', modified_text)
    return modified_text

def remove_unnecesary_texts_from_timeline(lines):
    if "mw-parser-output navbox{" in lines:
        lines = lines.split("mw-parser-output navbox{")[0]
    
    if "Pandemic chronology" in lines:
        lines = lines.split("Pandemic chronology")[2]
    
    
    return lines

def nextclean(lines):
    modified_text = re.sub(r"2020\^","",lines)
    modified_text = re.sub(r"2020\)","",modified_text)
    modified_text = re.sub(r"2020\s\(","",modified_text)

    return modified_text

# this function returns a list containing day, month and year from user inputed date
def day_month_year_from_date(date):
    day = date.strftime("%d")
    month = date.strftime("%m")
    year = date.strftime("%Y")
    
    if month == "01":
        month = "January"
    elif month == "02":
        month = "February"
    elif month == "03":
        month = "March"
    elif month == "04":
        month = "April"
    elif month == "05":
        month = "May"
    elif month == "06":
        month = "June"
    elif month == "07":
        month = "July"
    elif month == "08":
        month = "August"
    elif month == "09":
        month = "September"
    elif month == "10":
        month = "October"
    elif month == "11":
        month = "November"
    elif month == "12":
        month = "December"
    
    if day < "10":
        day = day[1:]
    return [day,month,year]


# This function returns a list of urls of timelines
def timeline_urls_from_time_range(date1, date2):
    monthArry = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    month_1 = int(date1.split("-")[1]) -1
    year_1 = int(date1.split("-")[2])
    month_2 = int(date2.split("-")[1]) -1

    main_timeline_url = "https://en.wikipedia.org/wiki/Timeline_of_the_COVID-19_pandemic_in_"
    urls = []

    while month_1 <= month_2 :
        url = main_timeline_url+monthArry[month_1]+"_"+str(year_1)
        urls.append(url)
        month_1 += 1
        if monthArry[month_1] == 'December':
            year_1 += 1

    # start = datetime.datetime.strptime(date1, '%d-%m-%Y')
    # end = datetime.datetime.strptime(date2, '%d-%m-%Y')
    # step = datetime.timedelta(days=28)
    # while start <= end:
    #     date = day_month_year_from_date(start.date())
    #     url = main_timeline_url+str(date[1])+"_"+str(date[2])
    #     # url = url+month+"_"+year
    #     urls.append(url)
    #     start += step
    return urls


# This function returns a list of urls of responses
def response_urls_from_time_range(date1, date2):
    main_response_url = "https://en.wikipedia.org/wiki/Responses_to_the_COVID-19_pandemic_in_"
    monthArry = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    month_1 = int(date1.split("-")[1]) -1
    year_1 = int(date1.split("-")[2])
    month_2 = int(date2.split("-")[1]) -1

    urls = []

    while month_1 <= month_2 :
        url = main_response_url+monthArry[month_1]+"_"+str(year_1)
        urls.append(url)
        month_1 += 1
        if monthArry[month_1] == 'December':
            year_1 += 1

    return urls

# =========================== Helper function for question 2 ===============================
# word_tokenize accepts
# a string as an input, not a file.


def filter_text(words):
    filtered_word = []
    stop_words = set(stopwords.words('english'))
    for r in words:
        if not r in stop_words:
            filtered_word.append(r)
    return filtered_word

# this function just plot the word-cloud
def plot_word_cloud(text):
    wordcloud = WordCloud(width = 1000, height = 500).generate(text)
    plt.figure(figsize=(15,8))
    plt.imshow(wordcloud)
    plt.axis("off")
    #plt.savefig("your_file_name"+".png", bbox_inches='tight')
    plt.show()
    plt.close()

def get_covid_dict():
    covid_dict = []
    inputFileName = "covid_word_dictionary.txt"
    # covid_dict : contains all covid related word
    with open(inputFileName) as f:
        covid_dict = [ line.strip() for line in list(f) ]
    
    return covid_dict

# common_words -> words from filtered text
def get_common_words(inputfilename):
    common_words = []
    with open(inputfilename, "r") as f:
        common_words = f.read().split()
    return common_words

# This function returns covid related common words
def covid_related_common_words(common_words,covid_dict):
    covid_word_in_text = []
    for commonword in common_words:
        if commonword in covid_dict:
            covid_word_in_text.append(commonword)

    return covid_word_in_text    

def percentage_of_covid_words_in_common_words(common_words,covid_dict):
    covid_word_in_text = covid_related_common_words(common_words,covid_dict) 
    percentage = (len(covid_word_in_text)/len(common_words))*100
    return percentage 

def top_20_words(words):
    common = " ".join(ele for ele in words)
    common_map = Counter(common.split())

    top_common_words = sorted(common_map.items(), key=lambda x: x[1], reverse=True)
    return top_common_words[:20]




# ========================= Helper Functions for question 3 ===============================

def fetch_country_urls():
    country_text = "covid_country_list.txt"
    with open(country_text) as f:
        countries = [ line.strip().replace(' ', '_') for line in list(f) ]

    countries[-1] = "the_"+countries[-1]
    # countries is a list of countries given in the text

    urls = []
    for coun in countries:
        url ="https://en.wikipedia.org/wiki/Timeline_of_the_COVID-19_pandemic_in_{name}".format(name =  coun)
        urls.append(url)

    if not os.path.exists("./COUNTRIES"): 
        os.mkdir("./COUNTRIES")

    for i in range(0,len(urls)):
        print(f'Fetching web page: {urls[i]}')
        try:
            response = requests.get(url=urls[i], headers=headers)
            page_content = response.text
        except Exception as e:
            print(e)
            continue
        coun_name = "./COUNTRIES/" + countries[i] + ".html"
        f = open(coun_name, 'w',encoding="utf8")
        f.write(page_content)
        f.close()


def date_range(coun):
    range = "Feb 2020 to June 2021 "
    if coun == "argentina": range = "Feb 2020 to Jan 2021 "
    elif coun == "australia": range =  "Jan 2020 to Feb 2022"
    elif coun == "bangladesh": range =  "Jan 2020 to Jun 2020"
    elif coun == "brazil": range =  "Nov 1019 to June 2022"
    elif coun == "canada": range =  "Dec 2019 to April 2021"
    elif coun == "india": range =  "Feb 2020 to Aug 2022"
    return range

# ============================= Helper functions for question 5 ==================================
def Jaccard_Similarity(doc1, doc2): 
    
    # List the unique words in a document
    words_doc1 = set(doc1.lower().split()) 
    words_doc2 = set(doc2.lower().split())
    
    # Find the intersection of words list of doc1 & doc2
    intersection = words_doc1.intersection(words_doc2)

    # Find the union of words list of doc1 & doc2
    union = words_doc1.union(words_doc2)
        
    # Calculate Jaccard similarity score 
    # using length of intersection set divided by length of union set
    return float(len(intersection)) / len(union)

# ============================ Merging assignment 4 ================================
def fetch_country_from_worldometer():
    inputFileName = "worldometers_countrylist.txt"
    main_url = "https://www.worldometers.info/coronavirus/"
    response = requests.get(main_url, headers=headers)
    page_content = response.text

    if not os.path.exists("./a4_COUNTRIES"): 
        os.mkdir("./a4_COUNTRIES")

    file_name = "./a4_COUNTRIES/world.html"
    f = open(file_name, 'w',encoding='utf-8')
    f.write(page_content)
    f.close()


    with open(inputFileName) as f:
        lines = [ line.strip('\n').lower().replace(' ', '-') for line in list(f) ]

    a = lines
    b = ["--------","--------","--------","--------", "","europe:", "---------","north-america:","asia","south-america","africa","oceania"]

    # country_list is the list of all the countries containing in the text file
    country_list = [x for x in a if (x not in b)]

    print("==================== country list =========================")
    for l in country_list:
        print(l)

    # urls is the list of all the urls of each country in the worldometers website
    urls = []
    for coun in country_list:
        url ="https://www.worldometers.info/coronavirus/country/{name}/".format(name =  coun)
        urls.append(url)


    # saving the html file of each country in the "COUNTRIES" folder
    if not os.path.exists("./a4_COUNTRIES"): 
        os.mkdir("./a4_COUNTRIES")

    for i in range(0,len(urls)):
        print(f'Fetching web page: {urls[i]}')
        try:
            response = requests.get(url=urls[i], headers=headers)
            page_content = response.text
        except Exception as e:
            print(e)
            continue
        coun_name = "./a4_COUNTRIES/" + country_list[i] + ".html"
        f = open(coun_name, 'w')
        f.write(page_content)
        f.close()

    

# writing grammer rules
def t_LTOTALCASE(t):
    r'<span\sstyle="color:\#aaa">'
    return t


def t_CLOSESPAN(t):
    r'</span>\n*'
    return t


def t_LRECOVERCASE(t):
    r'<div\sclass="maincounter-number"\sstyle="color:\#8ACA2B\s">\n*'
    return t


def t_LDEATH(t):
    r'<h1>Deaths:<\/h1>\n<div\sclass="maincounter-number">\n'
    return t

def t_LACTIVECASE(t):
    r'<div\sclass="number-table-main">*'
    return t
def t_EACTIVECASE(t):
    r'</div>\n<div\sstyle="font-size:13.5px">Currently'
    return t


def t_CLOSEDIV(t):
    r'</div>'
    return t

def t_OPENSPAN(t):
    r'<span>'
    return t


# lex-yacc for qn 3

# def L_STIME(t):
#     r'<\ul>'

# def L_ETIME(t):
#     r'\<\/ul\>'

# def t_OSP(t):
#     r'<li\sclass="toclevel-\d+\stocsection-\d+"><a\shref="\#[a-zA-Z]*_\d+"><span\sclass="tocnumber">\d+<\/span>\s<span\sclass="toctext">'
#     return t

# def t_CSP(t):
#     r'<\/span><\/a><\/li>'
#     return t


# def t_DATERANGE(t):
#     r"[A-Za-z]+\s\d{4}"
#     return t



def p_start(t):
    '''start : total_cases
                | recovered_cases 
                | deaths
                | active_cases 
             '''
# time for question 3

# def p_pdate(t):
#     'pdate : DATERANGE'
#     t[0] = t[1]
    

# def p_pdate_multi(t):
#     'pdate : DATERANGE pdate'
#     t[0] = t[1] + ' ' + t[2]


# def p_multi_news(t):
#     'news : OSP pdate CSP news '
#     t[0] = t[2] + ", " + t[4]
#     print(t[0])

# def p_time(t):
#     'time : STIME news ETIME'
#     print("time : ", t[2])



def t_NUM(t):
    r"\d+"
    return t


t_ignore = " \t"

def t_error(t):
    t.lexer.skip(1)


def p_total_cases(t):
    'total_cases : LTOTALCASE pnum CLOSESPAN'
    t[0] = t[2]
    # print("Total Cases : ", t[2].replace(" ", ""))
    total_cases.append(t[2].replace(" ", ""))

def p_recovered_cases(t):
    'recovered_cases : LRECOVERCASE OPENSPAN pnum CLOSESPAN CLOSEDIV'
    t[0] = t[3]
    #print("Recovered Cases : ", t[3])
    total_recovered.append(t[3].replace(" ", ""))

def p_deaths(t):
    'deaths : LDEATH OPENSPAN pnum CLOSESPAN CLOSEDIV'
    t[0] = t[3]
    #print("total deaths : ", t[3])
    total_deaths.append(t[3].replace(" ", ""))

def p_active_cases(t):
    'active_cases : LACTIVECASE pnum EACTIVECASE'
    t[0] = t[2]
    #print("Active cases : ", t[2])
    active_cases.append(t[2].replace(" ", ""))

def p_pnum(t):
    'pnum : NUM'
    t[0] = t[1]


def p_pnum_multi(t):
    'pnum : NUM pnum'
    t[0] = t[1] + ' ' + t[2]



# lex/yac for question 3

def p_error(t):
    pass



tokens = [
    'LTOTALCASE',
    'CLOSESPAN',
    'NUM',
    'LRECOVERCASE',
    'LDEATH',
    'CLOSEDIV',
    'OPENSPAN', 
    'LACTIVECASE',  
    'EACTIVECASE',
    ]  


# =========================== MAIN Module of the code ===============================
#main starts here
def main():

    monthArry = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

    while True:
            field_requested = input("Enter the field no.(1-4) \n1.News between time range and plot word cloud\n2.Response between time range and plot wordcloud\n3.percentage of covid related words in common words, top-20 common words and top 20 covid related words and plot two word clouds\n4. Input a country and find time-range \n5.Total cases, Active cases, Total deaths and Total recovered numbers of a particular country\n6.exit\n")
            
            if int(field_requested) not in range(1,7):
                print("Enter Val in range !!!")
                sys.exit(0)

          

            if(field_requested  == "1"):
                print("Enter the time range(format dd-mm-yyyy ie 01-03-2022) ")
                date1 = str(input("Enter the date from which you want to see the news(dd-mm-yyyy): "))
                date2 = str(input("Enter the date till which you want to see the news(dd-mm-yyyy): "))  

                month_1 = int(date1.split("-")[1])
                date_1 = int(date1.split("-")[0])
                month_1_name = monthArry[month_1-1]

                month_2 = int(date2.split("-")[1])
                date_2 = int(date2.split("-")[0])
                month_2_name = monthArry[month_2-1]

                # print(f'month {month_1} date {date_1}')

                timeline_urls = timeline_urls_from_time_range(date1,date2)
                print("======== TIMELINE URLS ========")
                for url in timeline_urls:
                    print(url)

                # SAVE THE TIMELINE URLS IN TIMELINE FOLDER
                if not os.path.exists("./TIMELINES"): 
                    os.mkdir("./TIMELINES")

                for i in range(0,len(timeline_urls)):
                    # print(f'Fetching web page: {timeline_urls[i]}')
                    try:
                        response = requests.get(url=timeline_urls[i], headers=headers)
                        page_content = response.text
                        clean_html = cleanhtml(page_content)
                        clean_html = remove_unnecesary_texts_from_timeline(clean_html)
                    except Exception as e:
                        print(e)
                        continue

                    timeline = "./TIMELINES/" + str(timeline_urls[i][-14:]) + ".txt"
                    with open(timeline, 'w',encoding="utf-8") as f:
                        f.write(clean_html)
                
                news_data_p = {}
                for i in range(0,len(timeline_urls)):
                    timeline = "./TIMELINES/" + str(timeline_urls[i][-14:]) + ".txt"
                    
                    with open(timeline,'r',encoding="utf-8") as f:
                        lines = str(f.readlines())
                        monthLines = re.findall(r"(.*?)([\d]+\s+January|[\d]+\s+February|[\d]+\s+March|[\d]+\s+April|[\d]+\s+May|[\d]+\s+June|[\d]+\s+July|[\d]+\s+August|[\d]+\s+September|[\d]+\s+October|[\d]+\s+November|[\d]+\s+December)",lines)
                        
                        previous_news = ""
                        date = 0
                        month = ""
                        for news in monthLines:
                            previous_news = news[0].strip()
                            previous_news = cleanhtml(previous_news)
                            previous_news = re.sub("2020 \^", "", previous_news)
                            previous_news = re.sub("2021 \^", "", previous_news)
                            previous_news = re.sub("2022 \^", "", previous_news)
                            previous_news = re.sub("2020\)", "", previous_news)
                            previous_news = re.sub("2021\)", "", previous_news)
                            previous_news = re.sub("2022\)", "", previous_news)
                            previous_news = re.sub("\[edit\]", "", previous_news)
                            previous_news = re.sub("a b c d e f g h i j", "", previous_news)
                            previous_news = re.sub("a b c d e f g h i j k l m n", "", previous_news)
                            previous_news = re.sub("a b c d e f g h i j k l m n o", "", previous_news)
                            previous_news = re.sub("a b c d e f g h i j k l m n o p q r s t", "", previous_news)
                            previous_news = re.sub("a b c d e f g h i j k l m n o p q r s t u", "", previous_news)
                            previous_news = re.sub("a b c d e f g h i j k l m n o p q r s", "", previous_news)
                            previous_news = re.sub("2020 Archived from the original on", "", previous_news)
                            previous_news = re.sub(r'mw-parser[\d\w\s,-]+{.*?}', "", previous_news)
                            previous_news = re.sub(r'mw-parser-output[\s\w\d:]+{.*?}', "", previous_news)
                            previous_news = previous_news.strip()
                            
                            if ((date_1 <= date and month_1_name == month and month_1_name != month_2_name) or (date_2 >= date and month_2_name == month and month_1_name != month_2_name) or (month_1_name == month_2_name and month_2_name  == month and date_2 >= date and  date_1 <= date) or (month != "" and monthArry.index(month_1_name) < monthArry.index(month) and  monthArry.index(month_2_name) > monthArry.index(month))):
                                if previous_news.count(" ") > 2:
                                    if month not in news_data_p:
                                        news_data_p[month] = {}
                                    if date not in news_data_p[month]:
                                        news_data_p[month][date] = []
                                    news_data_p[month][date].append(previous_news)
                                # print(f'{date} {month} \n {previous_news}')
                            
                            date = int(news[1].split(" ")[0])
                            month = news[1].split(" ")[1].strip()
                        
                        
                for news_p in news_data_p:
                    for date in news_data_p[news_p]:
                        print(f'{date} {news_p}')
                        for daily_news in news_data_p[news_p][date]:
                            print(f'\t{daily_news}')
                            


                # plot_word_cloud(lines)  # plotting the word cloud


            elif(field_requested == "2"):
                date1 = str(input("Enter the date from which you want to see the news(dd-mm-yyyy): "))
                date2 = str(input("Enter the date till which you want to see the news(dd-mm-yyyy): "))  

                month_1 = int(date1.split("-")[1])
                date_1 = int(date1.split("-")[0])
                month_1_name = monthArry[month_1-1]

                month_2 = int(date2.split("-")[1])
                date_2 = int(date2.split("-")[0])
                month_2_name = monthArry[month_2-1]
                
                response_urls = response_urls_from_time_range(date1,date2)
                print("======== RESPONSE URLS ========")
                for url in response_urls:
                    print(url)

                
                # SAVE THE RESPONSE URLS IN RESPONSE FOLDER
                if not os.path.exists("./RESPONSE"): 
                    os.mkdir("./RESPONSE")

                for i in range(0,len(response_urls)):
                    print(f'Fetching web page: {response_urls[i]}')
                    try:
                        response = requests.get(url=response_urls[i], headers=headers)
                        page_content = response.text
                        clean_html = cleanhtml(page_content)
                    except Exception as e:
                        print(e)
                        continue
                
                    response = "./RESPONSE/" + str(response_urls[i][-14:]) + ".txt"
                    with open(response, 'w',encoding="utf-8") as f:
                        f.write(clean_html)

                    
                # for i in range(0,len(response_urls)):
                #     timeline = "./RESPONSE/" + str(response_urls[i][-14:]) + ".txt"

                news_data_p = {}
                for i in range(0,len(response_urls)):
                    timeline = "./RESPONSE/" + str(response_urls[i][-14:]) + ".txt"
                    
                    with open(timeline,'r',encoding="utf-8") as f:
                        lines = str(f.readlines())
                        monthLines = re.findall(r"(.*?)([\d]+\s+January|[\d]+\s+February|[\d]+\s+March|[\d]+\s+April|[\d]+\s+May|[\d]+\s+June|[\d]+\s+July|[\d]+\s+August|[\d]+\s+September|[\d]+\s+October|[\d]+\s+November|[\d]+\s+December)",lines)
                        
                        previous_news = ""
                        date = 0
                        month = ""
                        for news in monthLines:
                            previous_news = news[0].strip()
                            previous_news = cleanhtml(previous_news)
                            previous_news = re.sub("2020 \^", "", previous_news)
                            previous_news = re.sub("2021 \^", "", previous_news)
                            previous_news = re.sub("2022 \^", "", previous_news)
                            previous_news = re.sub("2020\)", "", previous_news)
                            previous_news = re.sub("2021\)", "", previous_news)
                            previous_news = re.sub("2022\)", "", previous_news)
                            previous_news = re.sub("\[edit\]", "", previous_news)
                            previous_news = re.sub("a b c d e f g h i j", "", previous_news)
                            previous_news = re.sub("a b c d e f g h i j k l m n", "", previous_news)
                            previous_news = re.sub("a b c d e f g h i j k l m n o", "", previous_news)
                            previous_news = re.sub("a b c d e f g h i j k l m n o p q r s t", "", previous_news)
                            previous_news = re.sub("a b c d e f g h i j k l m n o p q r s t u", "", previous_news)
                            previous_news = re.sub("a b c d e f g h i j k l m n o p q r s", "", previous_news)
                            previous_news = re.sub("2020 Archived from the original on", "", previous_news)
                            previous_news = re.sub(r'mw-parser[\d\w\s,-]+{.*?}', "", previous_news)
                            previous_news = re.sub(r'mw-parser-output[\s\w\d:]+{.*?}', "", previous_news)
                            previous_news = previous_news.strip()
                            
                            if ((date_1 <= date and month_1_name == month and month_1_name != month_2_name) or (date_2 >= date and month_2_name == month and month_1_name != month_2_name) or (month_1_name == month_2_name and month_2_name  == month and date_2 >= date and  date_1 <= date) or (month != "" and monthArry.index(month_1_name) < monthArry.index(month) and  monthArry.index(month_2_name) > monthArry.index(month))):
                                if previous_news.count(" ") > 2:
                                    if month not in news_data_p:
                                        news_data_p[month] = {}
                                    if date not in news_data_p[month]:
                                        news_data_p[month][date] = []
                                    news_data_p[month][date].append(previous_news)
                                # print(f'{date} {month} \n {previous_news}')
                            
                            date = int(news[1].split(" ")[0])
                            month = news[1].split(" ")[1].strip()
                        
                
                for news_p in news_data_p:
                    for date in news_data_p[news_p]:
                        print(f'{date} {news_p}\n')
                        for daily_news in news_data_p[news_p][date]:
                            print(f'\t{daily_news}')
                    
                    # plot_word_cloud(lines)  # plotting the word cloud
                    


            elif(field_requested == "3"):
                # percentage of covid related words in common words
                print("Enter two non-overlapping time range(format dd-mm-yyyy)\n Enter the first overlapping time range ")
                date1 = str(input("Enter the date from which you want to see the news(dd-mm-yyyy): "))
                date2 = str(input("Enter the date till which you want to see the news(dd-mm-yyyy): "))  
                print("Enter the second overlapping time range")
                date3 = str(input("Enter the date from which you want to see the news(dd-mm-yyyy): "))
                date4 = str(input("Enter the date till which you want to see the news(dd-mm-yyyy): "))  

                # checking overlapping date times
                Range = namedtuple('Range', ['start', 'end'])
                start1 = datetime.datetime.strptime(date1, '%d-%m-%Y')
                end1 = datetime.datetime.strptime(date2, '%d-%m-%Y')
                start2 = datetime.datetime.strptime(date3, '%d-%m-%Y')
                end2 = datetime.datetime.strptime(date4, '%d-%m-%Y')
                
                if(start2 <=  end1):
                    print("date range is overlapping")
                    continue

                timeline_urls1 = timeline_urls_from_time_range(date1,date2)
                f_mergered1 = open("nonoverlapping_news1.txt","w",encoding="utf8")
                if not os.path.exists("./TIMELINES1"): 
                    os.mkdir("./TIMELINES1")

                timeline_urls2 = timeline_urls_from_time_range(date3,date4)
                f_mergered2 = open("nonoverlapping_news2.txt","w",encoding="utf8")
                if not os.path.exists("./TIMELINES2"): 
                    os.mkdir("./TIMELINES2")

                for i in range(0,len(timeline_urls1)):
                    print(f'Fetching web page: {timeline_urls1[i]}')
                    try:
                        response = requests.get(url=timeline_urls1[i], headers=headers)
                        page_content = response.text
                        clean_html = cleanhtml(page_content)
                      #  clean_html = remove_unnecesary_texts_from_timeline(clean_html)
                    except Exception as e:
                        print(e)
                        continue

                    timeline = "./TIMELINES1/" + str(timeline_urls1[i][-14:]) + ".txt"
                    with open(timeline, 'w',encoding="utf-8") as f:
                        f.write(clean_html)
                        
                for i in range(0,len(timeline_urls1)):
                    timeline = "./TIMELINES1/" + str(timeline_urls1[i][-14:]) + ".txt"

                    with open(timeline,'r',encoding="utf-8") as f:
                        lines = (str(f.readlines()))

                f_mergered1.write(lines)
                f_mergered1.close()

                for i in range(0,len(timeline_urls2)):
                    print(f'Fetching web page: {timeline_urls2[i]}')
                    try:
                        response = requests.get(url=timeline_urls2[i], headers=headers)
                        page_content = response.text
                        clean_html = cleanhtml(page_content)
                       # clean_html = remove_unnecesary_texts_from_timeline(clean_html)
                    except Exception as e:
                        print(e)
                        continue

                    timeline = "./TIMELINES2/" + str(timeline_urls2[i][-14:]) + ".txt"
                    with open(timeline, 'w',encoding="utf-8") as f:
                        f.write(clean_html)
                        
                for i in range(0,len(timeline_urls2)):
                    timeline = "./TIMELINES2/" + str(timeline_urls2[i][-14:]) + ".txt"

                    with open(timeline,'r',encoding="utf-8") as f:
                        lines = (str(f.readlines()))

                f_mergered2.write(lines)
                f_mergered2.close()

                common_words = get_common_words("nonoverlapping_news1.txt")
                covid_dict = get_covid_dict()
                percen = percentage_of_covid_words_in_common_words(common_words,covid_dict)
                print("\n\n Percentage of covid words in common words ",percen)

                covid_word_in_text = covid_related_common_words(common_words,covid_dict)
                top20common = top_20_words(common_words)
                top20covid = top_20_words(covid_word_in_text)
                print("\n\nPrinting top 20 common_words")
                for word in top20common:
                    print(word)
                print("\n\nPrinting top 20 covid words")
                for word in top20covid:
                    print(word)

                # unique_string=(" ").join(my_list)
                filtered_common = filter_text(common_words) # removing stop words
                plot_word_cloud((" ").join(filtered_common))
                plot_word_cloud((" ").join(covid_word_in_text))

            elif(field_requested == "4"):
                # finding the time range
                coun = str(input("Enter the country(country name in lower case): ") )
                print(date_range(coun))

            elif(field_requested == "5"):
                fetch_country_from_worldometer()

                country = str(input("Enter the country(country name in lower case from the above list): ") )

                country_path = "./a4_COUNTRIES/" + country + ".html"
                text = open(country_path, 'r', errors = 'ignore').read()

                lexer = lex.lex()
                parser = yacc.yacc()
                parser.parse(text)

                print("Total cases in "+ country+" = "+ total_cases[-1])
                print("Total deaths in "+ country+" = "+ total_deaths[-1])
                print("Total recovered in "+ country+" = "+ total_recovered[-1])
 

            elif(field_requested == "6"):
                sys.exit(0)



if __name__ == "__main__":
	main()
