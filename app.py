# import necessary laibraries
import webbrowser
import nltk
from nltk.stem import WordNetLemmatizer

import pickle
import numpy as np
from keras.models import load_model
import json
import random

import streamlit as st
# from chatbot_functions import predict_class, getResponse
import os
import pickle
import time
import re
import tensorflow as tf
from PIL import Image
import requests
from bs4 import BeautifulSoup

##################### PLOT DIAGRAM ########################


###################### END PLOT ##########################
logo_image = Image.open("diulogologo.png")
lemmatizer = WordNetLemmatizer()
model = load_model('model.h5')
# intents = json.loads(open('generalQuestion.json').read())
# Load intents all files
diubot_intents = json.loads(open('generalQuestion.json').read())['intents']
varsity_location = json.loads(open('varsity_locations.json').read())['intents']
diuCofc_intents = json.loads(open('diuCofc.json').read())['intents']
tution_fee_data = json.loads(open('tuition_fee_data.json').read())['intents']
international = json.loads(open('international&career.json').read())['intents']

admission = json.loads(open('admission.json').read())['intents']
campus = json.loads(open('campus.json').read())['intents']
teacher = json.loads(open('teacher_info.json').read())['intents']
department_office = json.loads(open('department_office_info.json').read())['intents']

# Combine the intents from both files
combined_intents = diubot_intents + diuCofc_intents + tution_fee_data + varsity_location + admission + campus + teacher + department_office + international

words = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))
################################# MANAGE HERE UNKNOW QUESTION ######################################
CONFIDENCE_THRESHOLD = 0.5
AUDIENCE_MANAGEMENT_MESSAGE = (
    "I'm sorry, still I don't have information on that topic. If you have a specific query or need assistance, "
    "please fill out this form, and one of our team members will get back to you shortly: [Google Form Link]"
)
GOOGLE_FORM_LINK = "https://forms.gle/VtipgNdwGtFBNyzG9"
################################# END MANAGE HERE UNKNOW QUESTION ######################################

################################ ADMISSION NOTICE ###############################################
import requests
from bs4 import BeautifulSoup


def scrape_admission_notice(url):
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')

        title_tag = soup.find('h6')
        admission_title = title_tag.text.strip() if title_tag else "Admission Notice Not Found"

        details_div = soup.find('div', class_='text_box')

        if details_div:

            details = [p.text.strip() for p in details_div.find_all('p')]
        else:
            details = ["Details not found"]

        return {
            'admission_title': admission_title,
            'details': details
        }
    else:
        print(f"Sorry I have no last information.")
        return None


# Function to fetch the latest admission notice
def fetch_latest_admission_notice():
    url = "https://daffodilvarsity.edu.bd/admission"
    return scrape_admission_notice(url)


latest_admission_notice = fetch_latest_admission_notice()

if latest_admission_notice:
    print(f"Title: {latest_admission_notice['admission_title']}")
    for detail in latest_admission_notice['details']:
        print(detail)
else:
    print("No information available.")

# Function to fetch the latest admission notice


################################## END ADMISSION NOTICE #######################################

################################## VARSITY RANKING  #######################################
import requests
from bs4 import BeautifulSoup


def varsity_ranking(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    div_tags = soup.find_all('div', class_='col-md-6 border-top border-start border-end')

    rank_details_list = []

    for div_tag in div_tags[:3]:
        data = div_tag.text.strip()
        rank_details_list.append({
            'rank': data,
            'link': None
        })

    return rank_details_list


def fetch_varsity_ranking():
    url = "https://daffodilvarsity.edu.bd/rankings"
    return varsity_ranking(url)


#
# rankings = fetch_varsity_ranking()


################################## END VARSITY RANKING  #######################################

# preprocessing user input
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)


# predicting the intent of user input
def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.5
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


# generate bot response
# def getResponse(ints, intents_json):
#     tag = ints[0]['intent']
#     list_of_intents = intents_json ['intents']
#     result = ""
#     for i in list_of_intents:
#         if i['tag'] == tag:
#
#             result = random.choice(i['responses'])
#             break
#     return result
def getResponse(ints, combined_intents):
    tag = ints[0]['intent']
    result = ""
    for i in combined_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


# def chatbot_response(msg,intents):
#     ints = predict_class(msg, model)
#     res = getResponse(ints, intents)
#     return res

# Function to fetch the latest admission notice
##################################Spelling mistake handle###################################
from spellchecker import SpellChecker

# ... (your existing code)

spell = SpellChecker()

# Define a list of terms that should not be corrected
# Define a list of terms that should not be corrected
no_correction_terms = ["diu", "vc", "nrc", "rkr", "mta", "fh", "sah", "srh", "mhs", "fkm", "pro-vc", "bot"]


def correct_spelling(query):
    words = query.split()
    corrected_words = []

    for word in words:
        if word.lower() in no_correction_terms:
            corrected_words.append(word)
        else:
            corrected_word = spell.correction(word)
            if corrected_word is not None:
                corrected_words.append(corrected_word)

    #
    for i in range(1, len(corrected_words) - 1):
        context_correction = spell.correction(corrected_words[i - 1] + corrected_words[i] + corrected_words[i + 1])
        if context_correction is not None:
            corrected_words[i] = context_correction

    return ' '.join(corrected_words)


def chatbot_response(user_input, intents):
    # user_input = correct_spelling(user_input)

    admission_keywords = ["admission notice", "upcoming admission", "next admission", "next semester"]
    ranking_keywords = ["diu ranking", "varsity ranking", "daffodil ranking", "rank"]
    if any(keyword in user_input.lower() for keyword in admission_keywords):
        latest_admission_notice = fetch_latest_admission_notice()

        if latest_admission_notice:
            return latest_admission_notice
        else:

            return "Sorry I have no idea."

    elif any(keyword in user_input.lower() for keyword in ranking_keywords):
        rankings = fetch_varsity_ranking()

        if rankings:
            return rankings
        else:
            return "Sorry I have no idea."

    else:

        ints = predict_class(user_input, model)

        if not ints or float(ints[0]['probability']) < CONFIDENCE_THRESHOLD:
            return AUDIENCE_MANAGEMENT_MESSAGE.replace("[Google Form Link]", GOOGLE_FORM_LINK)

        # Sort intents by confidence level
        ints.sort(key=lambda x: float(x['probability']), reverse=True)

        # Check if the most confident intent is below the confidence threshold
        # if float(ints[0]['probability']) < CONFIDENCE_THRESHOLD:
        #     return AUDIENCE_MANAGEMENT_MESSAGE.replace("[Google Form Link]", GOOGLE_FORM_LINK)

        response = getResponse(ints, combined_intents)
        return response


############################################## STREAMLIT DESING #########################################
from streamlit_option_menu import option_menu
import time


def show_home():
    centered_style = """
        <style>
        .centered {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 40vh;
        }
        </style>
    """

    st.markdown(centered_style, unsafe_allow_html=True)

    # st.markdown('<div class="centered"><h1>Diubot</h1></div>', unsafe_allow_html=True)
    # Specify the path to your image
    image_path = "diu.png"

    # Create columns for layout
    col1, col2, col3 = st.columns(3)

    # Center the image in the middle column
    with col2:
        st.image(image_path, width=154)

    # Center the text in the last column

    # Create three columns
    col1, col2, col3 = st.columns(3)

    # Display messages in each column

    with col1:
        st.header("Welcome")
        st.markdown(
            " Curious about Daffodil? Just type your question, and I'll share interesting details with you.")

    with col2:
        st.header("Ask Anything!")
        st.markdown(
            "Explore topics like admissions, tuition fees, campus facilities, and more. Simply type your query, and I'll guide you through.")

    with col3:
        st.header("Attention!")
        st.markdown("If you get the wrong answer, please try again.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Accept user input
    user_input = st.chat_input("Write here", key="dbot")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message(name="diubot", avatar="diulogologo.png"):
            message_placeholder = st.empty()
            hello_response = ""
            full_response = chatbot_response(user_input, combined_intents)

            if isinstance(full_response, dict):  # Check if it's a dictionary (admission notice)
                st.write(f"Title: {full_response['admission_title']}")
                for detail in full_response['details']:
                    st.write(detail)

            elif isinstance(full_response, list):
                for ranking in full_response:
                    st.write("Rank:", ranking['rank'])
                    st.write()


            else:
                for chunk in full_response.split():
                    hello_response += chunk + " "
                    time.sleep(0.05)
                    # Add a blinking cursor to simulate typing
                    message_placeholder.markdown(hello_response + "▌")

                line_breaks = full_response.replace('\n', '<br>')
                message_placeholder.markdown(line_breaks, unsafe_allow_html=True)

                # Add assistant response to chat history
                st.session_state.messages.append({"role": "D", "content": full_response})


# Sidebar
# page = st.sidebar.selectbox("Select Page", ["Home", "Notice", "Contacts"])

def show_notice():
    import streamlit as st
    import requests
    from bs4 import BeautifulSoup

    # Function to scrape notices from a website
    def scrape_notices(url):
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        notices = {}
        for notice_tag in soup.find_all('div', class_='row'):
            title_tag = notice_tag.find('h3')
            if title_tag:
                title = title_tag.find('a').text.strip()
                link = title_tag.find('a')['href']
                content_tag = notice_tag.find('p')
                content = content_tag.text.strip() if content_tag else "No content available."
                notices[title] = {'content': content, 'link': link}

        return notices

    # Sample website URL for each department
    department_urls = {
        "CSE": "https://daffodilvarsity.edu.bd/department/cse/notice",
        "SWE": "https://daffodilvarsity.edu.bd/department/swe/notice",
        "EEE": "https://daffodilvarsity.edu.bd/department/eee/notice",
        "TE": "https://daffodilvarsity.edu.bd/department/te/notice"

    }

    def main():
        st.title("Departmental Notice Board")

        selected_department = st.selectbox("Select a Department", list(department_urls.keys()))

        notices = scrape_notices(department_urls[selected_department])

        st.write(f"### {selected_department} Notices")
        for notice_title, notice_data in list(notices.items())[-3:]:
            st.write(f"#### {notice_title}")
            st.write(f"Link: [{notice_title}]({notice_data['link']})")
            st.write(notice_data['content'])

    main()


def show_contacts():
    st.title("DIU Contacts")

    # DIU Important Contacts
    contacts = {
        "ADMISSION OFFICE": "Phone: +8809617901212,  Email: admission@daffodilvarsity.edu.bd",
        "REGISTER OFFICE": "Phone: +8802224441833, +8802224441834, Cell: 01713493011, Email: registraroffice@daffodilvarsity.edu.bd",
        "IT SUPPORT": (
            "Location: Academic Building - 1, 2, 3\n"
            "Phone: 01847140136\n"
            "Email: it6@daffodilvarsity.edu.bd\n\n"
            "Location: Academic Building - 4\n"
            "Phone: 01847140138\n"
            "Email: it7@daffodilvarsity.edu.bd\n\n"
            "Location: Engineering Complex\n"
            "Phone: 01847140063\n"
            "Email: it8@daffodilvarsity.edu.bd"
        ),
        "STUDENTS' AFFAIRS": "Social Connection: https://www.facebook.com/diudsa/ ,Email: dsa@daffodilvarsity.edu.bd",
        "MEDICAL SERVICES": "Phone:01847140120, Ambulance Hotline:  01847334999, Email: diumc@daffodilvarsity.edu.bd",
        "LIBRARY": "Phone:01847-140068, Email: library@daffodilvarsity.edu.bd  ",
        "FINANCIAL AID & SCHOLARSHIPS": "Call: +88 02 9136694, Email: mzaman@daffodilvarsity.edu.bd",
        "CAREER DEVELOPMENT CENTER": "Cell: +8801847334707, Ext. 65571 (IP), E-mail: cdc@daffodilvarsity.edu.bd",
        "INTERNATIONAL  AFFAIRS": "Cell/WhatsApp: +8801811458865, Email: int@daffodilvarsity.edu.bd",
        "HALL SERVICES": "Male Hall(YKSG-1):01847334956 ,Male Hall(YKSG-2):01896034256,Female Hall(RASG):01896034255,Email: hall@daffodilvarsity.edu.bd",
        "MEET THE PSYCHOLOGIST": "Contact: 01847140065, Email: psychologist@daffodilvarsity.edu.bd,Contact: 01847334932,Email: psychologist2@daffodilvarsity.edu.bd ",
        "TRANSPORTATION SERVICES": "Call:01847-140037, Transport Supervisor:+8801713493083"
    }

    st.header("Important Contacts")

    for title, content in contacts.items():
        st.markdown(f"**{title}**: {content}")
    st.write("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")


# sidebar
with st.sidebar:
    page = option_menu(
        menu_title="Diubot",
        options=["Home", "Notice", "Contacts"],
        icons=["house", "book", "envelope"],
        menu_icon="cast",
        default_index=0,
    )
    st.write("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
    st.markdown("<div style='text-align: center;font-size: small;'><b>© 2024 Shahruk. All rights reserved.</b></div>",
                unsafe_allow_html=True)

# Display content based on user selection
if page == "Home":
    show_home()
elif page == "Notice":
    show_notice()
elif page == "Contacts":
    show_contacts()

###################### PLOT DIAGRAM ###############################


######################################### EXTRA SOMETHING############################################
