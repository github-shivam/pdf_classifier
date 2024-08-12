import requests
from io import BytesIO
from pdfminer.high_level import extract_text
from pdfminer.pdfparser import PDFSyntaxError
from pdf2image import convert_from_bytes
import pytesseract
import joblib
import pytesseract
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.preprocessing import LabelEncoder

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

Lemmatizer = WordNetLemmatizer()
vectorizer = TfidfVectorizer()
label_encoder = LabelEncoder()

vectorizer = joblib.load('tfidf_vectorizer.pkl')
model = joblib.load('best_classifier.sav')
label_encoder = joblib.load('label_encoder.pkl')

def extract(pdf_url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }

        session = requests.Session()

        response = requests.get(pdf_url, headers=headers, timeout = 30)

        if response.status_code == 200:
            pdf_file = BytesIO(response.content)

            try:

                text = extract_text(pdf_file)

                if text.strip():
                    return text
                
                else:
                    raise ValueError("Empty text extracted by PDFMiner, falling back to OCR.")
                
            except (PDFSyntaxError, ValueError) as e:
                print(f"PDFMiner failed or empty text extracted. Falling back to OCR for URL: {pdf_url}. Error: {e}")

                images = convert_from_bytes(response.content)

                text = ""

                for image in images:
                    text += pytesseract.image_to_string(image)
                
                return text
            
        else:
            print(f"Failed to retrieve PDF from {pdf_url}. Status code: {response.status_code}")
            return "Failed to retrieve PDF"
    
    except requests.exceptions.Timeout:
        print(f"Request timed out for URL: {pdf_url}")
        return None
        
    except Exception as e:
            print(f"An error occurred while fetching the PDF from {pdf_url}: {e}")
            return "Error fetching PDF"




def remove_users(text):
    clean_text = re.sub('(@[a-zA-Z]+[a-zA-Z0-9-_]+)', '', text)
    return clean_text

# remove links
def remove_links(text):
    clean_text = re.sub(r'http\S+', '', text)
    clean_text = re.sub(r'bit.ly/\S+', '', text)
    clean_text = clean_text.strip('[link]')
    return clean_text

# remove non ascii character

def non_ascii(text):
    return "". join(i for i in text if ord(i) < 128)

# lowercase

def lower_case(text):
    return text.lower()

# remove stop words

def remove_stopwords(text):
    cachedStopWords = set(stopwords.words('english'))
    cachedStopWords.update(('and','I','A','http','And','So','arnt','This','When','It','many','Many','so','cant','Yes','yes','No','no','These','these','mailto','regards','ayanna','like','email'))
    clean_text = ' '.join(word for word in text.split() if word not in cachedStopWords)
    return clean_text

#remove email address

def remove_email(text):
    clean_text = re.compile(r'[\w\.-]+@[\w\.-]+')
    return clean_text.sub(r'', text)

# remove punctuation

def punct(text):
    token = RegexpTokenizer(r'\w+')
    clean_text = token.tokenize(text)
    clean_text = ' '.join(clean_text)
    return clean_text

# remove digits and special characters

def remove_digits(text):
    pattern = r'[^a-zA-z.,!?/:;\"\'\s]'
    return re.sub(pattern, '', text)

def remove_special_charcter(text):
    pat = r'[^a-zA-z0-9.,!?/:;\"\'\s]+[_]+'
    return re.sub(pat, '', text)

# remove underscore

def remove_(text):
    clean_text = re.sub('([_]+)', '', text)
    return clean_text

def preprocess_input(text):

    c_text = remove_digits(text)
    c_text = remove_links(c_text)
    c_text = remove_email(c_text)
    c_text = remove_special_charcter(c_text)
    c_text = remove_users(c_text)
    c_text = remove_stopwords(c_text)
    c_text = remove_(c_text)
    c_text = punct(c_text)
    c_text = lower_case(c_text)
    c_text = non_ascii(c_text)
    
    words = word_tokenize(c_text)

    lemmatized_text = " ". join([Lemmatizer.lemmatize(word) for word in words])

    transformed_text = vectorizer.transform([lemmatized_text])

    return transformed_text


def prediction(link):
    extracted_text = extract(link)

    c_text = remove_digits(extracted_text)
    c_text = remove_links(c_text)
    c_text = remove_email(c_text)
    c_text = remove_special_charcter(c_text)
    c_text = remove_users(c_text)
    c_text = remove_stopwords(c_text)
    c_text = remove_(c_text)
    c_text = punct(c_text)
    c_text = lower_case(c_text)
    c_text = non_ascii(c_text)
    
    words = word_tokenize(c_text)

    lemmatized_text = " ". join([Lemmatizer.lemmatize(word) for word in words])

    transformed_text = vectorizer.transform([lemmatized_text])


    class_prob = model.predict_proba(transformed_text)[0]

    predicted_class_index = class_prob.argmax()
    predicted_class_name = label_encoder.inverse_transform([predicted_class_index])[0]

    predicted_class_prob = class_prob[predicted_class_index]

    class_prob_list = class_prob.tolist()

    print(f"Predicted Class: {predicted_class_name}")
    print(f"Predicted Class Probability: {predicted_class_prob:.2f}")


    for class_index, probability in enumerate(class_prob_list):
        class_name = label_encoder.inverse_transform([class_index])[0]
        print(f"Class: {class_name}, Probability: {probability:.2f}")


    return {
        'predicted_class': predicted_class_name,
        'predicted_class_probability': predicted_class_prob,
        'class_probabilities': class_prob_list  # Return as list
    }