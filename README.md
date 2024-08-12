# pdf_classifier

This PDF Classifier is a tool that takes a URL of a PDF document, extracts the text, and classifies it into one of four categories: Lighting, Fuses, Cables, or Others. The model behind this classifier is XGBoost, which achieves an impressive 93% accuracy. I've also created an API and deployed it on Render for easy access. You can use this API in two simple ways:

1 Using curl: Open your terminal and run the following command:

curl -X POST https://pdf-classifier.onrender.com/predict -H "Content-Type: application/json" -d "{\"pdf_url\": \"your_pdf_link\"}"


2 Using Python: Create a Python script with the code below, then run it from your command prompt:

import requests

url = "https://pdf-classifier.onrender.com/predict"
data = {
    "pdf_url": "your_pdf_link"
}

response = requests.post(url, json=data)
print(response.json())


Simply replace your_pdf_link with the URL of the PDF you want to classify, and you'll get the results in no time!

Thank you for visiting!
