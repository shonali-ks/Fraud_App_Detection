from flask import Flask,render_template,url_for,request
from google_play_scraper import app as gs
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    score=0
    if request.method == 'POST':
        url = request.form['url']
        file = open("cc.txt", "w",encoding='utf-8')
        link=url
        findId=link.find('id=')

        url=link[findId+3:]
        file.write(str(gs(
            url,
            lang='en', # defaults to 'en'
            country='us' # defaults to 'us'
        )))
        file.close()

        myfile=[]
        with open("cc.txt",encoding='utf8') as mydata:
            for data in mydata:
                myfile.append(data)

        start=myfile[0].find('comments')
        end=myfile[0].find('editorsChoice')
        c=data[start:end]

        cleandata = re.sub('[^A-Za-z0-9]+',' ',c)
        low=cleandata.lower()

        stop=set(stopwords.words('english'))
        wordstoken=word_tokenize(low)

        sentences=[w for w in wordstoken if not w in stop]
        sentences=[]


        for w in wordstoken:
            if w not in stop:
                sentences.append(w)

        total=0
        tot=0
        positive = open("positive.txt", "r",encoding='utf-8')
        negative = open("negative.txt", "r",encoding='utf-8')
        pos=positive.read().split()
        neg=negative.read().split()
        for word in sentences:
        #     print(word)
            tot=tot+1
            if word in pos:
                total=total+1
        #         print("good: "+word)
            if word in neg:
                total=total-1
        #         print("bad: "+word)
            
        score=total/tot

    return render_template('result.html',score = score)



if __name__ == '__main__':
	app.run(debug=True)