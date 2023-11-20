from flask import Flask
from flask_cors import CORS, cross_origin

import json

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/map')
@cross_origin()
def returnstuff():
    with open("./articles.json", "r") as f:
        articles = json.load(f)

    smallArticles = [
        {
            "id": article["id"],
            "title": article["title"],
            "description": article["description"],
            "source": article["source"],
            "profile": article["profile"],
            "publish_date": article["publish_date"],
            "ml": article["ml"],
        }
        for article in articles
    ]
    return smallArticles

if __name__ == '__main__':
    app.run(debug=True)
