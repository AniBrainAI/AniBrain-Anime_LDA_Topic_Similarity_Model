from flask import Flask, render_template,request,url_for, jsonify
from recommender import AnimeRecommender

app = Flask(__name__)
model = AnimeRecommender()

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    title = data['title'] if 'title' in data else None
    synopsis = data['synopsis'] if 'synopsis' in data else None
    synopsis_genres = data['synopsis_genres'] if 'synopsis_genres' in data else None
    measure = data['measure'] if 'measure' in data else None
    amount = data['amount'] if 'amount' in data else 100
    
    recs = model.recommend(title=title, synopsis=synopsis, synopsis_genres=synopsis_genres, measure=measure, k=amount)

    return jsonify(recs)

if __name__ == '__main__':
    app.run(host='0.0.0.0')