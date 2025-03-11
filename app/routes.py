from flask import render_template, request
from recommend import recommend_movies
from app import app

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = None
    if request.method == 'POST':
        user_id = int(request.form['user_id'])
        recommendations = recommend_movies(user_id)
    return render_template('index.html', recommendations=recommendations)