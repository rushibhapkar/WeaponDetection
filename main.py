
from flask import Flask, render_template
app = Flask(__name__)


@app.route("/")
def home():
    return render_template('Home.html')


@app.route("/run")
def run():
    import infere.py
    return render_template(infere.net)


# @app.route("/contact")
# def contact():
#     return render_template('contact.html')


app.run(debug=True)
