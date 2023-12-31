import settings
from flask import Flask
from views import router

app = Flask(__name__)
app.secret_key = "secret key"
app.register_blueprint(router)
app.static_folder = "static" ### read css and js files

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=settings.API_DEBUG)