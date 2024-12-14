from flask import Flask
from flask_assets import Environment, Bundle
from app.routes import main

def create_app():
    app = Flask(__name__)

    # Flask-Assets 설정
    assets = Environment(app)
    app.register_blueprint(main)

    # SCSS settings
    scss = Bundle('scss/styles.scss', filters='libsass', output='css/styles.css')
    assets.register('scss_all', scss)
    scss.build()

    return app
