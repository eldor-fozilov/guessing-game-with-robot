from flask import Flask
from flask_assets import Environment, Bundle
from app.routes import main

def create_app():
    app = Flask(__name__)

    # Flask-Assets 설정
    assets = Environment(app)
    app.register_blueprint(main)

    # SCSS 파일 설정
    scss = Bundle('scss/styles.scss', filters='libsass', output='css/styles.css')
    assets.register('scss_all', scss)
    scss.build()  # 서버 시작 시 SCSS를 컴파일

    return app
