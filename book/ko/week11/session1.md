# 11주차 1교시: 웹 개발과 Flask 입문

## 소개

웹 개발 강의에 오신 것을 환영합니다! 오늘은 웹의 작동 원리를 살펴보고, 파이썬 기반의 경량 웹 프레임워크인 Flask를 사용하여 웹 애플리케이션을 구축하는 방법을 배워보겠습니다. 이번 수업을 통해 클라이언트-서버 아키텍처, 프론트엔드와 백엔드 개발의 역할을 이해하고, 기본적인 라우팅과 템플릿을 포함한 첫 Flask 애플리케이션을 만들어볼 것입니다.

---

## 1. 웹 개발의 기초

### 1.1 클라이언트-서버 아키텍처

- **정의**: 다수의 클라이언트(사용자)가 중앙 서버에 서비스를 요청하고 받는 모델
- **작동 방식**:
  - **클라이언트**: 요청을 보내는 사용자의 웹 브라우저나 애플리케이션
  - **서버**: 요청을 처리하고 응답을 보내는 원격 컴퓨터
- **예시**:
  - 브라우저에 URL을 입력하면 여러분의 컴퓨터(클라이언트)가 서버에 요청을 보내고, 서버는 웹 페이지를 응답으로 전송합니다.

### 1.2 HTTP/HTTPS 프로토콜 개요

- **HTTP (HyperText Transfer Protocol)**:
  - 웹에서 데이터 통신의 기반
  - GET, POST, PUT, DELETE 등의 메서드를 사용하여 리소스와 상호작용
- **HTTPS (HTTP Secure)**:
  - HTTP의 보안 확장 버전
  - SSL/TLS 암호화를 사용하여 안전한 통신 보장
- **중요성**:
  - 메시지의 형식과 전송 방식을 정의
  - 웹 서버와 브라우저가 다양한 명령에 어떻게 응답해야 하는지 결정

### 1.3 프론트엔드와 백엔드 개발의 역할

- **프론트엔드 개발**:
  - 사용자 인터페이스와 사용자 경험을 다룸
  - 사용 기술: HTML, CSS, JavaScript
  - 책임: 레이아웃, 시각적 요소, 상호작용 요소 설계
- **백엔드 개발**:
  - 서버 측 로직과 통합에 중점
  - 사용 기술: Python, Java, Ruby, 데이터베이스
  - 책임: 애플리케이션 로직, 데이터베이스, 사용자 인증 관리

### 1.4 RESTful API 개념

- **REST (Representational State Transfer)**:
  - 네트워크 애플리케이션 설계를 위한 아키텍처 스타일
- **원칙**:
  - **무상태성**: 각 클라이언트 요청은 필요한 모든 정보를 포함해야 함
  - **통일된 인터페이스**: 리소스는 요청에서 식별됨 (예: URI를 통해)
  - **표현을 통한 리소스 조작**: 클라이언트는 표현(예: JSON, XML)을 사용하여 리소스를 수정할 수 있음
- **HTTP 메서드와 REST**:
  - **GET**: 데이터 조회
  - **POST**: 새 데이터 생성
  - **PUT**: 기존 데이터 수정
  - **DELETE**: 데이터 삭제

---

## 2. Flask 입문

### 2.1 Flask 프레임워크 개요 및 설치

- **Flask란?**
  - 파이썬으로 작성된 마이크로 웹 프레임워크
  - 단순성과 사용 용이성으로 유명
- **특징**:
  - 경량화되고 모듈화된 구조
  - 다양한 확장을 통한 확장성
- **설치**:

  ```bash
  pip install flask
  ```

### 2.2 첫 Flask 애플리케이션 만들기

- **기본 애플리케이션 구조**:

  ```python
  from flask import Flask

  app = Flask(__name__)

  @app.route('/')
  def home():
      return '안녕하세요!'

  if __name__ == '__main__':
      app.run(debug=True)
  ```

- **설명**:
  - Flask 클래스 임포트
  - Flask 애플리케이션 인스턴스 생성
  - URL '/'에 대한 라우트 정의
  - 애플리케이션 실행

### 2.3 라우트와 데코레이터 이해하기

- **라우트**:
  - 사용자가 접근할 수 있는 URL
  - `@app.route()` 데코레이터를 사용하여 정의
- **데코레이터**:

  - 함수의 동작을 수정
  - Flask에서는 URL을 함수에 바인딩하는 데 사용

- **예시**:

  ```python
  @app.route('/about')
  def about():
      return '소개 페이지입니다.'
  ```

### 2.4 다양한 HTTP 메서드 처리하기 (GET, POST)

- **메서드 지정**:

  ```python
  @app.route('/submit', methods=['GET', 'POST'])
  def submit():
      if request.method == 'POST':
          # POST 요청 처리
          pass
      else:
          # GET 요청 처리
          pass
  ```

- **요청 데이터 접근**:

  ```python
  from flask import request

  data = request.form['key']  # 폼 데이터
  data = request.args.get('key')  # 쿼리 파라미터
  ```

---

## 3. 기본 템플릿 렌더링

### 3.1 Jinja2 템플릿 엔진 소개

- **Jinja2란?**
  - 파이썬용 템플릿 엔진
  - HTML에 파이썬 스타일의 표현식을 포함할 수 있음
- **특징**:
  - 템플릿 상속
  - 제어 구조 (반복문, 조건문)
  - 필터와 매크로

### 3.2 HTML 템플릿 생성과 렌더링

- **폴더 구조**:

  - 템플릿은 `templates` 디렉토리에 저장

- **템플릿 예시 (`templates/home.html`)**:

  ```html
  <!DOCTYPE html>
  <html>
    <head>
      <title>{{ title }}</title>
    </head>
    <body>
      <h1>{{ title }}에 오신 것을 환영합니다</h1>
    </body>
  </html>
  ```

- **템플릿 렌더링**:

  ```python
  from flask import render_template

  @app.route('/')
  def home():
      return render_template('home.html', title='내 Flask 앱')
  ```

### 3.3 템플릿 상속과 재사용

- **기본 템플릿 (`templates/base.html`)**:

  ```html
  <!DOCTYPE html>
  <html>
    <head>
      <title>{% block title %}{% endblock %}</title>
    </head>
    <body>
      {% block content %}{% endblock %}
    </body>
  </html>
  ```

- **자식 템플릿 (`templates/home.html`)**:

  ```html
  {% extends 'base.html' %} {% block title %}홈페이지{% endblock %} {% block
  content %}
  <h1>홈페이지에 오신 것을 환영합니다</h1>
  {% endblock %}
  ```

### 3.4 템플릿에 변수 전달하기

- **데이터 전달**:

  ```python
  @app.route('/user/<username>')
  def profile(username):
      return render_template('profile.html', username=username)
  ```

- **템플릿에서 변수 사용**:

  ```html
  <h1>사용자 프로필: {{ username }}</h1>
  ```

---

## 실습

### 4.1 Flask 개발 환경 설정

- **가상환경 설정**:

  ```bash
  python -m venv venv
  source venv/bin/activate  # Windows의 경우 venv\Scripts\activate
  ```

- **Flask 설치**:

  ```bash
  pip install flask
  ```

### 4.2 간단한 "Hello, World!" 애플리케이션 만들기

- **애플리케이션 (`app.py`)**:

  ```python
  from flask import Flask

  app = Flask(__name__)

  @app.route('/')
  def hello():
      return '안녕하세요!'

  if __name__ == '__main__':
      app.run(debug=True)
  ```

- **애플리케이션 실행**:

  ```bash
  python app.py
  ```

- **브라우저에서 접속**:
  - `http://localhost:5000/` 접속

### 4.3 여러 엔드포인트를 가진 라우트 구현하기

- **라우트 추가**:

  ```python
  @app.route('/about')
  def about():
      return '소개 페이지'

  @app.route('/contact')
  def contact():
      return '연락처 페이지'
  ```

### 4.4 기본 템플릿 기반 웹 페이지 구축하기

- **템플릿 생성 (`templates/index.html`)**:

  ```html
  <!DOCTYPE html>
  <html>
    <head>
      <title>Flask 템플릿</title>
    </head>
    <body>
      <h1>{{ message }}</h1>
    </body>
  </html>
  ```

- **템플릿을 렌더링하도록 라우트 수정**:

  ```python
  @app.route('/')
  def home():
      return render_template('index.html', message='Flask와 Jinja2로부터 안녕하세요!')
  ```

---

## 결론

이번 수업에서는 웹 개발의 기초를 다루었습니다. 클라이언트-서버 모델과 HTTP 프로토콜을 이해하고, 프론트엔드와 백엔드 개발의 차이점을 배웠으며, RESTful API에 대해 알아보았습니다. 실습을 통해 Flask 개발 환경을 설정하고, 간단한 웹 서버를 만들어보았으며, Jinja2를 사용한 라우팅과 템플릿 처리 방법을 학습했습니다.

---

## 다음 수업 예고

다음 수업에서는 Flask의 더 고급 기능들을 살펴볼 예정입니다. 폼 처리, 데이터베이스 통합, 사용자 인증 등을 다룰 것이며, AI와 데이터 사이언스 애플리케이션을 최소한의 코드로 빠르게 개발할 수 있는 Streamlit 프레임워크도 소개할 예정입니다.

---

**추천 자료와 리소스**:

- [Flask 공식 문서](https://flask.palletsprojects.com/)
- [Jinja2 템플릿 디자이너 문서](https://jinja.palletsprojects.com/)
- [HTTP 프로토콜 개요](https://developer.mozilla.org/ko/docs/Web/HTTP/Overview)
- [RESTful API 설계](https://restfulapi.net/)

---

**연습문제**:

1. Flask 애플리케이션을 확장하여 `/user/<name>` 라우트를 추가하고, 사용자 이름으로 인사하는 기능을 구현해보세요.
2. 기본 템플릿을 만들고 템플릿 상속을 사용하여 일관된 레이아웃을 가진 여러 페이지를 만들어보세요.
