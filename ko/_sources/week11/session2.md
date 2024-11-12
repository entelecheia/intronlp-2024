# 11주차 세션 2: Flask 심화 학습과 Streamlit 입문

## 소개

다시 만나서 반갑습니다! 오늘은 폼 처리, 세션 관리, 파일 업로드 등 Flask의 고급 기능들을 살펴보겠습니다. 또한 최소한의 노력으로 대화형 데이터 기반 웹 애플리케이션을 구축할 수 있는 강력한 프레임워크인 Streamlit을 소개할 예정입니다. 이번 수업을 통해 더 동적인 웹 애플리케이션과 대화형 대시보드를 만들 수 있게 될 것입니다.

---

## 1. Flask 고급 기능

### 1.1 폼 처리와 유효성 검사

#### 1.1.1 Flask에서 폼 다루기

- **웹 애플리케이션의 폼**:
  - 사용자 입력을 수집하는 데 사용
  - 일반적으로 `GET`과 `POST` 메서드 사용
- **폼 설정하기**:

  - `<form>` 태그를 사용하여 HTML 폼 생성
  - `action`과 `method` 속성 지정

  ```html
  <form action="/submit" method="post">
    <input type="text" name="username" placeholder="이름을 입력하세요" />
    <input type="submit" value="제출" />
  </form>
  ```

#### 1.1.2 폼 데이터 접근하기

- **`request` 객체 사용**:

  ```python
  from flask import request

  @app.route('/submit', methods=['POST'])
  def submit():
      username = request.form['username']
      return f'안녕하세요, {username}님!'
  ```

- **GET 요청 처리하기**:

  ```python
  @app.route('/search')
  def search():
      query = request.args.get('q')
      return f'검색 결과: {query}'
  ```

#### 1.1.3 폼 유효성 검사

- **왜 폼 유효성 검사가 필요한가?**
  - 데이터 무결성 보장
  - 보안 취약점(예: SQL 인젝션) 방지
- **수동 유효성 검사**:

  ```python
  if not username:
      error = '사용자 이름은 필수입니다.'
      return render_template('form.html', error=error)
  ```

- **Flask-WTF 확장 사용하기**:

  - **설치**:

    ```bash
    pip install flask-wtf
    ```

  - **폼 클래스 생성**:

    ```python
    from flask_wtf import FlaskForm
    from wtforms import StringField, SubmitField
    from wtforms.validators import DataRequired

    class NameForm(FlaskForm):
        username = StringField('사용자 이름', validators=[DataRequired()])
        submit = SubmitField('제출')
    ```

  - **뷰에서 폼 사용하기**:

    ```python
    @app.route('/submit', methods=['GET', 'POST'])
    def submit():
        form = NameForm()
        if form.validate_on_submit():
            username = form.username.data
            return f'안녕하세요, {username}님!'
        return render_template('form.html', form=form)
    ```

### 1.2 세션 관리

#### 1.2.1 세션이란?

- **정의**: 여러 요청에 걸쳐 사용자 정보를 저장하는 방법
- **사용 사례**:
  - 사용자 인증
  - 장바구니
  - 사용자 설정

#### 1.2.2 Flask에서 세션 사용하기

- **`session` 임포트**:

  ```python
  from flask import session
  ```

- **세션 변수 설정**:

  ```python
  @app.route('/login', methods=['POST'])
  def login():
      session['username'] = request.form['username']
      return redirect(url_for('dashboard'))
  ```

- **세션 데이터 접근**:

  ```python
  @app.route('/dashboard')
  def dashboard():
      if 'username' in session:
          username = session['username']
          return f'다시 오신 것을 환영합니다, {username}님!'
      else:
          return redirect(url_for('login'))
  ```

- **세션 데이터 삭제**:

  ```python
  @app.route('/logout')
  def logout():
      session.pop('username', None)
      return redirect(url_for('home'))
  ```

- **비밀 키 설정**:

  - 세션 보안을 위해 필요

    ```python
    app.config['SECRET_KEY'] = '여기에_비밀_키를_입력하세요'
    ```

### 1.3 정적 파일 제공

#### 1.3.1 정적 파일 제공하기

- **정적 파일**:
  - CSS 스타일시트
  - JavaScript 파일
  - 이미지
- **기본 정적 폴더**: `static/`
- **템플릿에서 정적 파일 접근**:

  ```html
  <link
    rel="stylesheet"
    type="text/css"
    href="{{ url_for('static', filename='style.css') }}"
  />
  ```

[나머지 내용은 공간 제약으로 인해 생략되었습니다. 전체 번역본을 보시겠습니까?]

### 1.4 파일 업로드 처리

#### 1.4.1 파일 업로드 활성화하기

- **파일 업로드용 HTML 폼**:

  ```html
  <form action="/upload" method="post" enctype="multipart/form-data">
    <input type="file" name="file" />
    <input type="submit" value="업로드" />
  </form>
  ```

- **Flask에서 업로드 처리하기**:

  ```python
  from werkzeug.utils import secure_filename

  @app.route('/upload', methods=['POST'])
  def upload():
      if 'file' not in request.files:
          return '파일이 없습니다'
      file = request.files['file']
      if file.filename == '':
          return '선택된 파일이 없습니다'
      if file:
          filename = secure_filename(file.filename)
          file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
          return '파일이 성공적으로 업로드되었습니다'
  ```

### 1.5 오류 처리와 디버깅

#### 1.5.1 오류 처리

- **사용자 정의 오류 페이지**:

  ```python
  @app.errorhandler(404)
  def page_not_found(e):
      return render_template('404.html'), 404
  ```

#### 1.5.2 디버그 모드

- **디버그 모드 활성화**:

  ```python
  if __name__ == '__main__':
      app.run(debug=True)
  ```

- **장점**:
  - 자동 재로딩
  - 상세한 오류 페이지

---

## 2. Streamlit 입문

### 2.1 Streamlit 프레임워크 개요

- **Streamlit이란?**
  - 데이터 과학과 머신 러닝을 위한 웹 앱을 만드는 오픈소스 파이썬 라이브러리
- **특징**:
  - 빠른 설정과 실행
  - 대화형 위젯
  - 실시간 업데이트 지원

### 2.2 Streamlit 환경 설정

#### 2.2.1 설치

- **pip를 통한 설치**:

  ```bash
  pip install streamlit
  ```

#### 2.2.2 Streamlit 앱 실행

- **파이썬 스크립트 생성 (`app.py`)**:

  ```python
  import streamlit as st

  st.title('안녕하세요, Streamlit!')
  ```

- **앱 실행**:

  ```bash
  streamlit run app.py
  ```

### 2.3 Streamlit 기본 구성 요소

#### 2.3.1 텍스트 표시

- **헤더와 서브헤더**:

  ```python
  st.header('이것은 헤더입니다')
  st.subheader('이것은 서브헤더입니다')
  ```

- **마크다운**:

  ```python
  st.markdown('**굵은 텍스트**')
  ```

#### 2.3.2 데이터 표시

- **데이터프레임과 테이블**:

  ```python
  import pandas as pd

  df = pd.DataFrame({
      '항목 A': [1, 2, 3],
      '항목 B': [4, 5, 6]
  })

  st.dataframe(df)
  st.table(df)
  ```

#### 2.3.3 차트와 그래프

- **선 그래프**:

  ```python
  st.line_chart(df)
  ```

- **막대 그래프**:

  ```python
  st.bar_chart(df)
  ```

### 2.4 Streamlit을 활용한 데이터 시각화

- **Matplotlib과 Seaborn 사용**:

  ```python
  import matplotlib.pyplot as plt
  import seaborn as sns

  fig, ax = plt.subplots()
  sns.barplot(x='항목 A', y='항목 B', data=df, ax=ax)
  st.pyplot(fig)
  ```

- **Plotly 차트**:

  ```python
  import plotly.express as px

  fig = px.line(df, x='항목 A', y='항목 B')
  st.plotly_chart(fig)
  ```

---

## 3. 대화형 요소 구축

### 3.1 입력 위젯 생성

#### 3.1.1 일반적인 위젯들

- **텍스트 입력**:

  ```python
  name = st.text_input('이름을 입력하세요')
  ```

- **슬라이더**:

  ```python
  age = st.slider('나이를 선택하세요', 0, 100, 25)
  ```

- **체크박스**:

  ```python
  agree = st.checkbox('동의합니다')
  ```

- **버튼**:

  ```python
  if st.button('제출'):
      st.write('버튼이 클릭되었습니다!')
  ```

### 3.2 동적 콘텐츠 표시

- **조건부 표시**:

  ```python
  if agree:
      st.write(f'감사합니다, {name}님')
  ```

- **실시간 업데이트**:
  - 위젯은 자동으로 변수를 업데이트하고 재실행을 트리거함

### 3.3 Streamlit의 상태 관리

#### 3.3.1 세션 상태 이해하기

- **세션 상태란?**
  - 상호작용 간에 변수를 저장할 수 있게 해줌
- **세션 상태 사용**:

  ```python
  if 'count' not in st.session_state:
      st.session_state.count = 0

  increment = st.button('증가')
  if increment:
      st.session_state.count += 1

  st.write(f"카운트 = {st.session_state.count}")
  ```

### 3.4 캐싱과 성능 최적화

#### 3.4.1 `@st.cache_data` 데코레이터 사용

- **목적**:
  - 비용이 많이 드는 계산을 캐시하여 성능 향상
- **사용법**:

  ```python
  @st.cache_data
  def load_data():
      # 시간이 많이 걸리는 프로세스 시뮬레이션
      time.sleep(3)
      return pd.DataFrame({'A': range(100)})

  df = load_data()
  st.dataframe(df)
  ```

#### 3.4.2 캐시 동작 제어

- **매개변수**:
  - `max_entries`: 최대 캐시 항목 수
  - `ttl`: 캐시 유지 시간(초)

---

## 실습

### 4.1 파일 업로드 기능이 있는 Flask 폼 구축

#### 4.1.1 HTML 폼

```html
<form action="/upload" method="post" enctype="multipart/form-data">
  <label for="name">이름:</label>
  <input type="text" name="name" id="name" />
  <label for="file">파일 선택:</label>
  <input type="file" name="file" id="file" />
  <input type="submit" value="업로드" />
</form>
```

#### 4.1.2 Flask 뷰 함수

```python
@app.route('/upload', methods=['POST'])
def upload():
    name = request.form['name']
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return f'{name}님이 파일을 업로드했습니다'
    else:
        return '유효하지 않은 파일 형식입니다'
```

### 4.2 Streamlit을 사용한 간단한 대시보드 만들기

#### 4.2.1 대시보드 레이아웃

```python
st.title('간단한 대시보드')
st.sidebar.header('사용자 입력')
option = st.sidebar.selectbox('값을 선택하세요', ['옵션 1', '옵션 2', '옵션 3'])
st.write(f'선택하신 값: {option}')
```

### 4.3 대화형 데이터 시각화 구현

#### 4.3.1 데이터 로드 및 표시

```python
@st.cache_data
def load_data():
    return pd.read_csv('data.csv')

df = load_data()
st.line_chart(df['value'])
```

#### 4.3.2 대화형 필터

```python
value_filter = st.slider('값 필터링', min_value=int(df['value'].min()), max_value=int(df['value'].max()), value=int(df['value'].mean()))
filtered_df = df[df['value'] >= value_filter]
st.bar_chart(filtered_df['value'])
```

### 4.4 사용자 입력과 상태 관리 처리

#### 4.4.1 사용자 입력 수집

```python
user_input = st.text_input('메시지를 입력하세요')
```

#### 4.4.2 입력 표시

```python
if user_input:
    st.write(f'입력하신 내용: {user_input}')
```

#### 4.4.3 세션 상태 사용

```python
if 'messages' not in st.session_state:
    st.session_state.messages = []

if st.button('메시지 추가'):
    st.session_state.messages.append(user_input)

st.write('모든 메시지:', st.session_state.messages)
```

---

## 결론

이번 수업에서는 폼 처리, 세션 관리, 정적 파일 제공, 파일 업로드 처리 등 Flask의 고급 기능을 학습했습니다. 이러한 기술들은 견고하고 사용자 상호작용이 가능한 웹 애플리케이션을 만드는 데 필수적입니다. 또한 대화형 위젯과 실시간 업데이트가 가능한 데이터 기반 앱 개발을 단순화하는 Streamlit도 소개했습니다. 이러한 도구들을 결합함으로써, 이제 복잡한 사용자 상호작용과 데이터 시각화를 처리할 수 있는 정교한 웹 애플리케이션과 대시보드를 구축할 수 있게 되었습니다.

---

## 다음 수업 예고

마지막 수업에서는 웹 애플리케이션에 대규모 언어 모델(LLM) API를 통합하는 데 중점을 둘 것입니다. 이러한 API에 안전하게 연결하고, 사용자 입력을 처리하며, 응답을 표시하는 방법을 살펴볼 예정입니다. 또한 애플리케이션을 프로덕션 환경에 안전하게 배포하기 위한 모범 사례도 다룰 것입니다.

---

**추천 자료와 리소스**:

- [Flask 공식 문서](https://flask.palletsprojects.com/)
- [Flask-WTF 문서](https://flask-wtf.readthedocs.io/)
- [Streamlit 공식 문서](https://docs.streamlit.io/)
- [WTForms 문서](https://wtforms.readthedocs.io/)
- [Python `os` 모듈 문서](https://docs.python.org/3/library/os.html)

---

**연습문제**:

1. **Flask 연습**:

   - 유효성 검사가 포함된 등록 폼이 있는 Flask 애플리케이션을 만드세요.
   - 제출된 데이터를 세션에 저장하고 프로필 페이지에 표시하세요.
   - 일반적인 HTTP 오류에 대한 오류 처리를 구현하세요.

2. **Streamlit 연습**:
   - 데이터셋(예: CSV 파일)을 불러오는 Streamlit 앱을 만드세요.
   - 대화형 위젯을 사용하여 사용자가 플롯할 변수를 선택할 수 있게 하세요.
   - 세션 상태를 사용하여 상호작용 간에 사용자 선택을 추적하세요.
