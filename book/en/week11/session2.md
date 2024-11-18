# Week 11 Session 2: Advanced Flask and Introduction to Streamlit

## Introduction

Welcome back! In today's session, we'll delve deeper into Flask by exploring advanced features such as form handling, session management, and file uploads. We'll also introduce Streamlit, a powerful framework for building interactive, data-driven web applications with minimal effort. By the end of this session, you'll be equipped to create more dynamic web applications and interactive dashboards.

---

## 1. Advanced Flask Features

### 1.1 Form Handling and Validation

#### 1.1.1 Handling Forms in Flask

- **Forms in Web Applications**:
  - Used to collect user input.
  - Commonly involve `GET` and `POST` methods.
- **Setting Up a Form**:

  - Create an HTML form using `<form>` tags.
  - Specify the `action` and `method` attributes.

  ```html
  <form action="/submit" method="post">
    <input type="text" name="username" placeholder="Enter your name" />
    <input type="submit" value="Submit" />
  </form>
  ```

#### 1.1.2 Accessing Form Data

- **Using the `request` Object**:

  ```python
  from flask import request

  @app.route('/submit', methods=['POST'])
  def submit():
      username = request.form['username']
      return f'Hello, {username}!'
  ```

- **Handling GET Requests**:

  ```python
  @app.route('/search')
  def search():
      query = request.args.get('q')
      return f'Search results for: {query}'
  ```

#### 1.1.3 Form Validation

- **Why Validate Forms?**
  - Ensure data integrity.
  - Prevent security vulnerabilities (e.g., SQL injection).
- **Manual Validation**:

  ```python
  if not username:
      error = 'Username is required.'
      return render_template('form.html', error=error)
  ```

- **Using Flask-WTF Extension**:

  - **Installation**:

    ```bash
    pip install flask-wtf
    ```

  - **Creating a Form Class**:

    ```python
    from flask_wtf import FlaskForm
    from wtforms import StringField, SubmitField
    from wtforms.validators import DataRequired

    class NameForm(FlaskForm):
        username = StringField('Username', validators=[DataRequired()])
        submit = SubmitField('Submit')
    ```

  - **Using the Form in a View**:

    ```python
    @app.route('/submit', methods=['GET', 'POST'])
    def submit():
        form = NameForm()
        if form.validate_on_submit():
            username = form.username.data
            return f'Hello, {username}!'
        return render_template('form.html', form=form)
    ```

### 1.2 Session Management

#### 1.2.1 What are Sessions?

- **Definition**: A way to store information about a user across multiple requests.
- **Use Cases**:
  - User authentication.
  - Shopping carts.
  - Preferences.

#### 1.2.2 Using Sessions in Flask

- **Importing `session`**:

  ```python
  from flask import session
  ```

- **Setting a Session Variable**:

  ```python
  @app.route('/login', methods=['POST'])
  def login():
      session['username'] = request.form['username']
      return redirect(url_for('dashboard'))
  ```

- **Accessing Session Data**:

  ```python
  @app.route('/dashboard')
  def dashboard():
      if 'username' in session:
          username = session['username']
          return f'Welcome back, {username}!'
      else:
          return redirect(url_for('login'))
  ```

- **Clearing Session Data**:

  ```python
  @app.route('/logout')
  def logout():
      session.pop('username', None)
      return redirect(url_for('home'))
  ```

- **Configuring Secret Key**:

  - Necessary for session security.

    ```python
    app.config['SECRET_KEY'] = 'your_secret_key_here'
    ```

### 1.3 Static File Serving

#### 1.3.1 Serving Static Files

- **Static Files**:
  - CSS stylesheets.
  - JavaScript files.
  - Images.
- **Default Static Folder**: `static/`
- **Accessing Static Files in Templates**:

  ```html
  <link
    rel="stylesheet"
    type="text/css"
    href="{{ url_for('static', filename='style.css') }}"
  />
  ```

#### 1.3.2 Customizing Static Folder

- **Changing the Static Folder**:

  ```python
  app = Flask(__name__, static_folder='assets')
  ```

### 1.4 File Upload Handling

#### 1.4.1 Enabling File Uploads

- **HTML Form for File Upload**:

  ```html
  <form action="/upload" method="post" enctype="multipart/form-data">
    <input type="file" name="file" />
    <input type="submit" value="Upload" />
  </form>
  ```

- **Handling Upload in Flask**:

  ```python
  from werkzeug.utils import secure_filename

  @app.route('/upload', methods=['POST'])
  def upload():
      if 'file' not in request.files:
          return 'No file part'
      file = request.files['file']
      if file.filename == '':
          return 'No selected file'
      if file:
          filename = secure_filename(file.filename)
          file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
          return 'File uploaded successfully'
  ```

- **Configuring Upload Folder**:

  ```python
  app.config['UPLOAD_FOLDER'] = '/path/to/upload'
  ```

#### 1.4.2 Validating File Uploads

- **Allowed File Extensions**:

  ```python
  ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

  def allowed_file(filename):
      return '.' in filename and \
             filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
  ```

### 1.5 Error Handling and Debugging

#### 1.5.1 Handling Errors

- **Custom Error Pages**:

  ```python
  @app.errorhandler(404)
  def page_not_found(e):
      return render_template('404.html'), 404
  ```

#### 1.5.2 Debug Mode

- **Enabling Debug Mode**:

  ```python
  if __name__ == '__main__':
      app.run(debug=True)
  ```

- **Benefits**:
  - Automatic reloading.
  - Detailed error pages.

---

## 2. Introduction to Streamlit

### 2.1 Streamlit Framework Overview

- **What is Streamlit?**
  - An open-source Python library for creating web apps for data science and machine learning.
- **Features**:
  - Quick to set up and run.
  - Interactive widgets.
  - Supports real-time updates.

### 2.2 Setting Up Streamlit Environment

#### 2.2.1 Installation

- **Install via pip**:

  ```bash
  pip install streamlit
  ```

#### 2.2.2 Running a Streamlit App

- **Create a Python Script (`app.py`)**:

  ```python
  import streamlit as st

  st.title('Hello, Streamlit!')
  ```

- **Run the App**:

  ```bash
  streamlit run app.py
  ```

### 2.3 Basic Streamlit Components

#### 2.3.1 Displaying Text

- **Headers and Subheaders**:

  ```python
  st.header('This is a header')
  st.subheader('This is a subheader')
  ```

- **Markdown**:

  ```python
  st.markdown('**Bold Text**')
  ```

#### 2.3.2 Displaying Data

- **DataFrames and Tables**:

  ```python
  import pandas as pd

  df = pd.DataFrame({
      'Column A': [1, 2, 3],
      'Column B': [4, 5, 6]
  })

  st.dataframe(df)
  st.table(df)
  ```

#### 2.3.3 Charts and Graphs

- **Line Chart**:

  ```python
  st.line_chart(df)
  ```

- **Bar Chart**:

  ```python
  st.bar_chart(df)
  ```

### 2.4 Data Visualization with Streamlit

- **Matplotlib and Seaborn**:

  ```python
  import matplotlib.pyplot as plt
  import seaborn as sns

  fig, ax = plt.subplots()
  sns.barplot(x='Column A', y='Column B', data=df, ax=ax)
  st.pyplot(fig)
  ```

- **Plotly Charts**:

  ```python
  import plotly.express as px

  fig = px.line(df, x='Column A', y='Column B')
  st.plotly_chart(fig)
  ```

---

## 3. Building Interactive Elements

### 3.1 Creating Input Widgets

#### 3.1.1 Common Widgets

- **Text Input**:

  ```python
  name = st.text_input('Enter your name')
  ```

- **Slider**:

  ```python
  age = st.slider('Select your age', 0, 100, 25)
  ```

- **Checkbox**:

  ```python
  agree = st.checkbox('I agree')
  ```

- **Button**:

  ```python
  if st.button('Submit'):
      st.write('Button clicked!')
  ```

### 3.2 Displaying Dynamic Content

- **Conditional Display**:

  ```python
  if agree:
      st.write(f'Thank you, {name}')
  ```

- **Real-time Updates**:

  - Widgets automatically update variables and trigger re-runs.

### 3.3 State Management in Streamlit

#### 3.3.1 Understanding Session State

- **What is Session State?**
  - Allows storing variables across interactions.
- **Using Session State**:

  ```python
  if 'count' not in st.session_state:
      st.session_state.count = 0

  increment = st.button('Increment')
  if increment:
      st.session_state.count += 1

  st.write(f"Count = {st.session_state.count}")
  ```

### 3.4 Caching and Performance Optimization

#### 3.4.1 Using `@st.cache_data` Decorator

- **Purpose**:
  - Cache expensive computations to improve performance.
- **Usage**:

  ```python
  @st.cache_data
  def load_data():
      # Simulate a time-consuming process
      time.sleep(3)
      return pd.DataFrame({'A': range(100)})

  df = load_data()
  st.dataframe(df)
  ```

#### 3.4.2 Controlling Cache Behavior

- **Parameters**:
  - `max_entries`: Maximum number of cached entries.
  - `ttl`: Time to live in seconds.

---

## Practical Component

### 4.1 Building a Flask Form with File Upload Capability

#### 4.1.1 HTML Form

```html
<form action="/upload" method="post" enctype="multipart/form-data">
  <label for="name">Name:</label>
  <input type="text" name="name" id="name" />
  <label for="file">Choose a file:</label>
  <input type="file" name="file" id="file" />
  <input type="submit" value="Upload" />
</form>
```

#### 4.1.2 Flask View Function

```python
@app.route('/upload', methods=['POST'])
def upload():
    name = request.form['name']
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return f'File uploaded by {name}'
    else:
        return 'Invalid file type'
```

### 4.2 Creating a Simple Dashboard Using Streamlit

#### 4.2.1 Dashboard Layout

```python
st.title('Simple Dashboard')
st.sidebar.header('User Input')
option = st.sidebar.selectbox('Select a value', ['Option 1', 'Option 2', 'Option 3'])
st.write(f'You selected: {option}')
```

### 4.3 Implementing Interactive Data Visualization

#### 4.3.1 Loading and Displaying Data

```python
@st.cache_data
def load_data():
    return pd.read_csv('data.csv')

df = load_data()
st.line_chart(df['value'])
```

#### 4.3.2 Interactive Filters

```python
value_filter = st.slider('Filter values', min_value=int(df['value'].min()), max_value=int(df['value'].max()), value=int(df['value'].mean()))
filtered_df = df[df['value'] >= value_filter]
st.bar_chart(filtered_df['value'])
```

### 4.4 Handling User Input and State Management

#### 4.4.1 Collecting User Input

```python
user_input = st.text_input('Enter a message')
```

#### 4.4.2 Displaying Input

```python
if user_input:
    st.write(f'You entered: {user_input}')
```

#### 4.4.3 Using Session State

```python
if 'messages' not in st.session_state:
    st.session_state.messages = []

if st.button('Add Message'):
    st.session_state.messages.append(user_input)

st.write('All messages:', st.session_state.messages)
```

---

## Conclusion

In this session, we've expanded our knowledge of Flask by learning how to handle forms, manage user sessions, serve static files, and process file uploads. These skills are essential for creating robust, user-interactive web applications. We've also been introduced to Streamlit, a framework that simplifies the development of data-driven apps with interactive widgets and real-time updates. By combining these tools, you're now capable of building sophisticated web applications and dashboards that can handle complex user interactions and data visualizations.

---

## Looking Ahead

In our final session, we'll focus on integrating Large Language Model (LLM) APIs into web applications. We'll explore how to securely connect to these APIs, handle user inputs, and display responses. Additionally, we'll cover best practices for deploying your applications securely to a production environment.

---

**Recommended Reading and Resources**:

- [Flask Official Documentation](https://flask.palletsprojects.com/)
- [Flask-WTF Documentation](https://flask-wtf.readthedocs.io/)
- [Streamlit Official Documentation](https://docs.streamlit.io/)
- [WTForms Documentation](https://wtforms.readthedocs.io/)
- [Python's `os` Module Documentation](https://docs.python.org/3/library/os.html)

---

**Exercise**:

1. **Flask Exercise**:

   - Create a Flask application with a registration form that includes validation.
   - Store the submitted data in a session and display it on a profile page.
   - Implement error handling for common HTTP errors.

2. **Streamlit Exercise**:
   - Build a Streamlit app that loads a dataset (e.g., from a CSV file).
   - Allow users to select variables to plot using interactive widgets.
   - Use session state to keep track of user selections across interactions.
