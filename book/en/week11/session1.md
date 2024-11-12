# Week 11 Session 1: Introduction to Web Development and Flask

## Introduction

Welcome to the first session on web development! Today, we'll explore the fundamentals of how the web works and dive into building web applications using Flask, a lightweight Python web framework. By the end of this session, you'll have a solid understanding of client-server architecture, the roles of front-end and back-end development, and you'll have built your first Flask application with basic routing and templating.

---

## 1. Web Development Fundamentals

### 1.1 Client-Server Architecture

- **Definition**: A model where multiple clients (users) request and receive services from a centralized server.
- **How It Works**:
  - **Client**: The user's web browser or application that sends requests.
  - **Server**: A remote machine that processes requests and sends back responses.
- **Example**:
  - When you type a URL into your browser, your computer (client) sends a request to a server, which then sends back the web page.

### 1.2 HTTP/HTTPS Protocols Overview

- **HTTP (HyperText Transfer Protocol)**:
  - The foundation of data communication on the web.
  - Uses methods like GET, POST, PUT, DELETE to interact with resources.
- **HTTPS (HTTP Secure)**:
  - An extension of HTTP.
  - Uses SSL/TLS encryption for secure communication.
- **Importance**:
  - Defines how messages are formatted and transmitted.
  - Determines how web servers and browsers should respond to various commands.

### 1.3 Front-End vs. Back-End Development Roles

- **Front-End Development**:
  - Deals with the user interface and user experience.
  - Technologies: HTML, CSS, JavaScript.
  - Responsibilities: Designing the layout, visuals, and interactive elements.
- **Back-End Development**:
  - Focuses on server-side logic and integration.
  - Technologies: Python, Java, Ruby, databases.
  - Responsibilities: Managing application logic, databases, user authentication.

### 1.4 RESTful API Concepts

- **REST (Representational State Transfer)**:
  - An architectural style for designing networked applications.
- **Principles**:
  - **Statelessness**: Each request from a client contains all the information needed.
  - **Uniform Interface**: Resources are identified in requests (e.g., via URIs).
  - **Resource Manipulation via Representations**: Clients can modify resources using representations (e.g., JSON, XML).
- **HTTP Methods in REST**:
  - **GET**: Retrieve data.
  - **POST**: Create new data.
  - **PUT**: Update existing data.
  - **DELETE**: Remove data.

---

## 2. Introduction to Flask

### 2.1 Flask Framework Overview and Installation

- **What is Flask?**
  - A micro web framework written in Python.
  - Known for its simplicity and ease of use.
- **Features**:
  - Lightweight and modular.
  - Extensible with a wide range of extensions.
- **Installation**:

  ```bash
  pip install flask
  ```

### 2.2 Creating Your First Flask Application

- **Basic Application Structure**:

  ```python
  from flask import Flask

  app = Flask(__name__)

  @app.route('/')
  def home():
      return 'Hello, World!'

  if __name__ == '__main__':
      app.run(debug=True)
  ```

- **Explanation**:
  - Import the Flask class.
  - Create an instance of the Flask application.
  - Define a route for the URL `/`.
  - Run the application.

### 2.3 Understanding Routes and Decorators

- **Routes**:
  - URLs that users can access.
  - Defined using the `@app.route()` decorator.
- **Decorators**:

  - Modify the behavior of functions.
  - In Flask, used to bind URLs to functions.

- **Example**:

  ```python
  @app.route('/about')
  def about():
      return 'This is the about page.'
  ```

### 2.4 Handling Different HTTP Methods (GET, POST)

- **Specifying Methods**:

  ```python
  @app.route('/submit', methods=['GET', 'POST'])
  def submit():
      if request.method == 'POST':
          # Handle POST request
          pass
      else:
          # Handle GET request
          pass
  ```

- **Accessing Request Data**:

  ```python
  from flask import request

  data = request.form['key']  # For form data
  data = request.args.get('key')  # For query parameters
  ```

---

## 3. Basic Template Rendering

### 3.1 Introduction to Jinja2 Templating

- **What is Jinja2?**
  - A templating engine for Python.
  - Allows embedding Python-like expressions in HTML.
- **Features**:
  - Template inheritance.
  - Control structures (loops, conditionals).
  - Filters and macros.

### 3.2 Creating and Rendering HTML Templates

- **Folder Structure**:

  - Templates are stored in a `templates` directory.

- **Example Template (`templates/home.html`)**:

  ```html
  <!DOCTYPE html>
  <html>
    <head>
      <title>{{ title }}</title>
    </head>
    <body>
      <h1>Welcome to {{ title }}</h1>
    </body>
  </html>
  ```

- **Rendering Templates**:

  ```python
  from flask import render_template

  @app.route('/')
  def home():
      return render_template('home.html', title='My Flask App')
  ```

### 3.3 Template Inheritance and Reuse

- **Base Template (`templates/base.html`)**:

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

- **Child Template (`templates/home.html`)**:

  ```html
  {% extends 'base.html' %} {% block title %}Home Page{% endblock %} {% block
  content %}
  <h1>Welcome to the Home Page</h1>
  {% endblock %}
  ```

### 3.4 Passing Variables to Templates

- **Passing Data**:

  ```python
  @app.route('/user/<username>')
  def profile(username):
      return render_template('profile.html', username=username)
  ```

- **Using Variables in Templates**:

  ```html
  <h1>User Profile: {{ username }}</h1>
  ```

---

## Practical Component

### 4.1 Setting Up Flask Development Environment

- **Virtual Environment Setup**:

  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows use venv\Scripts\activate
  ```

- **Install Flask**:

  ```bash
  pip install flask
  ```

### 4.2 Creating a Simple "Hello, World!" Application

- **Application (`app.py`)**:

  ```python
  from flask import Flask

  app = Flask(__name__)

  @app.route('/')
  def hello():
      return 'Hello, World!'

  if __name__ == '__main__':
      app.run(debug=True)
  ```

- **Run the Application**:

  ```bash
  python app.py
  ```

- **Access in Browser**:
  - Navigate to `http://localhost:5000/`

### 4.3 Implementing Multiple Routes with Different Endpoints

- **Adding Routes**:

  ```python
  @app.route('/about')
  def about():
      return 'About Page'

  @app.route('/contact')
  def contact():
      return 'Contact Page'
  ```

### 4.4 Building a Basic Template-Based Web Page

- **Create Template (`templates/index.html`)**:

  ```html
  <!DOCTYPE html>
  <html>
    <head>
      <title>Flask Template</title>
    </head>
    <body>
      <h1>{{ message }}</h1>
    </body>
  </html>
  ```

- **Modify Route to Render Template**:

  ```python
  @app.route('/')
  def home():
      return render_template('index.html', message='Hello from Flask and Jinja2!')
  ```

---

## Conclusion

In this session, we've covered the basics of web development, focusing on the client-server model and the HTTP protocol. We've differentiated between front-end and back-end development and introduced RESTful APIs. Moving into practical application, we've set up a Flask development environment, created a simple web server, and learned how to handle routing and templating with Jinja2.

---

## Looking Ahead

In the next session, we'll explore more advanced features of Flask, such as handling forms, database integration, and user authentication. We'll also introduce Streamlit, a powerful framework for rapidly developing AI and data science applications with minimal code.

---

**Recommended Reading and Resources**:

- [Flask Official Documentation](https://flask.palletsprojects.com/)
- [Jinja2 Template Designer Documentation](https://jinja.palletsprojects.com/)
- [HTTP Protocol Overview](https://developer.mozilla.org/en-US/docs/Web/HTTP/Overview)
- [RESTful API Design](https://restfulapi.net/)

---

**Exercise**:

1. Extend your Flask application to include a new route `/user/<name>` that greets the user by name.
2. Create a base template and use template inheritance to create multiple pages with a consistent layout.
