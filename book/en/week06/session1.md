# Week 6 Session 1: Introduction to LLM APIs and OpenAI API Usage

## 1. Introduction to Large Language Model APIs

Large Language Model (LLM) APIs provide developers with access to powerful natural language processing capabilities without the need to train or host complex models themselves. These APIs offer a wide range of functionalities, including:

- Text generation
- Text completion
- Question answering
- Summarization
- Sentiment analysis
- Language translation

One of the most prominent LLM APIs is provided by OpenAI, which we'll focus on in this session.

## 2. OpenAI API Overview

The OpenAI API allows developers to access various models, including GPT-3.5 and GPT-4, for different natural language processing tasks. Key features include:

- **Multiple models**: Different models optimized for various tasks and performance levels.
- **Customizable parameters**: Control over generation parameters like temperature and max tokens.
- **Fine-tuning capabilities**: Ability to customize models for specific use cases.
- **Moderation tools**: Built-in content filtering for safety and compliance.

## 3. Setting Up the OpenAI API

To use the OpenAI API, follow these steps:

- a) Sign up for an OpenAI account at https://platform.openai.com/
- b) Navigate to the API keys section and create a new secret key
- c) Store your API key securely (never share or expose it publicly)
- d) Install the OpenAI Python library:

```bash
pip install openai
```

e) Set up your API key as an environment variable:

```bash
export OPENAI_API_KEY='your-api-key-here'
```

## 4. Making Your First API Call

Let's create a simple Python script to generate text using the OpenAI API:

```python
import openai

# Initialize the OpenAI client
client = openai.OpenAI()

# Make an API call
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is artificial intelligence?"}
    ]
)

# Print the generated response
print(response.choices[0].message.content)
```

This script does the following:

1. Imports the OpenAI library
2. Initializes the OpenAI client
3. Makes an API call to the chat completions endpoint
4. Prints the generated response

## 5. Understanding the API Response

The API response is a JSON object containing various fields. Key fields include:

- `id`: A unique identifier for the response
- `object`: The type of object returned (e.g., "chat.completion")
- `created`: The Unix timestamp of when the response was generated
- `model`: The name of the model used
- `choices`: An array containing the generated responses
- `usage`: Information about token usage for the request

## 6. API Parameters

When making API calls, you can control various aspects of the generation process. Some important parameters include:

- `model`: Specifies which model to use (e.g., "gpt-4o-mini", "gpt-4o")
- `messages`: An array of message objects representing the conversation history
- `max_tokens`: The maximum number of tokens to generate
- `temperature`: Controls randomness (0 to 2, lower is more deterministic)
- `top_p`: An alternative to temperature, using nucleus sampling
- `n`: Number of completions to generate for each prompt
- `stop`: Sequences where the API will stop generating further tokens

Example with additional parameters:

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a short poem about AI."}
    ],
    max_tokens=50,
    temperature=0.7,
    n=2,
    stop=["\n\n"]
)

for choice in response.choices:
    print(choice.message.content)
    print("---")
```

This example generates two short poems about AI, each limited to 50 tokens, with a temperature of 0.7, and stops generation at double newlines.

## 7. Best Practices and Rate Limits

When using the OpenAI API, keep in mind:

- **Rate limits**: OpenAI imposes rate limits on API calls. Monitor your usage and implement appropriate error handling.
- **Costs**: API usage is billed based on the number of tokens processed. Estimate costs and set up usage limits to avoid unexpected charges.
- **Prompt engineering**: Craft effective prompts to get the best results from the model.
- **Error handling**: Implement robust error handling to manage API errors and network issues.

## 8. Tokenization Basics

Understanding tokenization is crucial when working with LLM APIs:

- Tokens are the basic units processed by the model, roughly corresponding to parts of words.
- English text typically breaks down to about 4 characters per token.
- Both input (prompt) and output (completion) consume tokens.
- Different models have different token limits (e.g., 4096 for gpt-4o-mini).

You can use the OpenAI tokenizer to count tokens:

```python
from openai import OpenAI

client = OpenAI()

def num_tokens_from_string(string: str, model_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = client.tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

text = "Hello, world! How are you doing today?"
model = "gpt-4o-mini"
token_count = num_tokens_from_string(text, model)
print(f"The text contains {token_count} tokens for model {model}")
```

This function uses the `tiktoken` library (included in the OpenAI package) to accurately count tokens for a specific model.

## Conclusion

This session introduced the basics of working with LLM APIs, focusing on the OpenAI API. We covered API setup, making basic calls, understanding responses, and important concepts like tokenization. In the next session, we'll dive deeper into advanced usage patterns and explore different sampling methods to control text generation.

## References

- [1] https://blog.allenai.org/a-guide-to-language-model-sampling-in-allennlp-3b1239274bc3?gi=95ecfb79a8b8
- [2] https://huyenchip.com/2024/01/16/sampling.html
- [3] https://leftasexercise.com/2023/05/10/mastering-large-language-models-part-vi-sampling/
- [4] https://platform.openai.com/docs/quickstart
