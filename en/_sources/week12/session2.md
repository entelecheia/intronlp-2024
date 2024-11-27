# Week 12 Session 2: Advanced LLM Output Control

## Advanced Control Mechanisms for LLM Outputs

Building on our understanding of basic output structuring, this session explores advanced techniques for controlling LLM outputs, including temperature settings, sampling parameters, and sophisticated validation mechanisms.

### Learning Objectives

- Master temperature and sampling parameters for output control
- Implement advanced validation techniques
- Design robust error handling systems
- Create comprehensive output parsing solutions

### Temperature and Sampling Parameters

Temperature and other sampling parameters significantly influence LLM output characteristics.

#### 1. Temperature Control

```python
def generate_with_temperature(prompt, temperature=0.7):
    """
    Generate text with specified temperature.
    Lower temperature (0.1-0.5): More focused, deterministic outputs
    Higher temperature (0.7-1.0): More creative, diverse outputs
    """
    return llm.generate(
        prompt,
        temperature=temperature,
        max_tokens=100
    )
```

#### 2. Top-p (Nucleus) Sampling

```python
def nucleus_sampling(prompt, top_p=0.9):
    """
    Implement nucleus sampling for more natural text generation.
    top_p controls cumulative probability threshold for token selection.
    """
    return llm.generate(
        prompt,
        top_p=top_p,
        temperature=0.7
    )
```

### Advanced Validation Techniques

#### 1. Schema-Based Validation

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class AnalysisOutput(BaseModel):
    topic: str
    confidence: float = Field(ge=0.0, le=1.0)
    key_points: List[str]
    metadata: Optional[dict] = None

def validate_analysis(response: dict) -> bool:
    try:
        AnalysisOutput(**response)
        return True
    except ValidationError as e:
        logger.error(f"Validation failed: {e}")
        return False
```

#### 2. Custom Validation Rules

```python
class OutputValidator:
    def __init__(self):
        self.validators = []

    def add_rule(self, rule_func):
        self.validators.append(rule_func)

    def validate(self, output):
        results = []
        for validator in self.validators:
            results.append(validator(output))
        return all(results)

# Example usage
validator = OutputValidator()
validator.add_rule(lambda x: len(x.get('key_points', [])) >= 3)
validator.add_rule(lambda x: 0 <= x.get('confidence', 0) <= 1)
```

### Error Handling and Recovery

#### 1. Retry Mechanism

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_with_retry(prompt, format_type='json'):
    """
    Attempt generation with exponential backoff retry.
    """
    try:
        response = llm.generate(prompt)
        if validate_output(response, format_type):
            return response
        raise ValueError("Invalid output format")
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise
```

#### 2. Fallback Strategies

```python
class OutputGenerator:
    def __init__(self, primary_model, fallback_model):
        self.primary = primary_model
        self.fallback = fallback_model

    def generate_safe(self, prompt):
        try:
            return self.primary.generate(prompt)
        except Exception:
            logger.warning("Primary model failed, using fallback")
            return self.fallback.generate(prompt)
```

### Comprehensive Output Parsing

#### 1. Multi-Format Parser

```python
class OutputParser:
    def __init__(self):
        self.parsers = {
            'json': self._parse_json,
            'xml': self._parse_xml,
            'markdown': self._parse_markdown
        }

    def parse(self, content, format_type):
        parser = self.parsers.get(format_type)
        if not parser:
            raise ValueError(f"Unsupported format: {format_type}")
        return parser(content)

    def _parse_json(self, content):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return self._attempt_json_recovery(content)

    def _attempt_json_recovery(self, content):
        # Implement recovery strategies for malformed JSON
        pass
```

#### 2. Format Detection

```python
def detect_format(content: str) -> str:
    """
    Automatically detect the format of the output.
    """
    if content.strip().startswith('{'):
        return 'json'
    elif content.strip().startswith('<'):
        return 'xml'
    elif content.strip().startswith('#'):
        return 'markdown'
    return 'plain_text'
```

### Practical Implementation

Here's a complete example combining various control mechanisms:

```python
class LLMController:
    def __init__(self, model_name='gpt-3.5-turbo'):
        self.model = load_model(model_name)
        self.validator = OutputValidator()
        self.parser = OutputParser()

    def generate_controlled_output(
        self,
        prompt: str,
        format_type: str = 'json',
        temperature: float = 0.7,
        max_retries: int = 3
    ):
        for attempt in range(max_retries):
            try:
                # Generate response
                response = self.model.generate(
                    prompt,
                    temperature=temperature,
                    top_p=0.9
                )

                # Parse output
                parsed = self.parser.parse(response, format_type)

                # Validate
                if self.validator.validate(parsed):
                    return parsed

            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise

                # Adjust parameters for retry
                temperature = max(0.1, temperature - 0.2)

        raise ValueError("Failed to generate valid output")
```

### Best Practices for Production

1. **Monitoring and Logging**

   - Track success/failure rates
   - Monitor response times
   - Log validation failures for analysis

2. **Performance Optimization**

   - Cache common responses
   - Implement batch processing
   - Use async/await for concurrent requests

3. **Security Considerations**
   - Sanitize inputs and outputs
   - Implement rate limiting
   - Handle sensitive information appropriately

### Next Steps

In the next week, we'll focus on applying these concepts in real-world scenarios through the final project presentations.

### References

1. "Advanced LLM Control Mechanisms" - OpenAI API Documentation
2. "Sampling Strategies in Language Models" - DeepMind Research
3. "Production-Ready LLM Systems" - ML Engineering Best Practices
4. "Error Handling in AI Systems" - Google AI Platform Documentation
