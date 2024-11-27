# Week 12 Session 3

## Practical Applications and Advanced Use Cases

In this final session of Week 12, we'll explore practical applications and advanced use cases for controlling and structuring LLM outputs, focusing on real-world implementations and complex scenarios.

### Learning Objectives

- Apply structured output techniques to real-world use cases
- Implement complex validation workflows
- Design robust error handling systems
- Create production-ready LLM applications

### Real-World Applications

#### 1. Structured Information Extraction

```python
class InformationExtractor:
    def __init__(self):
        self.llm = LLMController()

    def extract_entities(self, text: str) -> dict:
        prompt = f"""
        Extract key entities from the following text and format as JSON:
        {text}

        Required format:
        {{
            "people": [{"name": str, "role": str}],
            "organizations": [{"name": str, "type": str}],
            "locations": [{"name": str, "context": str}]
        }}
        """
        return self.llm.generate_controlled_output(
            prompt,
            temperature=0.3  # Lower temperature for factual extraction
        )
```

#### 2. Automated Report Generation

```python
class ReportGenerator:
    def __init__(self):
        self.llm = LLMController()
        self.template = """
# Analysis Report
## Executive Summary
{summary}

## Key Findings
{findings}

## Recommendations
{recommendations}

## Technical Details
{details}
        """

    def generate_report(self, data: dict) -> str:
        sections = {}
        for section in ['summary', 'findings', 'recommendations', 'details']:
            prompt = f"Generate {section} section for report about: {data['topic']}"
            sections[section] = self.llm.generate_controlled_output(
                prompt,
                format_type='markdown'
            )

        return self.template.format(**sections)
```

### Complex Validation Workflows

#### 1. Multi-Stage Validation Pipeline

```python
class ValidationPipeline:
    def __init__(self):
        self.stages = []

    def add_stage(self, validator, error_handler=None):
        self.stages.append({
            'validator': validator,
            'error_handler': error_handler or self.default_handler
        })

    async def process(self, content):
        result = content
        for stage in self.stages:
            try:
                if not stage['validator'](result):
                    result = await stage['error_handler'](result)
            except Exception as e:
                result = await stage['error_handler'](result, error=e)
        return result

    @staticmethod
    async def default_handler(content, error=None):
        logger.error(f"Validation failed: {error}")
        raise ValueError("Content failed validation")
```

#### 2. Context-Aware Validation

```python
class ContextualValidator:
    def __init__(self):
        self.context = {}
        self.rules = {}

    def add_context(self, key: str, value: any):
        self.context[key] = value

    def add_rule(self, name: str, rule_func):
        self.rules[name] = rule_func

    def validate(self, content: dict) -> bool:
        for rule_name, rule_func in self.rules.items():
            if not rule_func(content, self.context):
                logger.error(f"Rule {rule_name} failed")
                return False
        return True

# Example usage
validator = ContextualValidator()
validator.add_context('max_length', 1000)
validator.add_context('required_fields', ['title', 'content'])
validator.add_rule('length_check',
    lambda x, ctx: len(str(x.get('content', ''))) <= ctx['max_length'])
```

### Production-Ready Implementation

#### 1. Scalable LLM Service

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

app = FastAPI()

class LLMService:
    def __init__(self):
        self.controller = LLMController()
        self.cache = {}

    async def generate(self,
                      prompt: str,
                      params: Dict[str, Any]) -> str:
        cache_key = self._get_cache_key(prompt, params)

        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            result = await self.controller.generate_controlled_output(
                prompt=prompt,
                **params
            )
            self.cache[cache_key] = result
            return result
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to generate response"
            )

    def _get_cache_key(self, prompt: str, params: Dict[str, Any]) -> str:
        return f"{hash(prompt)}_{hash(str(params))}"

service = LLMService()

@app.post("/generate")
async def generate_endpoint(request: dict):
    return await service.generate(
        request['prompt'],
        request.get('params', {})
    )
```

#### 2. Monitoring and Analytics

```python
class LLMMonitor:
    def __init__(self):
        self.metrics = defaultdict(Counter)
        self.latencies = defaultdict(list)

    async def track_request(self,
                          request_type: str,
                          start_time: float):
        duration = time.time() - start_time
        self.latencies[request_type].append(duration)
        self.metrics[request_type]['total_requests'] += 1

    def get_statistics(self) -> dict:
        stats = {}
        for req_type in self.metrics:
            stats[req_type] = {
                'total_requests': self.metrics[req_type]['total_requests'],
                'avg_latency': statistics.mean(self.latencies[req_type]),
                'p95_latency': statistics.quantiles(
                    self.latencies[req_type],
                    n=20
                )[18]  # 95th percentile
            }
        return stats
```

### Best Practices for Production Deployment

1. **Performance Optimization**

   - Implement caching strategies
   - Use async/await for I/O operations
   - Batch similar requests when possible

2. **Error Handling**

   - Implement circuit breakers
   - Use fallback responses
   - Log errors with context

3. **Monitoring**

   - Track response times
   - Monitor error rates
   - Set up alerts for anomalies

4. **Security**
   - Validate all inputs
   - Rate limit requests
   - Implement authentication

### Practical Exercise

Implement a complete LLM service with the following features:

1. Structured output generation
2. Multi-stage validation
3. Error handling and recovery
4. Monitoring and analytics
5. Caching mechanism

### Next Steps

As we conclude Week 12, you should now have a comprehensive understanding of:

- Output structuring techniques
- Control mechanisms
- Validation strategies
- Production deployment considerations

Apply these concepts in your final projects, focusing on creating robust and production-ready LLM applications.

### References

1. "Building Production-Ready LLM Applications" - AWS Machine Learning Blog
2. "Scaling Language Models in Production" - Microsoft Research
3. "Best Practices for LLM API Design" - OpenAI Engineering
4. "Monitoring ML Systems" - Google SRE Handbook
