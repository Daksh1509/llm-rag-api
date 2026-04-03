from openai import OpenAI

client = OpenAI(api_key="your-api-key-here")

try:
    r = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "say hello"}],
        max_tokens=10
    )
    print("SUCCESS:", r.choices[0].message.content)
except Exception as e:
    print("ERROR:", type(e).__name__, str(e))