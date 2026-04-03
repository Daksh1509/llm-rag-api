import time
from groq import Groq
from app.utils.config import get_settings
from app.utils.logger import setup_logger

settings = get_settings()
logger = setup_logger("llm_service")

SYSTEM_PROMPT = """You are a precise, helpful AI assistant.
Answer questions ONLY using the context provided below.
If the answer is not in the context, say 'I don't have enough information to answer that.'
Be concise. Use bullet points for lists. Do not hallucinate facts."""


class LLMService:
    """
    LLM Service using Groq API — free LLaMA3 inference.
    Free tier: 14,400 requests/day, 30 requests/minute.
    """

    def __init__(self):
        self.client = None

    def initialize(self) -> None:
        api_key = settings.groq_api_key
        if not api_key:
            logger.warning("GROQ_API_KEY not set in .env")
            return
        self.client = Groq(api_key=api_key)
        logger.info("LLMService ready. Model: llama-3.1-8b-instant (Groq free)")

    def build_prompt(self, query: str, context_chunks: list) -> str:
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            context_parts.append(
                f"[Source {i} — {chunk['source']} (relevance: {chunk['score']:.2f})]:\n{chunk['text']}"
            )
        context_text = "\n\n---\n\n".join(context_parts)
        return (
            f"CONTEXT:\n{context_text}\n\n"
            f"---\n"
            f"QUESTION: {query}\n\n"
            f"Answer based strictly on the context above:"
        )

    def generate(self, query: str, context_chunks: list, retries: int = 3) -> dict:
        if not self.client:
            raise RuntimeError("LLMService not initialized — check GROQ_API_KEY")

        prompt = self.build_prompt(query, context_chunks)

        for attempt in range(1, retries + 1):
            try:
                logger.info(f"Groq API call attempt {attempt}/{retries}")
                response = self.client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=1024,
                    temperature=0.2,
                )
                answer = response.choices[0].message.content.strip()
                logger.info(f"Groq responded successfully ({len(answer)} chars)")
                return {
                    "answer": answer,
                    "model_used": "llama-3.1-8b-instant (Groq free)",
                }

            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "rate" in error_str.lower():
                    wait = 2 ** attempt
                    logger.warning(f"Rate limit hit. Waiting {wait}s...")
                    time.sleep(wait)
                else:
                    logger.error(f"Groq API error: {e}")
                    raise

        raise RuntimeError("Groq LLM failed after maximum retries")


llm_service = LLMService()