import os
import time
import json
import logging
import requests

from dotenv import load_dotenv


# ---------------------------------------------------------
# 1. Logging Configuration
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


# ---------------------------------------------------------
# 2. Load Environment Variables
# ---------------------------------------------------------

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY is not configured.")


# ---------------------------------------------------------
# 3. OpenRouter Configuration
# ---------------------------------------------------------

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

MODEL = "openai/gpt-4o-mini"


# ---------------------------------------------------------
# 4. Instrumented LLM Execution
# ---------------------------------------------------------

def execute_instrumented_llm_call(prompt_text):

    start_time = time.time()

    logging.info("Initiating LLM request execution...")

    try:

        # -------------------------------------------------
        # OpenRouter Request
        # -------------------------------------------------

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt_text
                }
            ],
            "temperature": 0,
            "max_tokens": 400
        }

        response = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=60
        )

        response.raise_for_status()

        result = response.json()

        # -------------------------------------------------
        # Calculate Execution Time
        # -------------------------------------------------

        latency = round(time.time() - start_time, 3)

        # -------------------------------------------------
        # Extract LLM Output
        # -------------------------------------------------

        output_text = result["choices"][0]["message"]["content"]

        # -------------------------------------------------
        # Extract Token Usage
        # -------------------------------------------------

        usage = result.get("usage", {})

        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        total_tokens = usage.get(
            "total_tokens",
            input_tokens + output_tokens
        )

        # -------------------------------------------------
        # Telemetry Data
        # -------------------------------------------------

        telemetry = {
            "status": "SUCCESS",
            "model": MODEL,
            "latency_seconds": latency,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens
        }

        # -------------------------------------------------
        # Audit Log
        # -------------------------------------------------

        logging.info(
            f"Execution Successful | "
            f"Latency: {latency}s | "
            f"Input Tokens: {input_tokens} | "
            f"Output Tokens: {output_tokens} | "
            f"Total Tokens: {total_tokens}"
        )

        return output_text, telemetry

    except Exception as e:

        # -------------------------------------------------
        # Failure Telemetry
        # -------------------------------------------------

        latency = round(time.time() - start_time, 3)

        logging.error(
            f"Execution Failure after {latency}s: {e}"
        )

        telemetry = {
            "status": "FAILED",
            "model": MODEL,
            "latency_seconds": latency,
            "error": str(e)
        }

        return None, telemetry


# ---------------------------------------------------------
# 5. Main Execution
# ---------------------------------------------------------

if __name__ == "__main__":

    prompt = """
    Extract details for Pacchikolla Ravi Kiran in JSON format.
    Return the result in a structured JSON response.
    """

    output, metrics = execute_instrumented_llm_call(prompt)

    print("\n--- LLM Output ---")
    print(output)

    print("\n--- Audit Log Telemetry Data ---")
    print(json.dumps(metrics, indent=2))