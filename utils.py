import json
import logging
import logging.handlers
import anthropic
import openai
import re
from typing import List, Dict, Any
import sys

logger = logging.getLogger("study_design_agent")

def pretty_test_output(obj):
    print(json.dumps(obj, indent=2))

def json_array_slice(text: str) -> list:
    """Extract the first well-formed JSON array from text using bracket matching.
    Handles cases where the model returns multiple JSON blocks or trailing content.
    Returns [] if no valid array is found.
    """
    start = text.find('[')
    if start == -1:
        return []
    depth = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    return []
    return []

def json_slice(text: str) -> Dict[str, Any]:
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e != -1 and e > s:
        try:
            return json.loads(text[s:e+1])
        except Exception:
            return {}
    return {}

def taskresult_to_text(result: Any) -> str:
    """
    Extract plain text from autogen-agentchat TaskResult across versions.
    Falls back to str(result) if structure is unexpected.
    """
    # 1) Newer AgentChat: final_message with content (could be str or list of TextContent)
    fm = getattr(result, "final_message", None)
    if fm is not None:
        content = getattr(fm, "content", "")
        if isinstance(content, str):
            return content
        # content may be a list of objects with .text (TextContent)
        try:
            return "".join(getattr(piece, "text", "") for piece in content)
        except Exception:
            pass

    # 2) Sometimes there's a .messages list; take the last assistant message
    msgs = getattr(result, "messages", None)
    if msgs:
        last = msgs[-1]
        content = getattr(last, "content", "")
        if isinstance(content, str):
            return content
        try:
            return "".join(getattr(piece, "text", "") for piece in content)
        except Exception:
            pass

    # 3) Some builds surface .output_text
    output_text = getattr(result, "output_text", None)
    if isinstance(output_text, str):
        return output_text

    # 4) Fallback
    try:
        return str(result)
    except Exception:
        return ""

def complete_text_claude(prompt, stop_sequences=[anthropic.HUMAN_PROMPT], model="claude-v1", max_tokens_to_sample = 2000, temperature=0.5, log_file=None, **kwargs):
    """ Call the Claude API to complete a prompt."""

    ai_prompt = anthropic.AI_PROMPT
    if "ai_prompt" in kwargs is not None:
        ai_prompt = kwargs["ai_prompt"]
        del kwargs["ai_prompt"]
    if model.startswith("claude-3"):
        messages = [
            {'role': 'user', 'content': f"{anthropic.HUMAN_PROMPT} {prompt}"}
        ]
        rsp = anthropic_client.messages.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens_to_sample
        )
        completion = rsp.content[0].text
        if log_file is not None:
            log_to_file(log_file, prompt, completion, model, max_tokens_to_sample)
        return completion
    try:
        rsp = anthropic_client.completions.create(
            prompt=f"{anthropic.HUMAN_PROMPT} {prompt} {ai_prompt}",
            stop_sequences=stop_sequences,
            model=model,
            temperature=temperature,
            max_tokens_to_sample=max_tokens_to_sample,
            **kwargs
        )
    except anthropic.APIStatusError as e:
        print(e)
        exit()
        raise TooLongPromptError()
    except Exception as e:
        exit()
        raise LLMError(e)

    completion = rsp.completion
    if log_file is not None:
        log_to_file(log_file, prompt, completion, model, max_tokens_to_sample)
    return completion


def get_embedding_crfm(text, model="openai/gpt-4-0314"):
    request = Request(model="openai/text-similarity-ada-001", prompt=text, embedding=True)
    request_result: RequestResult = service.make_request(auth, request)
    return request_result.embedding

def complete_text_crfm(prompt=None, stop_sequences = None, model="openai/gpt-4-0314",  max_tokens_to_sample=2000, temperature = 0.5, log_file=None, messages = None, **kwargs):

    random = log_file
    if messages:
        request = Request(
                prompt=prompt,
                messages=messages,
                model=model,
                stop_sequences=stop_sequences,
                temperature = temperature,
                max_tokens = max_tokens_to_sample,
                random = random
            )
    else:
        print("model", model)
        print("max_tokens", max_tokens_to_sample)
        request = Request(
                prompt=prompt,
                model=model,
                stop_sequences=stop_sequences,
                temperature = temperature,
                max_tokens = max_tokens_to_sample,
                random = random
        )

    try:
        request_result: RequestResult = service.make_request(auth, request)
    except Exception as e:
        # probably too long prompt
        print(e)
        exit()

    if request_result.success == False:
        print(request.error)
    completion = request_result.completions[0].text
    if log_file is not None:
        log_to_file(log_file, prompt, completion, model, max_tokens_to_sample)
    return completion


def complete_text_openai(prompt, stop_sequences=[], model="gpt-3.5-turbo", max_tokens_to_sample=2000, temperature=0.5, log_file=None, **kwargs):

    """ Call the OpenAI API to complete a prompt."""
    raw_request = {
          "model": model,
          **kwargs
    }
    if model.startswith("gpt-3.5") or model.startswith("gpt-4") or model.startswith("o1"):
        # Requires openai==1.42.0
        messages = [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(**{"messages": messages,**raw_request})
        completion = response.choices[0].message.content
    else:
        response = client.completions.create(**{"prompt": prompt,**raw_request})
        completion = response.choices[0].text
    if log_file is not None:
        log_to_file(log_file, prompt, completion, model, max_tokens_to_sample)
    return completion

def complete_text(prompt, log_file, model, **kwargs):
    """ Complete text using the specified model with appropriate API. """

    if model.startswith("claude"):
        # use anthropic API
        completion = complete_text_claude(prompt, stop_sequences=[anthropic.HUMAN_PROMPT, "Observation:"], log_file=log_file, model=model, **kwargs)
    elif "/" in model:
        # use CRFM API since this specifies organization like "openai/..."
        completion = complete_text_crfm(prompt, stop_sequences=["Observation:"], log_file=log_file, model=model, **kwargs)
    else:
        # use OpenAI API
        completion = complete_text_openai(prompt, stop_sequences=["Observation:"], log_file=log_file, model=model, **kwargs)
    return completion

def split_text_into_chunks(text, chunk_size=1000):
    from camel.utils.commons import create_chunks
    if isinstance(text, list): # already been split
        blocks = text
    else:
        blocks = create_chunks(
            text,chunk_size
        )
    return blocks

def set_up_log():
    import os
    from datetime import datetime

    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Drop handlers that libraries installed with logging.basicConfig() at import time
    # (camel.logger does this). They have no level filter, so with root at DEBUG they
    # would echo every third-party DEBUG/INFO record to the console.
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    # Silence noisy third-party libraries — only WARNING+ from them
    for lib in ("camel", "openai", "httpcore", "httpx", "urllib3", "requests",
                "rquest", "primp", "duckduckgo_search"):
        logging.getLogger(lib).setLevel(logging.WARNING)

    # Console handler for root (WARNING+ only, catches third-party warnings/errors)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Dedicated INFO console handler for our own logger only
    own_logger = logging.getLogger("study_design_agent")
    own_console = logging.StreamHandler(sys.stderr)
    own_console.setLevel(logging.INFO)
    own_console.setFormatter(formatter)
    own_logger.addHandler(own_console)
    own_logger.propagate = True  # let messages reach root's file/error handlers too

    # Workflow file handler (daily rotation, keep 30 days)
    workflow_path = os.path.join(log_dir, f"workflow_{timestamp}.log")
    file_handler = logging.handlers.TimedRotatingFileHandler(
        workflow_path, when="midnight", backupCount=30, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Error-only file handler
    error_path = os.path.join(log_dir, f"errors_{timestamp}.log")
    error_handler = logging.handlers.TimedRotatingFileHandler(
        error_path, when="midnight", backupCount=30, encoding="utf-8"
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    root_logger.addHandler(error_handler)


def parse_peco_json(peco_json_str: str) -> Dict[str, Any]:
    """
    Parse PECO elements from JSON string

    Args:
        peco_json_str: JSON string containing PECO elements

    Returns:
        Dictionary with population, exposure, comparison, outcome keys
    """
    try:
        peco_elements = json.loads(peco_json_str)
        return peco_elements
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid PECO elements JSON: {e}")

def extract_peco_values(peco_elements: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract string values from structured PECO elements

    Args:
        peco_elements: Dictionary with population, exposure, comparison, outcome lists

    Returns:
        Dictionary with flattened string values for each PECO element
    """
    # Helper to safely extract terms (uses raw_term for display; falls back to legacy term field)
    def extract_terms(items: list) -> list:
        result = []
        for item in items:
            if item.get("raw_term") is not None:
                result.append(item["raw_term"])
            elif item.get("term") is not None:
                result.append(item["term"])
        return result

    pop_terms = extract_terms(peco_elements.get("population", []))
    population = ", ".join(pop_terms) if pop_terms else "any population"

    int_terms = extract_terms(peco_elements.get("exposure", []))
    exposure = ", ".join(int_terms) if int_terms else ""

    comp_terms = extract_terms(peco_elements.get("comparison", []))
    comparison = ", ".join(comp_terms) if comp_terms else "no specific comparison"

    out_terms = extract_terms(peco_elements.get("outcome", []))
    outcome = ", ".join(out_terms) if out_terms else ""

    return {
        "population": population,
        "exposure": exposure,
        "comparison": comparison,
        "outcome": outcome
    }


_DOI_PREFIX_MAP = {
    "10.1016": "elsevier",   # ScienceDirect
    "10.1016": "elsevier",
    "10.1053": "elsevier",   # e.g. Journal of Thoracic and Cardiovascular Surgery
    "10.1006": "elsevier",
    "10.1378": "elsevier",   # Chest
    "10.1016": "elsevier",
    "10.1002": "wiley",      # Wiley Online Library
    "10.1111": "wiley",
    "10.1161": "wiley",      # AHA journals via Wiley
    "10.1038": "springer",   # Nature portfolio
    "10.1007": "springer",   # Springer
    "10.1186": "springer",   # BioMed Central (OA)
    "10.1371": "plos",       # PLOS journals
    "10.1136": "bmj",        # BMJ journals
    "10.1056": "nejm",       # NEJM (no public API, will fall through)
    "10.15252": "embo",      # EMBO journals
}
 
def doi_publisher(doi):
    """Return a publisher hint from DOI prefix, or None."""
    if not doi:
        return None
    prefix = doi.split("/")[0] if "/" in doi else ""
    return _DOI_PREFIX_MAP.get(prefix)