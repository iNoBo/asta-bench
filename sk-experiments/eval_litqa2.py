# Using InspectAI's multiple-choice format (recommended)
import json
import os
import re
import string
from typing import Any

import inspect_ai
import requests
from dotenv import load_dotenv
from inspect_ai.solver import Generate, TaskState, solver

from astabench.evals.labbench.litqa2 import litqa2

load_dotenv("sk-experiments/.assistant_envs.env")
ASSISTANT_URL = os.getenv("ASSISTANT_URL")
MAX_ASSISTANT_ATTEMPTS = 3
INSUFFICIENT_INFORMATION_ANSWER = "Insufficient information to answer the question"

def assistant_search(
    query: str
):
    headers = {
        "Content-Type": "application/json"
    }
    base_payload = {
        "message": query
    }
    
    endpoint = f"{ASSISTANT_URL}/chat"
    payload = {
        **base_payload
    }

    for attempt in range(1, MAX_ASSISTANT_ATTEMPTS + 1):
        try:
            response = requests.post(endpoint, json=payload, headers=headers)
            response.raise_for_status()
            response_data = response.json()
            if not isinstance(response_data, dict):
                raise ValueError("Assistant response body is not a JSON object")  # noqa: TRY004

            reply = response_data.get("reply", "")
            if isinstance(reply, str):
                fenced_json = re.fullmatch(
                    r"```json\s*(.*?)\s*```",
                    reply.strip(),
                    flags=re.IGNORECASE | re.DOTALL,
                )
                if fenced_json:
                    json_text = fenced_json.group(1).strip()
                    json.loads(json_text)
                    return json_text

            return reply
        except (requests.RequestException, ValueError) as error:
            print(
                f"Assistant request attempt {attempt}/"
                f"{MAX_ASSISTANT_ATTEMPTS} failed: {error}"
            )

    return ""


def extract_answer_letter(response: Any) -> str:
    """Extract a multiple-choice letter from JSON or a plain-text answer."""
    if isinstance(response, dict):
        candidate = response.get("answer")
    elif isinstance(response, str):
        text = response.strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None

        if isinstance(parsed, dict):
            candidate = parsed.get("answer")
        else:
            # Accept "E", "E. 14 days", "E) 14 days", "Answer: E",
            # and "Answer is E. 14 days".
            match = re.search(
                r'\{\s*"answer"\s*:\s*"([A-Za-z])"\s*\}',
                text,
                flags=re.IGNORECASE,
            ) or re.search(
                r'\\boxed\s*\{\s*([A-Za-z])\s*\}',
                text,
                flags=re.IGNORECASE,
            ) or re.search(
                r"^\s*(?:answer\s*(?:is|:)\s*)?([A-Za-z])"
                r"(?=\s*(?:[.)]|$))",
                text,
                flags=re.IGNORECASE | re.MULTILINE,
            )
            candidate = match.group(1) if match else None
    else:
        candidate = None

    if not isinstance(candidate, str):
        return None

    letter = candidate.strip().upper()
    if len(letter) != 1 or letter not in string.ascii_uppercase:
        raise ValueError(f"Invalid answer letter: {letter!r}")

    return letter

def add_insufficient_information_choice(state: TaskState):
    """Add an "Insufficient information" choice if it doesn't exist."""
    answer_idx = next(
        (
            idx
            for idx, choice in enumerate(state.choices)
            if choice.value.strip() == INSUFFICIENT_INFORMATION_ANSWER
        ),
        None,
    )
    if answer_idx is None:
        raise ValueError(
            "Could not find the insufficient-information choice in "
            "state.choices"
        )
    letter = string.ascii_uppercase[answer_idx]
    return letter, answer_idx
    
    
@solver
def litqa2_solver():
    async def solve(state: TaskState, generate: Generate):
        # query the assistant for the answer
        # query = f"{state.input_text}\nSelect one from the choices: {state.choices}"
        assistant_response = assistant_search(state.input_text)
        if assistant_response:
            letter = extract_answer_letter(assistant_response)
            valid_letters = string.ascii_uppercase[:len(state.choices)]
            if letter not in valid_letters:
                letter, answer_idx = add_insufficient_information_choice(state)
            else:
                answer_idx = string.ascii_uppercase.index(letter)
        else:
            letter, answer_idx = add_insufficient_information_choice(state)

        answer_text = state.choices[answer_idx].value

        print(f"Response: {assistant_response!r}")
        print(f"Letter: {letter}")
        print(f"Index: {answer_idx}")
        print(f"Answer text: {answer_text}")

        # Mark choices with a single choice as correct
        for idx in range(len(state.choices)):
            state.choices.mark_choice(idx, idx == answer_idx)
        return state
    return solve

inspect_ai.eval(
    litqa2(
        with_search_tools=False,
        with_native_search_tools=False,
        split="test"
    ),
    # Model is not used in the example solver above
    model="mockllm/model",
    solver=litqa2_solver()
)
