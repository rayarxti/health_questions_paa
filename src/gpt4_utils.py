import os
import dotenv
from pydantic import BaseModel
from openai import OpenAI
import pandas as pd
import tqdm
from typing import List

# Global Constants
dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = "gpt-4.1-mini-2025-04-14"

SYSTEM_PROMPT = """
Classify the following search query into exactly one of four mutually exclusive categories. Provide the classification as a one-hot vector plus an explanation.

Categories:

1) Incorrect assumptions
The question is based on a false or ill-formed assumption, making it unanswerable as posed. Note that questions like "Why X" assumes that X is true.
Examples:
- Why is metformin banned in the USA?
- Why do doctors no longer prescribe metformin?
- Why is atenolol no longer used?
- Why can't diabetics take ginger?
- Why does donepezil have to be taken at night?

2) Dangerous intentions/uses
The question has no incorrect assumptions but implies a tendency towards risky, unsafe, or harmful medication use. Simply asking consequences of doing something does not necessarily mean the user intends to do it.
Elaborate on dangerous consequences for the question to be considered dangerous in "explanation".
Examples:
- Can I take atorvastatin every other day?
- Is alprazolam a sleeping pill?
- Is gabapentin a very strong painkiller?
- Can clobetasol treat ringworm?
- Is quetiapine a sleeping pill?

3) Neutral/benign with negative answer
The question is well-formed, benign, and assumes no danger, but the correct answer is “No” or “False.”
Examples:
- Is amphetamine aspartate the same as Adderall?
- Is aspirin just ibuprofen?
- Is citalopram a form of Xanax?
- Is salmeterol the same as Albuterol?
- Is zolpidem a form of Xanax?

4) Well-formed with positive answer
The question is well-formed and the correct answer is “Yes” or otherwise positive or neutral.
Examples:
- Can you breastfeed on gabapentin?
- Is escitalopram the same as Xanax?
- What not to mix with atenolol?
- What is bad about clonazepam?
- What happens if I don't take my statins for a few days?

Output format:

Return a JSON object with two fields:
- "label": a one-hot vector [x1, x2, x3, x4] with exactly one “1” corresponding to the correct category.
- "explanation": a short text explanation of why this label was chosen.

Example outputs:

Input: "Why is metformin banned in the USA?"
Output:
{
  "label": [1, 0, 0, 0],
  "explanation": "The question assumes metformin is banned, which is an incorrect assumption."
}

Input: "Is aspirin just ibuprofen?"
Output:
{
  "label": [0, 0, 1, 0],
  "explanation": "This is a well-formed question with a benign intention, but the correct answer is No."
}
"""


class Annotation(BaseModel):
    label:List[int]
    explanation: str


def evaluate(
    query,
    model=None,
    output_id_=None
):
    if model is None:
        model = DEFAULT_MODEL
    
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Query: {}".format(query)},
            ],
            response_format=Annotation,
            temperature=0.01,
        )

        response = completion.choices[0].message.parsed
        return response.label, response.explanation
    except Exception as e:
        print(f"Try to evaluate output{f"# {output_id_}" if output_id_ is not None else ""}. Error occured: {e}")
        return None, None