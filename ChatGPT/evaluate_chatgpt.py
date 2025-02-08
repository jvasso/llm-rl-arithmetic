from openai import OpenAI
import json
import torch
from transformers import AutoTokenizer
from dataset.generate_data import MultiDigitAdditionDataset
from dotenv import load_dotenv
import os


# Load environment variables from .env file
load_dotenv()

api_key = os.environ.get("OPENAI_API_KEY")

client = OpenAI(
  api_key=api_key,
)

tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")

BS = 2048
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = MultiDigitAdditionDataset(10, 10, 10, "english", tokenizer, model=None, use_flash=True, batch_size=BS, device=device).addition_datasets

dataset = dataset[5].dataset

prompt = """The addition algorithm can be formalized as follows for adding two integers:
    1. Alignment: Arrange the integers in a vertical format with each digit corresponding to its place value, aligning the digits from right to left (least significant to most significant). If necessary, prepend zeros to equalize the number of digits.
    2. Digit-wise Addition: Starting from the least significant digit (rightmost), add the corresponding digits of the two integers. If the sum of any pair of digits is greater than or equal to 10, the result is a two-digit number. Write the unit digit under the line at the current place value and carry the tens digit to the next higher place value.
    3. Propagation of Carry: For each subsequent place value, add the digits along with any carry from the previous step. Repeat the process of writing the unit digit and carrying the tens digit as required.
    4. Compilation of Result: Continue this procedure for all place values. The sequence of digits obtained from right to left constitutes the final sum of the two integers. 
    Solve the following addition problem using the addition algorithm, returning the final result between brakes: """

generated_codes = {}


for index, example in enumerate(dataset):

    problem = example[0]

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
          {"role": "system", "content": prompt},
          {"role": "user", "content": problem}
        ]
    )

    generated_codes[str(index)] = [problem, completion.choices[0].message.content, example[2]]

with open('chatgpt3.5_outputs_upperbound_6_digits.json', 'w') as f:
    json.dump(generated_codes, f)