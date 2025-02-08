import numpy as np
import logging
import socket
from torch.utils.data import Dataset
from collections import defaultdict
import torch
import random
import copy
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from collections import Counter
import logging
from tqdm import tqdm
import argparse
import pickle



host_name = socket.gethostname()
CHECKPOINT_DIR = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/{}/code/checkpoints/supervised/".format(host_name)
PAD_VALUE = -100

class MultiDigitAdditionDataset(Dataset):

    # Our main training set is a collection of num_examples examples, each with main_num_digits digits.
    # We also have a collection of num_old_examples examples for each digit from 1 to main_num_digits - 1.
    def __init__(self,
                 num_examples,
                 primary_num_digits,
                 num_old_examples,
                 dataset_type,
                 tokenizer,
                 type="decomp",
                 silent=False,
                 force_min_number_examples=0,
                 use_flash=False,
                 flash_multiplier=3,
                 model=None,
                 batch_size=-1,
                 device = None,
                 repeat_subproblem=False,
                 stop_generation_digit=np.inf,
                 uuid=None):

        # print("Starting the data generation with the model")
        self.uuid = uuid
        self.num_examples = num_examples
        self.primary_num_digits = primary_num_digits
        self.num_old_examples = num_old_examples
        self.dataset_type = dataset_type
        self.tokenizer = tokenizer
        self.type = type
        self.silent = silent
        self.force_min_number_examples = force_min_number_examples
        self.use_flash = use_flash
        self.model = model
        self.batch_size = batch_size
        self.repeat_subproblem = repeat_subproblem
        self.stop_generation_digit = stop_generation_digit
        
        if self.model and device:
            self.model.to(device)

        primary_digit_dataset = AdditionDataset(num_examples if self.stop_generation_digit >= primary_num_digits else 0, 
                                                primary_num_digits,
                                                dataset_type,
                                                tokenizer,
                                                type=type,
                                                silent=silent,
                                                force_min_number_examples=force_min_number_examples,
                                                model=model,
                                                batch_size=batch_size,
                                                device = device,
                                                repeat_subproblem=repeat_subproblem,
                                                uuid=uuid)
        
        self.pad_length = primary_digit_dataset.max_output_length

        # Only ever model generate the primary digit dataset

        self.addition_datasets = []
        for num_digits in range(1, primary_num_digits):

            if num_digits >= self.stop_generation_digit:
                num_examples_to_generate = 0
            else:
                num_examples_to_generate = self.num_old_examples

            self.addition_datasets.append(
                                  AdditionDataset(num_examples_to_generate,
                                                  num_digits,
                                                  self.dataset_type,
                                                  tokenizer,
                                                  force_pad_len=self.pad_length,
                                                  type=type,
                                                  silent=silent,
                                                  force_min_number_examples=force_min_number_examples,
                                                  repeat_subproblem=repeat_subproblem,
                                                  uuid=uuid))
        self.addition_datasets.append(primary_digit_dataset)

        if use_flash:
            self.flash_datasets = []

            for num_digits in range(1, primary_num_digits):
                
                if num_digits >= self.stop_generation_digit:
                    num_examples_to_generate = 0
                else:
                    num_examples_to_generate = flash_multiplier * self.num_old_examples

                self.flash_datasets.append(
                                   AdditionDataset(num_examples_to_generate,
                                                   num_digits,
                                                   self.dataset_type,
                                                   tokenizer,
                                                   force_pad_len=self.pad_length,
                                                   type="flash",
                                                   silent=silent,
                                                   force_min_number_examples=flash_multiplier*force_min_number_examples,
                                                   repeat_subproblem=repeat_subproblem,
                                                   uuid=uuid))
            
        
            if primary_num_digits >= self.stop_generation_digit:
                num_examples_to_generate = 0            
            else:
                num_examples_to_generate = flash_multiplier * self.num_examples

            self.flash_datasets.append(AdditionDataset(num_examples_to_generate,
                                                       primary_num_digits,
                                                       self.dataset_type,
                                                       tokenizer,
                                                       force_pad_len=self.pad_length,
                                                       type="flash",
                                                       silent=silent,
                                                       force_min_number_examples=flash_multiplier*force_min_number_examples,
                                                       model=model,
                                                       batch_size=batch_size,
                                                       device=device,
                                                       repeat_subproblem=repeat_subproblem,
                                                       uuid=uuid))
        else:
            self.flash_datasets = []

        self.regenerate_partial_sums()
    
    # Merge all the datasets together
    def merge(self, other_datasets):
        for other_data in other_datasets:
            assert other_data.primary_num_digits == self.primary_num_digits, "Can't merge datasets with different primary num digits"

            self.pad_length = max(self.pad_length, other_data.pad_length)

            for i in range(len(self.addition_datasets)):
                self.addition_datasets[i].merge(other_data.addition_datasets[i])
            
            if self.use_flash and other_data.use_flash:
                for i in range(len(self.flash_datasets)):
                    self.flash_datasets[i].merge(other_data.flash_datasets[i])
        
        self.pad_everything()
        self.regenerate_partial_sums()
    
    def load_from_file(self, num_digits, filename, type, percentage=0.1):

        if type == "flash":
            self.flash_datasets[num_digits - 1].load_from_file(filename, percentage=0.1)
            self.pad_length = max(self.pad_length, self.flash_datasets[num_digits - 1].max_output_length)
        elif type == "decomp":
            self.addition_datasets[num_digits - 1].load_from_file(filename, percentage=0.1)
            self.pad_length = max(self.pad_length, self.addition_datasets[num_digits - 1].max_output_length)
        else:
            raise Exception("Invalid type")
        
        self.pad_everything()
        self.regenerate_partial_sums()

    
    # Pad everything to self.pad_length
    def pad_everything(self):
        for dataset in self.addition_datasets:
            dataset.max_output_length = self.pad_length
            dataset.pad_everything()
        
        for dataset in self.flash_datasets:
            dataset.max_output_length = self.pad_length
            dataset.pad_everything()

    # Split out the validation data in place
    def split(self, validation_size):

        val_datasets = []
        for num_digits in range(1, self.primary_num_digits + 1):
            current_addition_dataset = self.get_addition_dataset(num_digits)

            if validation_size > len(current_addition_dataset):
                logging.info("No validation set for " + str(num_digits) + " digit addition")
                val_dataset = []
            else:
                train_dataset, val_dataset = torch.utils.data.random_split(current_addition_dataset, [len(current_addition_dataset) - validation_size, validation_size])
                if num_digits <= 2:
                    # Overfit on 1-2 digit addition exactly.
                    train_dataset = current_addition_dataset

                self.set_addition_dataset(num_digits, train_dataset, "decomp")

            val_datasets.append(val_dataset)
        
        if self.use_flash:
            for num_digits in range(1, self.primary_num_digits + 1):
                current_flash_dataset = self.get_flash_dataset(num_digits)

                train_dataset, val_dataset = torch.utils.data.random_split(current_flash_dataset, [len(current_flash_dataset) - validation_size, validation_size])
                if num_digits <= 2:
                    # Overfit on 1-2 digit addition exactly.
                    train_dataset = current_flash_dataset

                self.set_addition_dataset(num_digits, train_dataset, "flash")
                val_datasets.append(val_dataset)
        
        self.regenerate_partial_sums()
        return val_datasets

    def regenerate_partial_sums(self):
        self.partial_sums = [len(self.addition_datasets[0])]
        for i in range(1, self.primary_num_digits):
            self.partial_sums.append(len(self.addition_datasets[i]) + self.partial_sums[-1])

        if self.use_flash:
            for i in range(self.primary_num_digits):
                self.partial_sums.append(len(self.flash_datasets[i]) + self.partial_sums[-1])

    def set_addition_dataset(self, num_digits, dataset, type):
        if type == "decomp":
            self.addition_datasets[num_digits - 1] = copy.deepcopy(dataset)
        elif type == "flash":
            self.flash_datasets[num_digits - 1] = copy.deepcopy(dataset)
        else:
            raise Exception("Invalid type")
        self.regenerate_partial_sums()

    def set_flash_dataset(self, num_digits, dataset):
        if not self.use_flash:
            raise Exception("Cannot set flash dataset if not using flash")

        self.flash_datasets[num_digits - 1] = copy.deepcopy(dataset)
        self.regenerate_partial_sums()

    def get_addition_dataset(self, num_digits):
        return self.addition_datasets[num_digits - 1]
    
    def get_flash_dataset(self, num_digits):

        if not self.use_flash:
            raise Exception("Cannot get flash dataset if not using flash")


        return self.flash_datasets[num_digits - 1]
    
    def __len__(self):
        return self.partial_sums[-1]
    
    def __getitem__(self, index):

        # Find highest number in self.partial_sums lower than index in one line
        digit_index = np.searchsorted(self.partial_sums, index, side="right")
        if digit_index > 0:
            example_index = index - self.partial_sums[digit_index - 1]
        else:
            example_index = index

        if digit_index >= len(self.addition_datasets):

            if not self.use_flash:
                raise Exception("Cannot get flash dataset if not using flash")

            return self.flash_datasets[digit_index - len(self.addition_datasets)].__getitem__(example_index)
        else:
            return self.addition_datasets[digit_index].__getitem__(example_index)

class AdditionDataset(Dataset):
    def __init__(self,
                 num_examples,
                 num_digits,
                 dataset_type,
                 tokenizer,
                 force_pad_len=None,
                 type="decomp",
                 silent=False,
                 force_min_number_examples=0,
                 model=None,
                 batch_size=-1,
                 device=None,
                 repeat_subproblem=False,
                 uuid=None):

        self.num_examples = num_examples
        self.uuid = uuid
        self.num_digits = num_digits
        self.dataset_type = dataset_type
        self.generated = {}
        self.tokenizer = tokenizer
        self.type = type
        self.silent = silent
        self.force_min_number_examples = force_min_number_examples
        self.model = model
        self.max_output_length = 1000
        self.batch_size = batch_size
        self.device = device
        self.repeat_subproblem = repeat_subproblem

        # If we're only generating decomposition problems, we can't generate any problems with only one digit.
        # Similar for full problems, where they aren't necessary because of flashing.
        if (self.type == "decomp" or self.type == "full") and num_digits == 1:
            self.num_examples = 0

        if self.type == "remove":
            total_potential_problems = 10 ** (self.num_digits) - 10 ** (self.num_digits - 1)
        elif self.type == "decomp" or self.type == "full" or self.type == "flash":
            total_potential_problems = (10 ** (self.num_digits) - 10 ** (self.num_digits - 1))**2
        else:
            raise ValueError("Unknown type {}".format(self.type))

        if self.num_examples >= total_potential_problems:
            logging.warning("The requested number of examples is greater than the total number of potential problems. We can only deliver {} examples".format(total_potential_problems))
            method = "all"
            self.num_examples = total_potential_problems
        else:
            method = "random"

        assert isinstance(self.num_digits, int) and self.num_digits >= 1, "Digit length must be a positive integer"

        self.generate_data(method=method)
        # self.force_seperation()

        if force_pad_len is not None:
            self.max_output_length = force_pad_len
        else:
            self.calculate_max_output_length()

        self.tokenized_dataset = tokenize_data(self.tokenizer, self.dataset, self.type, self.max_output_length)

        # Stop the infinite loop
        while self.num_examples > 0 and self.force_min_number_examples > self.num_examples:
            # logging.info("Doubling dataset size to {}".format(self.num_examples * 2))
            self.num_examples *= 2
            self.tokenized_dataset += copy.deepcopy(self.tokenized_dataset)
    
    def merge(self, other):
        self.dataset += other.dataset
        self.tokenized_dataset += other.tokenized_dataset
        self.num_examples += other.num_examples
    
    def force_seperation(self, deliminator="|||"):
        for i, (question, answer, numerical_answer) in enumerate(self.dataset):
            self.dataset[i] = (self.delimit(question, deliminator), self.delimit(answer, deliminator), numerical_answer)

    def write_to_file(self, filename):
        with open(filename, "wb") as f:
            pickle.dump((self.dataset, self.tokenized_dataset, self.max_output_length), f)
    
    # Make sure to regenerate partial sums in MultiAdditionDataset after loading
    def load_from_file(self, filename, percentage=0.1):
        with open(filename, "rb") as f:
            new_dataset, new_tokenized_dataset, new_max_output_length = pickle.load(f)
        
        # Randomly choose 10% of the indices to keep
        indices_to_keep = random.sample(range(len(new_dataset)), int(percentage * len(new_dataset)))

        self.dataset += [new_dataset[i] for i in indices_to_keep]
        self.tokenized_dataset += [new_tokenized_dataset[i] for i in indices_to_keep]
        
        self.max_output_length = max(self.max_output_length, new_max_output_length)
        self.num_examples += len(indices_to_keep)
    
    def clear(self):
        self.dataset = []
        self.tokenized_dataset = []
        self.num_examples = 0
        self.max_output_length = 0

    # squash every number with a deliminator to split it up
    def delimit(self, str, deliminator):
        new_str = ""
        last_is_digit = False
        for c in str:
            if c.isdigit() or last_is_digit:
                new_str += deliminator

            if c.isdigit():
                last_is_digit = True
            else:
                last_is_digit = False
            
            new_str += c
        
        if last_is_digit:
            new_str += deliminator
        
        return new_str
    
    def calculate_max_output_length(self):
        self.max_output_length = 0 

        for question, answer, numerical_answer in random.sample(self.dataset, min(1000, len(self.dataset))):
            self.max_output_length = max(self.max_output_length, len(self.tokenizer(answer).input_ids))
        
        # Add 20% margin
        self.max_output_length = int(1.2 * self.max_output_length)
        if self.max_output_length == 0:
            logging.warn("Max output length is 0")

    def __len__(self):
        return len(self.tokenized_dataset)

    def __getitem__(self, idx):
        return self.tokenized_dataset[idx]

    def generate_data(self, method="random"):

        if self.type == "remove":
            self.generate_data_remove(method)
        else:
            self.generate_data_addition(method)
    
    def generate_data_remove(self, method):
        self.generated[None] = True
        self.dataset = []
        num1 = None

        if method == "random":
            numbers = []

            for _ in range(self.num_examples):

                # Generate two numbers not in the generated set
                while num1 in self.generated:
                    num1 = number_with_digits(self.num_digits)
                
                self.generated[num1] = True
                numbers.append(num1)
        elif method == "all":
            numbers = [n1 for n1 in range(10 ** (self.num_digits - 1), 10 ** (self.num_digits))]

            assert len(numbers) == (10 ** (self.num_digits) - 10 ** (self.num_digits - 1)), "The number of problems generated does not match the total number of potential problems"
        else:
            raise Exception("Invalid method")

        for num in numbers:
            question = str(num) + ". "
            answer = "The answer is {}.".format(str(num)[:-1])

            self.dataset.append((question, answer, int(num // 10)))
    
    def generate_data_addition(self, method):

        non_int_answers = 0
        num1, num2, carry = None, None, None
        self.generated[None, None, None] = True
        self.dataset = []

        if method == "random":
            numbers = []

            for _ in range(self.num_examples // 2):

                # Generate two numbers not in the generated set
                while (num1, num2, carry) in self.generated:
                    num1 = number_with_digits(self.num_digits)
                    num2 = number_with_digits(self.num_digits)
                    # carry = random.randint(0, 1) ### HERE FOR ADD A +1, I DONT UNDERSTAND WHY IN THE EXPE
                    carry = 0
                
                self.generated[num1, num2, carry] = True
                self.generated[num2, num1, carry] = True
                numbers.append((num1, num2, carry))
                numbers.append((num2, num1, carry))
        elif method == "all":
            numbers = [(n1, n2, carry) for n1 in range(10 ** (self.num_digits - 1), 10 ** (self.num_digits)) for n2 in range(10 ** (self.num_digits - 1), 10 ** (self.num_digits)) for carry in [0, 1]]
            assert len(numbers) == 2 * (10 ** (self.num_digits) - 10 ** (self.num_digits - 1))**2, "The number of problems generated does not match the total number of potential problems"
        else:
            raise Exception("Invalid method")

        if not self.model:

            # assert self.num_digits < 6 or self.num_examples == 0, "Cannot generate examples for 10 or more digits"

            for num1, num2, carry in numbers:
                word_answer = "A: " + generate_solution_english(num1, num2, carry, "", decomp_only=(self.type == "decomp"), repeat_subproblem=self.repeat_subproblem)
                symbol_answer = "A: " + generate_solution_scratchpad(num1, num2, carry, "")

                question = generate_question(num1, num2, carry)
                numerical_answer = num1 + num2 + carry

                if self.type == "flash":
                    output_answer = "A: " + str(numerical_answer)
                else:
                    if self.dataset_type == "scratchpad":
                        output_answer = symbol_answer
                    elif self.dataset_type == "english":
                        output_answer = word_answer
                    else:
                        raise Exception("Invalid dataset type")
                
                datapoint = question, output_answer, numerical_answer
                self.dataset.append(datapoint)
 
        else:
            assert self.batch_size > 0 and isinstance(self.batch_size, int), "Batch size must be a positive integer"

            # Loop through numbers in size batch_size
            num1s, num2s, carries = [], [], []
            rejected, accepted = [0, 0], [0, 0]

            consistency_check = {}

            num_attempted = 0
            for i, (num1, num2, carry) in enumerate(tqdm(numbers)):

                num1s.append(num1)
                num2s.append(num2)
                carries.append(carry)

                # Or it's the last batch
                if i == len(numbers) - 1 or len(num1s) == self.batch_size:
                    num_attempted += len(num1s)
                    truthful_numerical_answers = [num1 + num2 + carry for num1, num2, carry in zip(num1s, num2s, carries)]
                    questions = [generate_question(num1, num2, carry) for num1, num2, carry in zip(num1s, num2s, carries)]

                    # You need to set numerical_answer and output_answer correctly in the branch below
                    if self.type == "flash":
                        numerical_answers = self.calculate_numerical_answer(num1s, num2s, carries)
                        output_answers = ["A: " + str(numerical_answer) for numerical_answer in numerical_answers]

                        final_numerical_answers = []
                        # Need to make sure that the non int answers are discarded
                        for i, (numerical_answer, truthful_numerical_answer) in enumerate(zip(numerical_answers, truthful_numerical_answers)):
                            try:
                                final_numerical_answers.append(int(numerical_answer))
                            except:
                                non_int_answers += 1
                                # Give a random answer so it's discarded in the consistency check
                                final_numerical_answers.append(random.randint(0, 1000000))
                                
                        numerical_answers = final_numerical_answers
                        # datapoints = [(question, output_answer, numerical_answer) for question, output_answer, numerical_answer in zip(questions, output_answers, numerical_answers)]


                        # num_correct = sum([1 if int(numerical_answer == truthful_numerical_answer else 0 for numerical_answer, truthful_numerical_answer in zip(numerical_answers, truthful_numerical_answers)])
                    else:

                        output_answers = self.generate_answer(num1s, num2s, carries, type="decomp")

                        # Each entry is a tuple of (raw_number_pieces, partial_answer) 
                        subproblems = [extract_decomp_answer(output_answer, full=True)[1] for output_answer in output_answers]

                        sub_n1s, sub_n2s, sub_carries = [], [], []
                        for raw_number_pieces, partial_answer in subproblems:
                            num1, num2 = raw_number_pieces[0], raw_number_pieces[1]
                            carry = raw_number_pieces[2] if len(raw_number_pieces) == 3 else 0
                            sub_n1s.append(num1)
                            sub_n2s.append(num2)
                            sub_carries.append(carry)

                        subproblem_raw_answers = self.generate_answer(sub_n1s, sub_n2s, sub_carries, type="flash")
                        numerical_answers = [remove_non_numeric(answer) for answer in subproblem_raw_answers]

                    for num1, num2, carry, question, output_answer, numerical_answer, truthful_numerical_answer in \
                                zip(num1s, num2s, carries, questions, output_answers, numerical_answers, truthful_numerical_answers):

                        consistency_check[num1, num2, carry] = (question, output_answer, numerical_answer, truthful_numerical_answer)
                    
                    for num1, num2, carry in consistency_check.keys():

                        if self.type == "flash":
                            correct_answer = num1 + num2 + carry
                            is_correct = consistency_check[num1, num2, carry][2] == correct_answer

                        else:
                            correct_output_answer = "A: " + generate_solution_english(num1, num2, carry, "", decomp_only=True, repeat_subproblem=self.repeat_subproblem)
                            is_correct = consistency_check[num1, num2, carry][1] == correct_output_answer

                        if consistency_check[num1, num2, carry][2] == consistency_check[num2, num1, carry][2]:
                            self.dataset.append(consistency_check[num1, num2, carry][:3])
                            accepted[is_correct] += 1
                        else:
                            rejected[is_correct] += 1

                    num1s, num2s, carries = [], [], []

                else:
                    continue


            if self.model and num_attempted > 0:

                try:
                    message = "For {} digit {} problems, we accepted {} correct answers and {} wrong answers. We rejected {} correct answers and {} wrong answers".format(self.num_digits, self.type, accepted[1], accepted[0], rejected[1], rejected[0])
                    master_location = CHECKPOINT_DIR + self.uuid + "/master.log"
                    with open(master_location, "a+") as w:
                        w.write(message + "\n")
                except Exception as e:
                    print(str(e))
                    print(self.uuid)
                    print("Failed to write to master log")

                accuracy = accepted[1] / (accepted[1] + accepted[0])

                if accuracy < 0.9:
                    raise Exception("Accuracy of {} is too low. Quiting at digit length {}".format(accuracy, self.num_digits))
                

    def calculate_numerical_answer(self, n1s, n2s, orig_carries):

        num1s, num2s, carries = n1s.copy(), n2s.copy(), orig_carries.copy()
        actual_answers = [n1 + n2 + carry for n1, n2, carry in zip(num1s, num2s, carries)]
        guesses = [[] for _ in range(len(num1s))]
        full_partial_answers = ["" for _ in range(len(num1s))]
        num_digits = len(str(num1s[0]))

        # Can't decomp if we don't have at least two digits
        # We test at most 5 decompositions
        for _ in range(min(5, num_digits - 1)):
            decomp_outputs = self.generate_answer(num1s, num2s, carries, "decomp")
            num1s, num2s, carries = [], [], []

            for i, decomp_output in enumerate(decomp_outputs):
                _, (raw_number_pieces, partial_answer) = extract_decomp_answer(decomp_output, full=True)
                num1, num2 = raw_number_pieces[0], raw_number_pieces[1]
                carry = raw_number_pieces[2] if len(raw_number_pieces) == 3 else 0
                num1s.append(num1)
                num2s.append(num2)
                carries.append(carry)
                full_partial_answers[i] = str(partial_answer) + full_partial_answers[i]

            generated_answers = self.generate_answer(num1s, num2s, carries, "flash")
            flash_guesses = [str(remove_non_numeric(guess)) for guess in generated_answers]

            for i, flash_guess in enumerate(flash_guesses):
                guesses[i].append(flash_guess + full_partial_answers[i])

            # print("Our current partial answer is {}. The next problem is {} + {} + {} = ? Our flash guess is {}.".format(full_partial_answer, num1, num2, carry, flash_guess))
        
        # Return majority vote guess
        final_answers = []
        for i, (answer, guess) in enumerate(zip(actual_answers, guesses)):

            # If we have no guesses, then we just return an empty string
            try:
                candidate_ans = Counter(guess).most_common(1)[0][0]
                candidate_agreement = Counter(guess).most_common(1)[0][1]

                final_answers.append(candidate_ans)

                if int(final_answers[-1]) != answer:
                    pass

            except:
                if len(final_answers) < i:
                    final_answers.append("")
        
        return final_answers
    
    def pad_everything(self):

        for datapoint in self.tokenized_dataset:
            # Pad each datapoint

            try:
                assert len(datapoint['labels']) <= self.max_output_length, "Output length is too long"
            except:
                print(len(datapoint['labels']))
                print(self.max_output_length)
                raise
            padding = torch.full((self.max_output_length - len(datapoint['labels']),), PAD_VALUE, dtype=datapoint['labels'].dtype, device=datapoint['labels'].device)
            datapoint['labels'] = torch.cat((datapoint['labels'], padding))


    # num1s, num2s, carries are lists of numbers which wil lbe batched
    def generate_answer(self, num1s, num2s, carries, type):
        questions = [generate_question(num1, num2, carry) for num1, num2, carry in zip(num1s, num2s, carries)]
        prompts = [tokenize_datapoint(self.tokenizer, type, self.max_output_length, 0, question, "", -1) for question in questions]
        input_ids = torch.stack([prompt['input_ids'] for prompt in prompts]).to(self.device)
        attention_mask = torch.stack([prompt['attention_mask'] for prompt in prompts]).to(self.device)

        # jinput_ids = prompt['input_ids'].unsqueeze(0)
        # attention_mask = prompt['attention_mask'].unsqueeze(0)

        raw_output = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens = self.max_output_length)

        outputs = [self.tokenizer.decode(tokens, skip_special_tokens=True) for tokens in raw_output]
        return outputs



def generate_question(num1, num2, carry):
    if carry:
        question = "Q: " + str(num1) + "+" + str(num2) + "+1=?"
    else:
        question = "Q: " + str(num1) + "+" + str(num2) + "=?"
    
    return question

def tokenize_datapoint(tokenizer, type, max_output_length, idx, question, answer, numerical_answer):

    if type == "full" or type == "decomp":
        sentence_input = "Add slow.\n" + question
        max_length = 32
    elif type == "remove":
        sentence_input = "Remove the last digit of the following number: " + question
        max_length = 32
    elif type == "flash":
        sentence_input = "Add fast.\n" + question
        max_length = 32
    else:
        raise Exception("Invalid type")

    max_length += 50
    model_inputs = tokenizer(sentence_input, padding="max_length", truncation=True, max_length=max_length)
    raw_labels = tokenizer(answer, padding="max_length", truncation=True, max_length=max_output_length).input_ids
    labels_with_ignore_index = [label if label != 0 else PAD_VALUE for label in raw_labels]

    model_inputs["labels"] = labels_with_ignore_index

    # We don't use numerical answer and it overflows
    model_inputs["numerical_answer"] = 0
    model_inputs["lookup_id"] = idx

    for key, value in model_inputs.items():
        model_inputs[key] = torch.tensor(value)
    
    return model_inputs

def tokenize_data(tokenizer, raw_dataset, type, max_output_length):
    tokenized_dataset = []

    for i, (question, answer, numerical_answer) in enumerate(raw_dataset):
        tokenized_dataset.append(tokenize_datapoint(tokenizer, type, max_output_length, i, question, answer, numerical_answer))

    return tokenized_dataset

def remove_non_numeric(str):
    try:
        return int("".join([c for c in str if c.isdigit()]))
    except ValueError:
        logging.info("Tried to convert non-numeric string to int")
        return 0

# The thought is NOT guaranteed to be correct
def extract_answer_from_prompt(thought):
    # Extract everything between Q: and A: in thought
    query = thought.split("Q:")[1].split("A:")[0]
    nums = query.split("+")

    return sum([remove_non_numeric(num) for num in nums])

def extract_decomp_answer(solution, full=False):
    try:
        subproblem = solution.split("The next subproblem is")[-1]
        numbers = subproblem.split("+")
        partial_answer_str = solution.split("partial answer is")[1].split(".")[0]
        partial_answer = remove_non_numeric(partial_answer_str)

        raw_number_pieces = [remove_non_numeric(num) for num in numbers]
        subproblem_ans = sum(raw_number_pieces)

        ans = int(str(subproblem_ans) + str(partial_answer))
        if full:
            return ans, (raw_number_pieces, partial_answer)
        else:
            return ans

    except:
        logging.info("The solution ````{}'''' does not contain the final answer in the proper format".format(solution))
        if full:
            return None, None, None
        else:
            return None

def extract_full_answer(solution):
    try:
        partial_str = solution.split("The final answer is")[1]

        # Remove all non-numeric characters
        return remove_non_numeric(partial_str)

    except:
        logging.info("The solution ````{}'''' does not contain the final answer in the proper format".format(solution))
        return None

def extract_answer_from_solution(solution, type="decomp"):

    if type == "decomp":
        return extract_decomp_answer(solution)
    elif type == "full":
        return extract_full_answer(solution)
    elif type == "remove" or type == "flash":
        return extract_remove_answer(solution)
    else:
        raise Exception("Invalid type")

# With one digit answers, there will be no answer
def extract_remove_answer(solution):
    try:
        return remove_non_numeric(solution.split()[-1])
    except:
        return 0
        
def generate_solution_scratchpad(num1, num2, carry, previous_solution):
    assert len(str(num1)) == len(str(num2)), "The numbers must be the same length"
    assert num1 >= 0 and num2 >= 0, "The numbers must be positive"

    answer = num1 + num2
    carry = 0

    partial_ans = "<scratchpad>\n"
    partial_sum = ""

    while num1 > 0 or num2 > 0 or carry > 0:

        last_num1, last_num2 = num1 % 10, num2 % 10
        partial_sum = str((last_num1 + last_num2 + carry) % 10) + partial_sum
        carry = (last_num1 + last_num2 + carry) // 10
        partial_ans += "{} + {}, {}, C: {} | ".format(last_num1, last_num2, partial_sum, carry)

        num1 //= 10
        num2 //= 10

    return partial_ans + "</scratchpad>\n{}".format(answer)


def generate_solution_english(num1, num2, carry, previous_solution, use_commas=False, decomp_only=False, repeat_subproblem=False):

    assert len(str(num1)) == len(str(num2)), "The numbers must be the same length"
    assert num1 >= 0 and num2 >= 0, "The numbers must be positive"

    last_digit_1 = num1 % 10
    last_digit_2 = num2 % 10

    # If this is the first step, we need to repeat the numbers
    # Turning off the repeat for now. Hopefully this doesn't break anything
    partial_ans = ""
    if repeat_subproblem:
        if previous_solution == "":
            if carry:
                partial_ans = "The next subproblem is {:,} + {:,} + 1. ".format(num1, num2)
            else:
                partial_ans = "The next subproblem is {:,} + {:,}. ".format(num1, num2)

        else:
            partial_ans = ""

    partial_ans += "The first number's last digit is {}. The second number's last digit is {}. ".format(last_digit_1, last_digit_2)

    if carry:
        partial_ans += "We are carrying a 1 from the previous step. {}+{}+1={}. ".format(last_digit_1, last_digit_2, last_digit_1 + last_digit_2 + 1)
    else:
        partial_ans += "{}+{}={}. ".format(last_digit_1, last_digit_2, last_digit_1 + last_digit_2)
    
    next_sum = last_digit_1 + last_digit_2 + carry
    next_last_digit = next_sum % 10
    next_carry = next_sum >= 10

    partial_ans += "The last digit of this sum is {}, so ".format(next_last_digit)

    if len(previous_solution) > 0:
        partial_ans += "we prepend it to the previous partial answer of {} to get the new partial answer of {}. ".format(
            previous_solution, str(next_last_digit) + previous_solution)
    else:
        partial_ans += "our initial partial answer is {}. ".format(next_last_digit)
    
    if next_carry:
        partial_ans += "Since this sum is two digits, we carry a 1 to the next step. "
    
    if num1 < 10 and not next_carry:
        partial_ans += "We have reached the end of the algorithm. The final answer is {:,}. ".format(int(str(next_last_digit) + previous_solution))
        return partial_ans

    partial_ans += "We can now recursively solve a smaller problem. Removing the last digit from each number, we have {:,} and {:,}. ".format(num1 // 10, num2 // 10)

    if next_carry:
        partial_ans += "The next subproblem is {:,} + {:,} + 1. ".format(num1 // 10, num2 // 10)
    else:
        partial_ans += "The next subproblem is {:,} + {:,}. ".format(num1 // 10, num2 // 10)
    
    if not decomp_only:
        partial_ans += generate_solution_english(num1 // 10, num2 // 10, next_carry, str(next_last_digit) + previous_solution, decomp_only=False, repeat_subproblem=repeat_subproblem)

    if not use_commas:
        partial_ans = partial_ans.replace(",", "")
    
    return partial_ans

# Random number of n digits
def number_with_digits(n):
    if n <= 0:
        raise ValueError("The number of digits must be greater than 0")

    min_num = 10 ** (n - 1)
    max_num = (10 ** n) - 1

    return random.randint(min_num, max_num)

def main():


    tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")

    BS = 2048
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reference = MultiDigitAdditionDataset(1000, 10, 1000, "english", tokenizer, use_flash=False)
    full_dataset = MultiDigitAdditionDataset(100, 10, 1000, "english", tokenizer, model=None, use_flash=True, batch_size=BS, device=device)
    addition_dataset = reference.addition_datasets

    for x in addition_dataset:
        print(x)
        a = 0

   
if __name__ == "__main__":
    main()