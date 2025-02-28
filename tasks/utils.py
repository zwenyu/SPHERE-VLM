import string, re
import random
import torch
import numpy as np
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag


# Ensure required NLTK resources are downloaded
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')


# set seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def clean_string(input_str):
    # Create a translation table to remove punctuation
    translator = str.maketrans("", "", string.punctuation)

    # Remove punctuation and trailing spaces
    cleaned_str = input_str.translate(translator).strip()

    return cleaned_str.lower()


def remove_stopwords(input_str):
    stop_words = set(["a", "the"])
    return " ".join([word.lower() for word in input_str.split() if word.lower() not in stop_words])


def stem_words(text):
    stemmer = PorterStemmer()
    tokens = word_tokenize(text)
    stemmed_words = [stemmer.stem(word) for word in tokens]
    return " ".join(stemmed_words)


def lemmatize_words(text):
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(lemmatized_words)


def longest_matching_subsequence(str1, str2):

    str1 = lemmatize_words(str1)
    str2 = lemmatize_words(str2)
    
    # Split the strings into words
    words1 = str1.split()
    words2 = str2.split()

    # Lengths of the word lists
    m, n = len(words1), len(words2)

    # Create a DP table initialized with empty lists
    dp = [[([], None, None, None, None) for _ in range(n+1)] for _ in range(m+1)]

    # Fill the DP table
    for i in range(1, m+1):
        for j in range(1, n+1):
            if words1[i-1] == words2[j-1]:
                subseq, start1, start2, _, _ = dp[i-1][j-1]
                new_start1 = start1 if start1 is not None else i-1
                new_start2 = start2 if start2 is not None else j-1
                dp[i][j] = (subseq + [words1[i-1]], new_start1, new_start2, i-1, j-1)
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1], key=lambda x: len(x[0]))

    # The longest subsequence and positions are at the bottom-right corner of the table
    longest_subseq, start1, start2, end1, end2 = dp[m][n]
    if len(longest_subseq) == 0:
        start1 = end1 = m 
        start2 = end2 = n

    return longest_subseq, start1, start2, end1, end2


def find_unique_nouns(str1, str2):
    str1 = lemmatize_words(str1)
    str2 = lemmatize_words(str2)

    # Tokenize and find unique tokens in str1
    tokens_str1 = word_tokenize(str1)
    tokens_str2 = word_tokenize(str2)

    # POS tag
    pos_tags_str1 = pos_tag(tokens_str1)
    unique_pos_tags_str1 = [tag for tag in pos_tags_str1 if tag[0] not in tokens_str2]

    # Extract nouns (NN, NNS, NNP, NNPS)
    unique_nouns_str1 = {word.lower() for word, pos in unique_pos_tags_str1 if pos.startswith('NN')}

    # Remove common terms labeled as nouns
    common_terms = set(["one"])
    unique_nouns_str1 = {word for word in unique_nouns_str1 if word not in common_terms}

    return unique_nouns_str1


def word2num(predicted_answer):
    word_to_number = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
        "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
        "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
        "eighteen": 18, "nineteen": 19, "twenty": 20,
        "no": 0
    }
    pattern = re.compile(r"\b(" + "|".join(word_to_number.keys()) + r")\b")
    predicted_answer_pattern = re.sub(pattern, lambda x: str(word_to_number[x.group().lower()]), predicted_answer)
    return predicted_answer_pattern


def num2word(predicted_answer):
    number_to_word = {
        0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
        5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
        10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen",
        14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen",
        18: "eighteen", 19: "nineteen", 20: "twenty"
    }
    number_to_word = {str(k): v for k, v in number_to_word.items()}
    pattern = re.compile(r"\b(" + "|".join(number_to_word.keys()) + r")\b")
    predicted_answer_pattern = re.sub(pattern, lambda x: str(number_to_word[x.group().lower()]), predicted_answer)
    return predicted_answer_pattern


def process_format_num(predicted_answer):
    if predicted_answer.isdigit():
        return predicted_answer
    else:
        predicted_answer_pattern = word2num(predicted_answer)
        numbers = re.findall(r"\b\d+\.?\d*\b", predicted_answer_pattern)
        if numbers:
            return numbers[0]
        else:
            return "-"


def process_format_mcq(predicted_answer, option_list, text):

    # process only first sentence
    predicted_answer = re.split(r'[.!?]', predicted_answer, maxsplit=1)[0].strip()

    # process predicted answer
    predicted_answer = num2word(clean_string(predicted_answer))

    # process text and detect predictions that mention nouns not in text
    text = num2word(clean_string(text))
    unique_nouns = list(find_unique_nouns(predicted_answer, text))
    option_list_expanded = option_list + unique_nouns

    predicted_answer = remove_stopwords(predicted_answer)
    text = remove_stopwords(text)

    # process option and find match
    len_subseq = {}
    start_subseq = {}
    end_subseq = {}
    for option in option_list_expanded:
        option_pattern = num2word(clean_string(remove_stopwords(option)))

        longest_subseq, start1, start2, end1, end2 = longest_matching_subsequence(predicted_answer, option_pattern)
        len_subseq[option] = len(longest_subseq) / len(option_pattern.split())
        start_subseq[option] = start1
        end_subseq[option] = end1

    max_len_subseq = max(len_subseq.values())
    if max_len_subseq > 0:
        count_max_len_subseq = len([k for k, v in len_subseq.items() if v == max_len_subseq])
        # return longest match
        if count_max_len_subseq == 1:
            selected_option = max(len_subseq, key=len_subseq.get)
            if selected_option not in unique_nouns:
                return selected_option

        # return earliest match
        min_start_subseq = min(start_subseq.values())
        min_end_subseq = min(end_subseq.values())
        count_min_start_subseq = len([k for k, v in start_subseq.items() if v == min_start_subseq])
        count_min_end_subseq = len([k for k, v in end_subseq.items() if v == min_end_subseq])
        if count_min_start_subseq == 1:
            selected_option = min(start_subseq, key=start_subseq.get)
            if selected_option not in unique_nouns:
                return selected_option            
        if count_min_end_subseq == 1:
            selected_option = min(end_subseq, key=end_subseq.get)
            if selected_option not in unique_nouns:
                return selected_option            

    return "-"
