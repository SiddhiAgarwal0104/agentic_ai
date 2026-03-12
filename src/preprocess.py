import re
import string

HINGLISH_MAP = {
    "mera": "my",
    "karo": "do",
    "nahi": "not",
    "hospital": "hospital",
    "shikayat": "complaint"
}

def clean_text(text):
    text = text.lower()

    text = re.sub(r'http\S+|www\S+|<.*?>', '', text)

    text = text.translate(str.maketrans('', '', string.punctuation))

    text = " ".join(text.split())

    return text


def normalize_hinglish(text):
    words = text.split()
    normalized_words = [HINGLISH_MAP.get(word, word) for word in words]
    return " ".join(normalized_words)


def preprocessing_node(user_input):
    cleaned = clean_text(user_input)
    return cleaned


# Test only when file is run directly
if __name__ == "__main__":
    sample_query = "Mera RATION card issue solve karo ASAP!!!"
    print(f"Before: {sample_query}")
    print(f"After:  {preprocessing_node(sample_query)}")