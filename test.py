from model import CharCNN
import pandas as pd
import numpy as np
import torch
import re
import torch.nn.functional as F


def lower(text):
    return text.lower()

def remove_hashtags(text):
    clean_text = re.sub(r'#[A-Za-z0-9_]+', "", text)
    return clean_text


def remove_user_mentions(text):
    clean_text = re.sub(r'@[A-Za-z0-9_]+', "", text)
    return clean_text

def remove_urls(text):
    clean_text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    return clean_text

pretrained_model = CharCNN.load_from_checkpoint("lightning_logs/version_9/checkpoints/epoch=44.ckpt")
# print(pretrained_model)
pretrained_model.freeze()


def process_input(raw_text):
    text = lower(raw_text)
    text = remove_hashtags(text)
    text = remove_user_mentions(text)
    text = remove_urls(text)

    vocabulary =  "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}"
    number_of_characters = 69
    max_length = 150
    identity_mat = np.identity(number_of_characters)

    data = np.array([identity_mat[vocabulary.index(i)] for i in list(raw_text)[::-1] if i in vocabulary], dtype=np.float32)
    if len(data) > max_length:
        data = data[:max_length]
    elif 0 < len(data) < max_length:
        data = np.concatenate(
            (data, np.zeros((max_length - len(data), number_of_characters), dtype=np.float32)))
    elif len(data) == 0:
        data = np.zeros(
            (max_length, number_of_characters), dtype=np.float32)

    data = torch.Tensor(data)
    data = data.unsqueeze(0)
    return data

def predict(text):
    data = process_input(text)
    prediction = pretrained_model(data)
    probabilities = F.softmax(prediction, dim=1)
    probabilities = probabilities.detach().cpu().numpy()
    # print(probabilities)
    return probabilities


# raw_text = "some text"
# preds = predict(raw_text)
# print(preds)

# import pandas as pd
# df = pd.read_csv("test.csv")
# df['prediction'] = df.text.apply(predict)
# df.to_csv("train_preds.csv")


