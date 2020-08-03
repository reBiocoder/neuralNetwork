import torch.nn.functional as F
import torch


def prepare_data():
    """
    构建字典
    :return:
    """
    files_content = ""
    with open('test_poetry.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            files_content += line.strip()
    counted_words = {}
    for char in files_content:
        if char in counted_words:
            counted_words[char] += 1
        else:
            counted_words[char] = 1
    chars = sorted(counted_words.items(), key=lambda x: -x[1])
    chars = list(zip(*chars))[0]
    ix_to_char = {}
    char_to_ix = {}
    for i, ch in enumerate(chars):
        char_to_ix[ch] = i
        ix_to_char[i] = ch
    return char_to_ix, ix_to_char, files_content


def get_one_hot(char: str):
    char_to_ix, ix_to_char, files_content = prepare_data()
    one_hot = F.one_hot(torch.tensor(char_to_ix[char]),
                        len(char_to_ix))
    return one_hot


if __name__ == '__main__':
    print(get_one_hot('飘'))