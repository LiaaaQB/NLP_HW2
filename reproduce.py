import train_tokenizer
import train_tokenizer_2
import train_tokenizer_3


def merge_txt_files(file1_path, file2_path, output_path=None):
    """
    Appends the contents of file2 to file1 and writes the result to output_path.
    If output_path is None, overwrites file1 with the merged content.
    """
    with open(file1_path, 'r', encoding='utf-8') as f1:
        content1 = f1.read()

    with open(file2_path, 'r', encoding='utf-8') as f2:
        content2 = f2.read()

    merged_content = content1 + content2

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as out:
            out.write(merged_content)

if __name__ == "__main__":
    # reproduce the tokenizers 1 + 2
    train_tokenizer.train_tokenizer('data/domain_1_train.txt', 'tokenizers', 600)
    train_tokenizer_2.train_tokenizer('data/domain_2_train.txt', 'tokenizers', 4000)

    # make domain_3_train for 3rd tokenizer:
    merge_txt_files('data/domain_1_train.txt', 'data/domain_2_train.txt', 'data/domain_3_train.txt')

    # reproduce tokenizer 3
    train_tokenizer_3.train_tokenizer('data/domain_3_train.txt', 'tokenizers', 5000)
