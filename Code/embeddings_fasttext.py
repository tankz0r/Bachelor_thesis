import subprocess
from pygoose import kg
from config import *


def create_vocabulary():
    vocab = kg.io.load_lines(VOCAB_FILE)
    with open(OUTPUT_FILE, 'w') as f:
        print('%s %s' %(len(vocab), EMBEDDING_DIM), file=f)

    with open(VOCAB_FILE) as f_vocab:
        with open(OUTPUT_FILE, 'a') as f_output:
            subprocess.run(
                [FASTTEXT_EXECUTABLE, 'print-word-vectors', PRETRAINED_MODEL_FILE],
                stdin=f_vocab,
                stdout=f_output,
            )
