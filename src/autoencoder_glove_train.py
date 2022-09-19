import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

from src.autoencoder_model import run_experiments, process_experiments, set_seeds
from src.paths import GLOVE_DATA_BASE_PATH, GLOVE_EXPERIMENT_PATH


def load_glove_model(File, cache=False):
    print("Loading Glove Model...")
    file_name = f"{File}.pkl"
    if cache:
        print("Loading from cache.")
        glove_model = pd.read_pickle(file_name)
    else:
        with open(File, encoding="utf-8") as f:
            col_names = [
                "dim" + str(dim[0]) for dim in enumerate(f.readline().split()[1:])
            ]

            """
            Need to stop reading the file early if we want to load into memory
            -1 = read the whole file
            """
            num_words_to_add = -1
            rows, words = [], []
            for cnt, line in enumerate(f):
                split_line = line.split()
                word = split_line[0]
                embedding = np.array(split_line[1:], dtype=np.float64)
                rows.append(list(embedding))
                words.append(word)

                if cnt % 100 == 0:
                    print("Reading word: " + str(cnt))
                if cnt == num_words_to_add:
                    break

            print("Loading into DataFrame (this may take a while)")
            glove_model = pd.DataFrame(rows, columns=col_names)
            glove_model = glove_model.set_index(pd.Index(words))
            scaler = MinMaxScaler()
            glove_model[col_names] = scaler.fit_transform(glove_model[col_names])
            print(f"Saving {file_name} to cache")
            glove_model.to_pickle(file_name)

    print(f"{len(glove_model)} words loaded!")
    return glove_model


def run_glove(seed, num_epochs, lr, middle_node_size):
    set_seeds(seed)
    file_name = "glove.twitter.27B.100d"
    file_extension = ".txt"
    glove_twitter_data = load_glove_model(
        GLOVE_DATA_BASE_PATH + file_name + file_extension, cache=True
    )
    run_type = "no_batching"

    sample_size = 10000

    shuffle_glove_twitter_data = shuffle(glove_twitter_data)
    train_set, test_set = (
        shuffle_glove_twitter_data[:sample_size],
        shuffle_glove_twitter_data[sample_size + 1 : (sample_size + 1) * 2],
    )

    run_histories = run_experiments(
        train_set,
        test_set,
        run_type=run_type,
        experiment_path=GLOVE_EXPERIMENT_PATH,
        num_epochs=num_epochs,
        lr=lr,
        middle_node_size=middle_node_size,
    )
    process_experiments(name=file_name, experiment_path=GLOVE_EXPERIMENT_PATH)
    return run_histories


if __name__ == "__main__":
    run_glove()
