import pandas as pd 
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",default=None,type=str,help="Path to the root of the wsj0 dataset. Use \"\" to reset to original metadata.")
    args = parser.parse_args()
    root = args.root.rstrip("/") 

    # Change root path of metadata
    for partition_name in ["tr", "cv", "tt"]:
        for min_n_speaker, max_n_speaker in [(2, 2), (2, 3), (2, 4), (2, 5), (3, 3), (4, 4), (5, 5)]:
            csv_path = f"./data/var_num_speakers/wav8k/min/metadata/{partition_name}/{min_n_speaker}_{max_n_speaker}.csv"
            df = pd.read_csv(csv_path, index_col=0)
            path_start = f"{max_n_speaker}speakers"
            new_path_start = f"{root}/{path_start}" if root else path_start
            df["mix"] = df["mix"].apply(lambda x: f"{new_path_start}{x.split(path_start)[1]}" if type(x)==str and  path_start in x else float("nan"))
            for n_speaker in range(1, max_n_speaker+1):
                df[f"s{n_speaker}"] = df[f"s{n_speaker}"].apply(lambda x: f"{new_path_start}{x.split(path_start)[1]}" if type(x)==str and path_start in x else float("nan"))
            df.to_csv(csv_path)