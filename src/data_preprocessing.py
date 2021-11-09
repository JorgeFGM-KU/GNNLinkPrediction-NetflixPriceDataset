from typing import List, Dict
import pandas as pd


def get_dataframe_from_dataset(
    rating_files: List[str]
    ) -> pd.DataFrame:
    """This function constructs a single csv file that represents all of the
    movie ratings made by the users from a series of files. The generated final
    csv file has the following columns:
        movie_id - user_id - score - date

    rating_files: all the documents where data has to be extracted from.
    dest_path: path (and file) where the final document has to be saved.
    show_progress (default False): wether to print the progress to terminal.
    progress_step (default 100): print progress every this many lines."""
    ratings: List[Dict[str, int]] = []
    for current_file_idx, rating_file in enumerate(rating_files):
        with open(rating_file) as ratings_data:
            for ln_idx, ln in enumerate(ratings_data):
                if ":" in ln:
                    movie_id = int(ln.split(":")[0])
                else:
                    splt_ln = ln.split(",")
                    user_id, score = int(splt_ln[0]), int(splt_ln[1])
                    ratings.append({
                        "user_id": user_id,
                        "movie_id": movie_id,
                        "score": score
                    })
    return pd.DataFrame(ratings)