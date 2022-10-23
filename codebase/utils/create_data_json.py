import re
import json
import argparse

from pathlib import Path

from natsort import natsorted
from typing import Any, Dict, List, Tuple
from sklearn.model_selection import train_test_split

# What kind of images to look for
EXTNS_SUPP: Tuple[str] = ('.png', '.jpeg', '.jpg', '.bmp', '.tif', '.tiff')


def main(img_path: str, txt_path: str, output_path: str,
         train_data_frac: str) -> None:
    """
    Args:
    img_path (str): Path to the image files.
    txt_path (str): Path to the text files.
    output_path (str): Where to save the data jsons.
    train_data_frac (float): Fraction of the data to have as
    train data.

    Note:
    We ignore text and image files that don't have their counter-parts,
    wiz. image and text files respectively.
    BIG ASSUMPTION: We assume that the page number is indicated in the
    file name. This number may be indicated as an integer positioned
    after a `_` or a `-` at the tail of the filename.
    """
    # Document discovery
    documents: List[Dict[str, Any]] = list()

    # Gather all image files only.
    img_files: List[Path] = [
        p.resolve() for p in Path(img_path).glob("**/*")
        if p.suffix in EXTNS_SUPP
    ]

    img_files = natsorted(img_files)

    # The way the loops are structured:
    # Outer Loop creates new documents
    # Inner loop extends existing documents.
    curr_idx = 0
    while curr_idx < len(img_files):
        base_fname = img_files[curr_idx].stem

        text_file: Path = Path(txt_path) / f"{base_fname}.txt"

        # Check if there is an accompanying text file for this page.
        if not text_file.is_file(): continue

        doc_name: str = re.split(r"[-_]", base_fname, maxsplit=1)[0]
        # Start a new document.
        document: Dict[str, Any] = {
            "docName": doc_name,
            "pages": [{
                "img": img_files[curr_idx].parts[-1],
                "txt": text_file.parts[-1]
            }]
        }

        # Advance the curr_idx by 1
        curr_idx += 1

        while curr_idx < len(img_files):
            nx_base_fname = img_files[curr_idx].stem

            # Check if they have the same `doc_name`
            nx_doc_name: str = re.split(r"[-_]", nx_base_fname, maxsplit=1)[0]

            # Break if not the same
            if nx_doc_name != doc_name: break

            nx_text_file: Path = Path(txt_path) / f"{nx_base_fname}.txt"

            # Check if there is an accompanying text file for this page.
            if not nx_text_file.is_file(): continue

            # Append to the pages of the current document
            document['pages'].append({
                "img": img_files[curr_idx].parts[-1],
                "txt": nx_text_file.parts[-1]
            })

            # Advance the curr_idx by 1
            curr_idx += 1

        # Append to the list of documents
        documents.append(document)

    # Constructing the data json
    data_json: Dict[str, Any] = {"imgPath": img_path, "txtPath": txt_path}

    # Splitting the documents into a train and test set.
    doc_train, doc_test = train_test_split(documents,
                                           train_size=train_data_frac,
                                           random_state=88)

    # Save the train data
    data_json.update(documents=doc_train)
    with open(str(Path(output_path) / "train_data.json"), 'w',
              encoding='utf-8') as data_file:
        data_file.wrtie(json.dumps(data_json))

    # Save the test data
    data_json.update(documents=doc_test)
    with open(str(Path(output_path) / "test_data.json"), 'w',
              encoding='utf-8') as data_file:
        data_file.wrtie(json.dumps(data_json))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generates the Data Json'
                                     'for SDS model training.')

    parser.add_argument(
        '--img_path',
        type=str,
        help='Path to document images. Ensure a flat directory '
        'structure.')
    parser.add_argument(
        '--txt_path',
        type=str,
        help='Path to document txt. Ensure the file name is the same and a '
        'flat directory structure.')

    parser.add_argument('--output_path',
                        type=str,
                        help='Where to save the data jsons.')

    parser.add_argument('--train_data_frac',
                        type=float,
                        default=0.85,
                        help='Fraction of the data to have as train data. '
                        'Defaults to 0.85')

    args = parser.parse_args()

    main(args.img_path, args.txt_path, args.output_path, args.train_data_frac)
