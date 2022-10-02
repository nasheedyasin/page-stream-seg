import json

from typing import Tuple


class Document:
    """The Document base class. Casting your document indices as
    as a variable of type `Document` can save a bunch of time when
    you want to equate, sort and index documents.
    """
    def __init__(self, dossier_name: str, start_idx: int, end_idx: int) -> None:
        """
        Args:
            dossier_name (str): dossier_name of the the larger dossier containing
            the document. This would generally be set to something indicative such as
            `doss1.pdf`. But feel free, no restrictions imposed.
            start_idx (int): Start index of the document in the larger dossier.
            end_idx (int): End index of the document in the larger dossier.
            Note: 
                Both the `start_idx` and `end_idx` are `1-based` indices. i.e.
                the first page of the larger dossier would be index 1.
                
                Ensure that the end_idx >= start_idx.
        """
        self._dossier_name = dossier_name

        if not (end_idx >= start_idx):
            raise ValueError(
                "Ensure that the end_idx >= start_idx. Values entered were "
                f"{start_idx} and {end_idx} respectively.")

        self.__start_idx = start_idx
        self.__end_idx = end_idx
        self.__page_count = 1 + end_idx - start_idx

    @property
    def dossier_name(self) -> str:
        return self._dossier_name

    @property
    def start_idx(self) -> int:
        return self.__start_idx

    @property
    def end_idx(self) -> int:
        return self.__end_idx

    def __len__(self) -> int:
        """Returns the page count of the document.
        """
        return self.__page_count

    def __str__(self) -> str:
        """Returns a `Json`-ised string version of the document meta information.
        """
        return json.dumps({
            'dossierName': self.dossier_name,
            'startIdx': self.start_idx,
            'endIdx': self.end_idx
        })

    def __key__(self) -> Tuple[str, int, int]:
        """Returns the `key` information of the object.
        """
        return self.dossier_name, self.start_idx, self.end_idx

    def __call__(self) -> Tuple[str, int, int]:
        """Returns the `key` information of the object. Basically what the
        `__key__` func would do to an object of this class. I have implemented
        this for completeness and user friendliness.
        """
        return self.__key__()

    def __hash__(self) -> int:
        """Returns a unique hash for the object.
        """
        return hash(self())

    def __eq__(self, __o: object) -> bool:
        try:
            return self() == __o()
        except:
            TypeError('Unsupported operand type(s) for `==`: '
                      f'{type(self)} and {type(__o)}')

    def __gt__(self, __o: object):
        return self() > __o()
    
    def __ge__(self, __o: object):
        return self() >= __o()
