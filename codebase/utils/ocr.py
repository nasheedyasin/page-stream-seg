from abc import ABC, abstractmethod, abstractproperty
from typing import List, Union


class BaseOcr(ABC):
    def __init__(self) -> None:
        """_summary_Initialize what your OCR Engine requires here.
        For instance if you are using GVision, maybe you need an API key.
        """
        pass

    @abstractproperty
    def engine(self) -> str:
        """
        Returns:
            str: The OCR engine being used.
        """
        pass

    @abstractproperty
    def raw_data(self) -> str:
        """
        Returns:
            str: The path to the raw data.
        """
        pass

    @abstractproperty
    def output_format(self) -> str:
        """
        Returns:
            str: The output format of the OCR engine, let this be something the
            user can set. I am thinking we support xml, json and plain old txt.
        """
        pass

    @abstractproperty
    def output_dump(self) -> str:
        """
        Returns:
            str: The path to the output directory.
        """
        pass

    @abstractmethod
    def prep_data(self):
        pass

    @abstractmethod
    def do_ocr(self):
        pass

    @abstractmethod
    def prep_output(self):
        pass


class TesseractOcr(BaseOcr):
    def __init__(self,
                 raw_data: str,
                 output_dump: str,
                 output_format: str = 'txt',
                 lang_hints: Union[List[str], str] = 'en',
                 imgs_only: bool = True) -> None:
        """
        Args:
            raw_data (str): Path to the raw files to be OCRed. Does not search
            sub-directories.
            output_dump (str): Path to save the output files.
            output_format (str, optional): Format to save outpu in.
            Defaults to 'txt'.
            lang_hints (Union[List[str], str], optional): Language hints to the
            OCR engine. Hinting occasionally improves OCR perfermoance.
            Defaults to 'en'.
            imgs_only (bool, optional): Whether to process only the images in `raw_data`. Defaults to True.
        """

        self.__raw_data = raw_data
        self.__output_dump = output_dump
        self.__output_format = output_format
        self.__lang_hints = lang_hints
        self.__imgs_only = imgs_only

    @property
    def raw_data(self) -> str:
        """
        Returns:
            str: The path to the raw data.
        """
        return self.__raw_data

    @property
    def output_format(self) -> str:
        """
        Returns:
            str: The output format of the OCR engine, let this be something the
            user can set. I am thinking we support xml, json and plain old txt.
        """
        return self.__output_format

    @property
    def output_dump(self) -> str:
        """
        Returns:
            str: The path to the output directory.
        """
        return self.__output_dump


    @property
    def lang_hints(self) -> Union[List[str], str]:
        return self.__lang_hints

    @property
    def imgs_only(self) -> bool:
        return self.__imgs_only

    def prep_data(self):
        pass

    def do_ocr(self):
        pass

    def prep_output(self):
        pass