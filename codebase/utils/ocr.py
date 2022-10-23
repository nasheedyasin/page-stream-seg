import os
import re
import pytesseract

from tqdm import tqdm
from typing import List, Tuple, Union
from abc import ABC, abstractmethod, abstractproperty


class BaseOcr(ABC):
    def __init__(self) -> None:
        """Initialize what your OCR Engine requires here.
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
                 output_dump: str = "",
                 output_format: str = 'txt',
                 lang_hints: Union[List[str], str] = 'en',
                 imgs_only: bool = True) -> None:
        """
        Args:
            raw_data (str): Path to the raw files to be OCRed. Processes all
            files in the dorectory and its sub-directories.
            output_dump (str): Path to save the output files.
            output_format (str, optional): Format to save output in.
            Defaults to 'txt'.
            lang_hints (Union[List[str], str], optional): Language hints to the
            OCR engine. Hinting occasionally improves OCR perfermoance. Ensure
            that all codes are ISO 639-1 two-digit language codes.
            Defaults to 'en'.
            imgs_only (bool, optional): Whether to process only the images in
            `raw_data`. Defaults to True.
        """

        self.__raw_data = raw_data
        self.__output_dump = output_dump
        self.__lang_hints = lang_hints
        self.__imgs_only = imgs_only

        # Make the output folder if not present
        if (len(output_dump) > 0) and (not os.path.isdir(output_dump)):
            os.mkdir(output_dump)

        # Exclude the `.` if present in the output format
        self.__output_format = re.sub(r'\.', '', output_format)

    EXTNS_SUPP: Tuple[str] = ('png', 'jpeg', 'jpg', 'bmp', 'tif', 'tiff')

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
    def engine(self) -> str:
        """
        Returns:
            str: The path to the output directory.
        """
        return f"Tesseract {pytesseract.get_tesseract_version()}"

    @property
    def lang_hints(self) -> Union[List[str], str]:
        return self.__lang_hints

    @property
    def imgs_only(self) -> bool:
        return self.__imgs_only

    def prep_data(self):
        pass

    def do_ocr(self, fpath: str) -> str:
        """
        Args:
            fpath (str): Path to the file.

        Returns:
            str: OCRed Text of the file.
        """
        if not self.__output_format.lower().endswith('txt'):
            raise NotImplementedError

        return pytesseract.image_to_string(fpath)

    def prep_output(self, text: str) -> str:
        """
        Args:
            text (str): Raw text.

        Returns:
            str: The preprocessed text ready for archival.
        """
        # Will implement this on a need basis.
        return text

    def save_text_file(self,
                       text: str,
                       fpath: str,
                       encoding: str = 'utf-8') -> None:
        with open(fpath, 'w', encoding=encoding) as file:
            file.write(text)

    def __call__(self) -> None:
        """Orchestrate the whole OCR pipeline here.
        """
        # Direct PDF processing later to be implemented.
        if not self.imgs_only: raise NotImplementedError

        # File discovery
        files_to_ocr: List[str] = list()
        for root, _, files in os.walk(self.raw_data):
            for fpath in files:
                if fpath.lower().endswith(self.EXTNS_SUPP):
                    files_to_ocr.append(os.path.join(root, fpath))

        # OCR Process
        for fpath in tqdm(files_to_ocr):
            ftext = self.prep_output(self.do_ocr(fpath))

            if len(self.output_dump) > 0:
                txt_fname: str = os.path.splitext(
                    os.path.split(fpath)[1])[0] + f'.{self.output_format}'
                txt_fname = os.path.join(self.output_dump, txt_fname)

            # Save the output in the directory of the raw input file.
            else:
                txt_fname: str = os.path.splitext(
                    fpath)[0] + f'.{self.output_format}'

            # Save the file to disk
            if self.output_format.lower().endswith('txt'):
                self.save_text_file(ftext, txt_fname)
