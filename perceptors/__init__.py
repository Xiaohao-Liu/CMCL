from .imagebind import ImagebindPreceptor
from .languagebind import LanguageBindPreceptor
from .unibind import UniBindPreceptor


PERCEPTORS = {
    "imagebind": ImagebindPreceptor,
    "languagebind": LanguageBindPreceptor,
    "unibind": UniBindPreceptor,
}