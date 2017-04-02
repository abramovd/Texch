from __future__ import print_function
from texch.exceptions import NotRunYet


def already_run(instance):
    if not instance._is_run:
        raise NotRunYet("Firstly you need to call .run()")
