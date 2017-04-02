class NewsterException(Exception):
    pass


class InputDataException(NewsterException):
    pass


class PreprocessingException(NewsterException):
    pass


class NotRunYet(NewsterException):
    pass