class Error(Exception):
    pass

class InvalidMove(Error):
    """ Raised when the user enters an invalid move """