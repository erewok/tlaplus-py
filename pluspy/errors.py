class PlusPyError(Exception):
    def __init__(self, descr):
        self.descr = descr

    def __str__(self):
        return f"PlusPyError({self.descr})"

