def print_progress(cur, tot, pre="", fill="=", close=["[", "]"], length=20, opt=None, sep=" - ", add_newline=False, notebook=False, old_version=False):
    """
    Creates a progress bar that updates itself given the current value and max value.

    Parameters
    ----------
    cur : int
        Current progress.
    tot : int
        Max. progress.
    fill : str
        What to fill the progress bar with.
    close : list(str)
        Closing ends of the progress bar.
    length : int
        Length of the progress bar.
    opt : list(str)
        List of strings to be printed after the progress bar, separated by `sep`.
    sep : str
        Separate `opt` with sep.
    add_newline : bool
        Forces a newline at the end. A newline is always inserted when `cur` == `tot`.

    Examples
    --------
    >>> print_progress(76, 100, opt=[
            "time: {:.3f}".format(15.123),
            "loss: {:.3f}".format(2.121),
            "accuracy: {:.3f}".format(0.960)
        ])
     76/100 [===============     ] - time: 15.123 - loss: 2.121 - accuracy: 0.960

    >>> print_progress(76, 100, length=50, fill="-", close=["(", ")"], sep=", ", opt=[
            "time: {:.3f}".format(15.123),
            "loss: {:.3f}".format(2.121),
            "accuracy: {:.3f}".format(0.960)
        ])
     76/100 (----------------------        ), time: 15.123, loss: 2.121, accuracy: 0.960
    """

    ratio = cur / tot
    perc = int(length * ratio)
    char_size = len(str(tot))

    if opt is not None and len(opt) > 0:
        opt = sep + sep.join(opt)
    else:
        opt = ""

    end = '' if not old_version else ('\r' if notebook else '\x1b[2k\r')
    start = '\r' if not old_version else ''
    print(f"{start}{pre}{cur:>{char_size}}/{tot} {close[0]}{fill * perc:{length}}{close[1]}{opt}", end=end, flush=True)

    if cur == tot or add_newline:
        print()  # print a new line
