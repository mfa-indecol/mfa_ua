import scipy.stats as sts


class RightSkewedLognorm:
    """
    Works as a class that mirrors a given lognormal
    distribution at the y axis. To ensure compatibility with
    the UA module, rvs, pdf, cdf, ppf, mean are implemented.
    """

    def __init__(self):
        pass

    def rvs(self, s: float = 1, scale: float = 1, loc: float = 0, size: int = 1):
        return -sts.lognorm.rvs(s=s, scale=scale, loc=loc, size=size)

    def pdf(self, x, s: float, scale: float, loc: float):
        return sts.lognorm.pdf(x=-x, s=s, scale=scale, loc=loc)

    def cdf(self, x, s: float, scale: float, loc: float):
        if hasattr(x, "__iter__"):
            oj_cdf = sts.lognorm.cdf(x=-x, s=s, scale=scale, loc=loc)
            return np.subtract(np.ones(len(x)), oj_cdf)
        else:
            return 1 - sts.lognorm.cdf(x=-x, s=s, scale=scale, loc=loc)

    def ppf(self, x, s: float, scale: float, loc: float):
        if hasattr(x, "__iter__"):
            return -sts.lognorm.ppf(
                np.subtract(np.ones(len(x)), x), s=s, scale=scale, loc=loc
            )
        else:  # if a single x:
            return -sts.lognorm.ppf(1 - x, s=s, scale=scale, loc=loc)

    def mean(self, s: float, scale: float, loc: float):
        return -sts.lognorm.mean(s=s, scale=scale, loc=loc)
