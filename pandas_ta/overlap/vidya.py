# -*- coding: utf-8 -*-
from numpy import nan as npNaN
from pandas import Series
from pandas_ta.utils import get_drift, get_offset, verify_series


def vidya(close, length=None, drift=None, offset=None, **kwargs):
    """Indicator: Variable Index Dynamic Average (VIDYA)"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 14
    close = verify_series(close, length)
    drift = get_drift(drift)
    offset = get_offset(offset)

    if close is None:
        return

    def _cmo(source: Series, n: int, d: int):
        """Chande Momentum Oscillator (CMO) Patch
        For some reason: from pandas_ta.momentum import cmo causes
        pandas_ta.momentum.coppock to not be able to import it's
        wma like from pandas_ta.overlap import wma?
        Weird Circular TypeError!?!
        """
        mom = source.diff(d)
        positive = mom.copy().clip(lower=0)
        negative = mom.copy().clip(upper=0).abs()
        pos_sum = positive.rolling(n).sum()
        neg_sum = negative.rolling(n).sum()
        return (pos_sum - neg_sum) / (pos_sum + neg_sum)

    # Calculate Result
    m = close.size
    alpha = 2.0 / (length + 1.0)  # Ensure float division
    abs_cmo = _cmo(close, length, drift).abs()

    # Create a copy of close as float64 and fill with NaN initially
    vidya = close.astype("float64").copy()
    vidya.iloc[:] = npNaN

    # Set the first valid value (at length-1 index) to the close price
    start_idx = length - 1
    vidya.iloc[start_idx] = close.iloc[start_idx]

    # Vectorized calculation where possible, but keep loop for dependency
    for i in range(length, m):
        if not (abs_cmo.iloc[i] != abs_cmo.iloc[i]):  # Check for NaN
            factor = alpha * abs_cmo.iloc[i]
            vidya.iloc[i] = factor * close.iloc[i] + vidya.iloc[i - 1] * (1.0 - factor)

    # Offset
    if offset != 0:
        vidya = vidya.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        vidya.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        vidya.fillna(method=kwargs["fill_method"], inplace=True)

    # Name & Category
    vidya.name = f"VIDYA_{length}"
    vidya.category = "overlap"

    return vidya


vidya.__doc__ = """Variable Index Dynamic Average (VIDYA)

Variable Index Dynamic Average (VIDYA) was developed by Tushar Chande. It is
similar to an Exponential Moving Average but it has a dynamically adjusted
lookback period dependent on relative price volatility as measured by Chande
Momentum Oscillator (CMO). When volatility is high, VIDYA reacts faster to
price changes. It is often used as moving average or trend identifier.

Sources:
    https://www.tradingview.com/script/hdrf0fXV-Variable-Index-Dynamic-Average-VIDYA/
    https://www.perfecttrendsystem.com/blog_mt4_2/en/vidya-indicator-for-mt4

Calculation:
    Default Inputs:
        length=14, drift=1
    
    CMO = Chande Momentum Oscillator
    alpha = 2 / (length + 1)
    VIDYA[i] = alpha * |CMO[i]| * close[i] + VIDYA[i-1] * (1 - alpha * |CMO[i]|)

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 14
    drift (int): The difference period for CMO calculation. Default: 1
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
