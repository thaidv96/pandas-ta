# -*- coding: utf-8 -*-
import numpy as np
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
        """Chande Momentum Oscillator (CMO) Patch"""
        # Ensure source is float64
        source = source.astype("float64")
        mom = source.diff(d)
        positive = mom.where(mom > 0, 0.0)
        negative = (-mom).where(mom < 0, 0.0)
        pos_sum = positive.rolling(n, min_periods=n).sum()
        neg_sum = negative.rolling(n, min_periods=n).sum()
        total = pos_sum + neg_sum
        # Avoid division by zero
        cmo_result = np.where(total != 0, (pos_sum - neg_sum) / total, 0.0)
        return Series(cmo_result, index=source.index, dtype="float64")

    # Ensure close is float64 from the start
    close = close.astype("float64")

    # Calculate CMO and get absolute values
    cmo_vals = _cmo(close, length, drift)
    abs_cmo = np.abs(cmo_vals.values)  # Work with numpy array to avoid pandas issues

    # Initialize result array with NaN
    result = np.full(len(close), np.nan, dtype=np.float64)
    close_vals = close.values  # Work with numpy arrays

    # VIDYA calculation parameters
    alpha = 2.0 / (length + 1.0)

    # Set initial value at length-1 position
    if len(close_vals) > length - 1:
        result[length - 1] = close_vals[length - 1]

    # Calculate VIDYA values
    for i in range(length, len(close_vals)):
        if not np.isnan(abs_cmo[i]) and not np.isnan(result[i - 1]):
            vi = alpha * abs_cmo[i]
            result[i] = vi * close_vals[i] + result[i - 1] * (1.0 - vi)

    # Convert back to Series
    vidya = Series(result, index=close.index, dtype="float64")
    vidya.name = f"VIDYA_{length}"
    vidya.category = "overlap"

    # Offset
    if offset != 0:
        vidya = vidya.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        vidya.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        vidya.fillna(method=kwargs["fill_method"], inplace=True)

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
    VI = alpha * |CMO|
    VIDYA[i] = VI * close[i] + VIDYA[i-1] * (1 - VI)

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
