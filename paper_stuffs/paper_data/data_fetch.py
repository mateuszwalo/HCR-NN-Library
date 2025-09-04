import yfinance as yf
import pandas as pd
from enum import Enum

"""
To jest skrypt do wyciągania danych giełdowych. 
Aby go odaplić trzeba zmienić kod spółki giełdowej i przedziały czasowe.
"""

__all__ = ['fetch_stock_data']

class stock_data(Enum):
    code = 'OTGLF'
    start = "2015-01-01"
    end = "2025-01-01"

def fetch_stock_data(code:str = stock_data.code,
                     start:str = stock_data.start,
                     end:str = stock_data.end,
                     save_path:str = './') -> None:

    data = yf.download(code, start=start, end=end)

    data.to_csv(f"{save_path}{code}_stock_data.csv")

    print(f"Data saved to {code}_stock_data.csv")