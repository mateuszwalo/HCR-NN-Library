import yfinance as yf
import pandas as pd

"""
To jest skrypt do wyciągania danych giełdowych. 
Aby go odaplić trzeba zmienić kod spółki giełdowej i przedziały czasowe.
"""

__all__ = ['fetch_stock_data']

def fetch_stock_data(kod:str = "OTGLF",
                     start:str = "2015-01-01",
                     end:str = "2025-01-01",
                     save_path:str = './') -> None:

    data = yf.download(kod, start=start, end=end)

    data.to_csv(f"{save_path}{kod}_stock_data.csv")

    print(f"Data saved to {kod}_stock_data.csv")