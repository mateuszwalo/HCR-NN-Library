import yfinance as yf

"""
To jest skrypt do wyciągania danych giełdowych. 
Aby go odaplić trzeba zmienić kod spółki giełdowej i przedziały czasowe.
"""

kod:str = "OTGLF"
start:str = "2015-01-01"
end:str = "2025-01-01"

data = yf.download(kod, start=start, end=end)

data.to_csv("CDPR_stock_data.csv")

print("Data saved to CDPR_stock_data.csv")