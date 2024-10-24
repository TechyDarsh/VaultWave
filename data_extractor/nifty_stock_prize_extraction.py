import yfinance as yf


nifty_50 = yf.download("^NSEI")


nifty_close = nifty_50['Close']
nifty_close.to_csv(r"C:\Users\darsh\OneDrive\Desktop\nifty_data.csv")
