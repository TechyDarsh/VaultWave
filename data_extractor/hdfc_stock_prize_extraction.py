import yfinance as yf
hdfc_bank = yf.download("HDFCBANK.BO")
hdfc_close = hdfc_bank['Close']
hdfc_close.to_csv(r"C:\Users\darsh\OneDrive\Desktop\stock_datasets\HDFC_Stock_data.csv")
