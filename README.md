# Index options decision support (India)

`pip install -r requirements.txt` then `python trade.py`. First run asks capital and max loss % per trade.

Keys: n Nifty, b BankNifty, s Sensex, t Time Machine, c chart, p live premium (IV), e I entered, x exit, r replay date, h help, q quit.

Replay: `python trade.py --replay YYYY-MM-DD --speed 20`. Paper journal only; never places orders or stores broker credentials.

Not financial advice and not a forecast: setups, news score, and Time Machine probabilities are uncertain historical samples.

Most option buyers lose money. Verify lot sizes (NSE circular 3 Oct 2026), holidays, and charges in `config.json`.
