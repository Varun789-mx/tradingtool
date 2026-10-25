# Indian Index Options Decision Support

A Python terminal dashboard for Indian index-option decision support. It displays market context and setups, estimates option plans, simulates historical outcomes, and keeps a paper-trade journal. It does not connect to a broker, place orders, or store broker credentials.

## Requirements

- Python 3.10 or newer
- Internet access for market history and news feeds
- Windows, macOS, or Linux with a terminal that supports interactive key input

Dependencies are listed in `requirements.txt`: `yfinance`, `pandas`, `numpy`, `rich`, `plotly`, `feedparser`, and `tzdata`.

## Setup

Run from the project directory. On Windows, use `py` instead of `python` if that is how your Python installation is invoked.

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS or Linux, activate it with:

```sh
source .venv/bin/activate
```

Install the dependencies and launch the dashboard:

```sh
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python trade.py
```

If `config.json` is missing, the first run asks for capital and maximum loss percentage per trade, then creates the configuration. This project includes a `config.json`; edit it directly to change risk limits, index settings, holidays, costs, events, or news queries.

## Dashboard Commands

```sh
python trade.py --help
python trade.py
python trade.py --replay 2026-10-21
python trade.py --replay 2026-10-21 --speed 20
```

`--replay` opens the specified historical session for the selected index. `--speed` sets the simulated clock speed (default: `20`). Replay uses cached history when available and otherwise attempts to download it. News is disabled during replay to avoid lookahead.

### Dashboard Keys

| Key | Action |
| --- | --- |
| `n` | Select NIFTY |
| `b` | Select BANKNIFTY |
| `s` | Select SENSEX |
| `t` | Recompute the Time Machine analysis |
| `c` | Generate and open the live chart in a browser |
| `p` | Enter an observed option premium to calibrate implied volatility |
| `e` | Enter the premium at which you paper-entered; records a paper position when a valid plan is available |
| `x` | Enter the exit premium to close the paper position and journal it |
| `r` | Enter a replay date in `YYYY-MM-DD` format |
| `h` | Toggle on-screen help |
| `q` | Quit |

For prompts, press `Enter` to submit, `Esc` to cancel, and `Backspace` to edit. Premiums must be positive numbers. These controls only update the local paper journal; they never place a real order.

## Standalone Stock Scanner

`india.py` is a separate opening-range breakout scanner/planner for a fixed watchlist of NSE stocks. It is not the interactive index-options dashboard.

```sh
python india.py --help
python india.py scan
python india.py scan --r 2
python india.py plan RELIANCE
python india.py plan RELIANCE --r 2 --maxloss 1000
```

- `scan` checks the watchlist and prints available opening-range signals. `--r` sets the risk/reward multiple (default: `1.5`).
- `plan SYMBOL` prints an entry, stop, target, quantity, and historical simulation for one watchlist symbol, such as `RELIANCE`, `INFY`, or `TCS`.
- `plan` accepts `--maxloss` to set the maximum rupee loss used for quantity sizing (default: `500`).

## Project Files

- `trade.py`: interactive index-options dashboard and command-line replay options.
- `india.py`: standalone stock opening-range scanner and planner.
- `config.json`: capital, risk limits, index contracts, estimated costs, holidays, event dates, and news queries.
- `core/config.py`: configuration defaults, loading, and first-run setup.
- `core/data.py`: market data retrieval, caching, session handling, and indicators.
- `core/signals.py` and `core/bias.py`: setup evaluation and market-bias calculations.
- `core/options.py`: option pricing, implied volatility, sizing, expiry, and cost calculations.
- `core/timemachine.py`: historical setup statistics and simulation/replay support.
- `core/news.py`: news collection and sentiment scoring.
- `core/journal.py`: paper positions and CSV journal operations.
- `core/chart.py`: interactive HTML chart generation.
- `data/sentiment_lexicon.json`: editable sentiment terms and news tags.
- `journal.csv`: local paper-trade records.
- `cache/`, `charts/`, and `logs/`: downloaded data, generated charts, and application logs.

## Risk Notice

This software is not financial advice, a forecast, or an order router. Setups, news scores, modeled premiums, and Time Machine probabilities are estimates based on limited and imperfect data; historical simulations do not predict future results. Most option buyers lose money. Verify current NSE/BSE lot sizes, expiry schedules, holidays, taxes, and transaction costs before relying on displayed calculations.
