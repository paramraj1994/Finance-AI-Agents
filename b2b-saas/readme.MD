# B2B SaaS Financial Model (Streamlit)

A driver-based B2B SaaS model with:
- Bear / Base / Bull scenarios
- Year-wise editable assumptions per scenario
- Dashboard with SaaS charts
- Full financial statements (Drivers, P&L, Cash Flow)
- DCF valuation with Gordon Growth or Exit Multiple (EV/Revenue)
- Export to Excel (multi-sheet)
- Save / load assumptions (JSON)

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate  # mac/linux
# .venv\Scripts\activate   # windows

pip install -r requirements.txt
streamlit run app.py
