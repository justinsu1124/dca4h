from datetime import datetime, timedelta

current_date = datetime(2025, 9, 21)
start_2022 = datetime(2022, 1, 1)
days_total = (current_date - start_2022).days

print(f'Total days from 2022-01-01 to 2025-09-21: {days_total}')
print(f'Estimated candles needed: {days_total * 6} (6 candles per day for 4H)')
print(f'API requests needed: {(days_total * 6) // 300 + 1}')

years = [2022, 2023, 2024, 2025]
for year in years:
    if year == 2025:
        start = datetime(2025, 1, 1)
        end = datetime(2025, 9, 21)
    else:
        start = datetime(year, 1, 1)
        end = datetime(year + 1, 1, 1)
    
    days = (end - start).days
    candles = days * 6
    print(f'{year}: {days} days, ~{candles} candles')
