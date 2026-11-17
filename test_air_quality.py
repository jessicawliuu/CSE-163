"""
Jessica Liu
CSE 163 AC

Tests the functions in air_quality.py on test_data.csv, a small file in
the same format as the EPA daily PM2.5 files. The expected values were
worked out by hand from the ten rows in that file.
"""

import pandas as pd

import air_quality

TEST_FILE = 'test_data.csv'


def test_load_king_county() -> None:
    """
    Tests that only King County, Washington rows are kept, so the Pierce
    County row and the King County, Texas row are dropped.
    """
    raw = air_quality.load_king_county([TEST_FILE])
    assert 8 == len(raw)
    assert {'King'} == set(raw['County Name'])
    assert {'Washington'} == set(raw['State Name'])


def test_clean_readings() -> None:
    """
    Tests that the '1 HOUR' row and the 'Excluded' row are dropped and
    that the date columns are added.
    """
    raw = air_quality.load_king_county([TEST_FILE])
    readings = air_quality.clean_readings(raw)
    assert 6 == len(readings)
    assert '1 HOUR' not in set(readings['Sample Duration'])
    assert 300.0 not in set(readings['Arithmetic Mean'])
    assert [2014, 2014, 2014, 2020, 2020, 2020] == list(readings['year'])
    assert [8, 8, 1, 9, 9, 7] == list(readings['month'])


def test_daily_county() -> None:
    """
    Tests the average and maximum PM2.5 for each day across monitors.
    """
    raw = air_quality.load_king_county([TEST_FILE])
    daily = air_quality.daily_county(air_quality.clean_readings(raw))
    assert 4 == len(daily)
    assert 30.0 == daily.loc[pd.Timestamp('2014-08-01'), 'mean_pm25']
    assert 40.0 == daily.loc[pd.Timestamp('2014-08-01'), 'max_pm25']
    assert 140.0 == daily.loc[pd.Timestamp('2020-09-12'), 'mean_pm25']
    assert 150.0 == daily.loc[pd.Timestamp('2020-09-12'), 'max_pm25']


def test_period_label() -> None:
    """
    Tests the years on each side of the split between periods.
    """
    assert '2000-2014' == air_quality.period_label(2000)
    assert '2000-2014' == air_quality.period_label(2014)
    assert '2015-2025' == air_quality.period_label(2015)
    assert '2015-2025' == air_quality.period_label(2025)


def test_unhealthy_days_per_year() -> None:
    """
    Tests that a day counts as unhealthy when its highest monitor is at
    or above 35.5 ug/m3, even if the county average is below it.
    """
    raw = air_quality.load_king_county([TEST_FILE])
    daily = air_quality.daily_county(air_quality.clean_readings(raw))
    counts = air_quality.unhealthy_days_per_year(daily)
    assert {2014: 1, 2020: 1} == counts.to_dict()


def test_unhealthy_days_by_month() -> None:
    """
    Tests the unhealthy day counts for each month in each period.
    """
    raw = air_quality.load_king_county([TEST_FILE])
    daily = air_quality.daily_county(air_quality.clean_readings(raw))
    by_month = air_quality.unhealthy_days_by_month(daily)
    assert ['2000-2014', '2015-2025'] == list(by_month['period'])
    assert [8, 9] == list(by_month['month'])
    assert [1, 1] == list(by_month['unhealthy_days'])


def test_missing_days() -> None:
    """
    Tests the count of days with no readings between the first day
    (2014-01-15) and last day (2020-09-12) in the test file, which is
    2433 calendar days with only 4 of them having readings.
    """
    raw = air_quality.load_king_county([TEST_FILE])
    daily = air_quality.daily_county(air_quality.clean_readings(raw))
    assert 2429 == air_quality.missing_days(daily)


def test_monthly_means() -> None:
    """
    Tests the average daily PM2.5 for each month in each period.
    """
    raw = air_quality.load_king_county([TEST_FILE])
    daily = air_quality.daily_county(air_quality.clean_readings(raw))
    means = air_quality.monthly_means(daily)
    result = {}
    for i in range(len(means)):
        row = means.iloc[i]
        result[(row['period'], row['month'])] = row['mean_pm25']
    expected = {('2000-2014', 1): 10.0, ('2000-2014', 8): 30.0,
                ('2015-2025', 7): 5.0, ('2015-2025', 9): 140.0}
    assert expected == result


def test_summer_days() -> None:
    """
    Tests that only July to September days are kept.
    """
    raw = air_quality.load_king_county([TEST_FILE])
    daily = air_quality.daily_county(air_quality.clean_readings(raw))
    summer = air_quality.summer_days(daily)
    assert 3 == len(summer)
    assert ['2000-2014', '2015-2025', '2015-2025'] == list(summer['period'])


def test_missing_summary() -> None:
    """
    Tests that the empty 'Event Type' column is reported as 100% missing
    and that columns with no missing values are left out.
    """
    raw = air_quality.load_king_county([TEST_FILE])
    readings = air_quality.clean_readings(raw)
    missing = air_quality.missing_summary(readings)
    assert ['Event Type'] == list(missing.index)
    assert 100.0 == missing.loc['Event Type', 'percent']


def test_seven_number_summary() -> None:
    """
    Tests the summary of a small Series worked out by hand.
    """
    summary = air_quality.seven_number_summary(pd.Series([1, 2, 3, 4, 5]))
    assert 3.0 == summary['mean']
    assert 1.0 == summary['min']
    assert 2.0 == summary['25%']
    assert 3.0 == summary['50%']
    assert 4.0 == summary['75%']
    assert 5.0 == summary['max']


def main():
    test_load_king_county()
    test_clean_readings()
    test_daily_county()
    test_period_label()
    test_unhealthy_days_per_year()
    test_unhealthy_days_by_month()
    test_missing_days()
    test_monthly_means()
    test_summer_days()
    test_missing_summary()
    test_seven_number_summary()
    print('All tests passed!')


if __name__ == '__main__':
    main()
