"""
Jessica Liu
CSE 163 AC

Exploratory data analysis of daily PM2.5 air quality readings in King
County, Washington from 2000 to 2025, using the EPA AirData daily summary
files. Loads and cleans the yearly files, prints the size, missing values
and summaries of the variables of interest, saves plots of unhealthy air
days per year, average PM2.5 by month, and summer PM2.5 distributions,
and checks the assumptions of the statistical tests planned for the
final report.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

DATA_FOLDER = 'data'
CLEAN_FILE = 'king_county_pm25.csv'
UNHEALTHY_PM25 = 35.5
SPLIT_YEAR = 2015
SUMMER_MONTHS = [7, 8, 9]


def load_king_county(paths: list[str]) -> pd.DataFrame:
    """
    Takes a list of paths to EPA daily PM2.5 summary files (csv or zipped
    csv) and returns one DataFrame with only the rows for King County,
    Washington from all of the files combined.
    """
    frames = []
    for path in paths:
        df = pd.read_csv(path, low_memory=False)
        is_king = ((df['State Name'] == 'Washington') &
                   (df['County Name'] == 'King'))
        frames.append(df[is_king])
    return pd.concat(frames, ignore_index=True)


def clean_readings(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the raw King County rows and returns one 24-hour reading per
    monitor per day. Drops the '1 HOUR' summary rows, which repeat the
    same monitor and day without an AQI, and drops any rows flagged as
    'Excluded' events. Adds a datetime 'date' column and 'year' and
    'month' columns.
    """
    is_daily = raw['Sample Duration'] != '1 HOUR'
    not_excluded = raw['Event Type'] != 'Excluded'
    readings = raw[is_daily & not_excluded]
    readings = readings.drop_duplicates(subset=['Site Num', 'POC',
                                                'Date Local'])
    readings = readings.copy()
    readings['date'] = pd.to_datetime(readings['Date Local'])
    readings['year'] = readings['date'].dt.year
    readings['month'] = readings['date'].dt.month
    return readings


def daily_county(readings: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the cleaned readings and returns a DataFrame with one row per
    day for all of King County. 'mean_pm25' is the average PM2.5 across
    all monitors that day and 'max_pm25' is the highest monitor's PM2.5.
    Also keeps the 'year' and 'month' of each day.
    """
    by_date = readings.groupby('date')['Arithmetic Mean']
    daily = pd.DataFrame({'mean_pm25': by_date.mean(),
                          'max_pm25': by_date.max()})
    daily['year'] = daily.index.year
    daily['month'] = daily.index.month
    return daily


def period_label(year: int) -> str:
    """
    Returns which period of the study the given year falls in:
    '2000-2014' for years before SPLIT_YEAR and '2015-2025' otherwise.
    """
    if year < SPLIT_YEAR:
        return '2000-2014'
    return '2015-2025'


def unhealthy_days_per_year(daily: pd.DataFrame) -> pd.Series:
    """
    Takes the daily county DataFrame and returns a Series with the number
    of days each year where at least one monitor's 24-hour PM2.5 was at
    or above UNHEALTHY_PM25 (AQI over 100, 'Unhealthy for Sensitive
    Groups'). Years with no such days are included with a count of 0.
    """
    unhealthy = daily[daily['max_pm25'] >= UNHEALTHY_PM25]
    counts = unhealthy.groupby('year').size()
    all_years = sorted(daily['year'].unique())
    return counts.reindex(all_years, fill_value=0)


def unhealthy_days_by_month(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the daily county DataFrame and returns a DataFrame with the
    number of unhealthy days (see unhealthy_days_per_year) in each month
    of the year for each period, with columns 'period', 'month' and
    'unhealthy_days'. Months with no unhealthy days are left out.
    """
    unhealthy = daily[daily['max_pm25'] >= UNHEALTHY_PM25].copy()
    unhealthy['period'] = unhealthy['year'].apply(period_label)
    counts = unhealthy.groupby(['period', 'month']).size()
    counts.name = 'unhealthy_days'
    return counts.reset_index()


def missing_days(daily: pd.DataFrame) -> int:
    """
    Takes the daily county DataFrame and returns how many calendar days
    between its first and last day have no reading from any monitor.
    """
    all_days = pd.date_range(daily.index.min(), daily.index.max())
    return len(all_days) - len(daily)


def monthly_means(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the daily county DataFrame and returns a DataFrame with the
    average daily PM2.5 for each month of the year in each period
    (see period_label), with columns 'period', 'month' and 'mean_pm25'.
    """
    with_period = daily.copy()
    with_period['period'] = with_period['year'].apply(period_label)
    means = with_period.groupby(['period', 'month'])['mean_pm25'].mean()
    return means.reset_index()


def summer_days(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the daily county DataFrame and returns only the days in
    SUMMER_MONTHS (July to September), with a 'period' column added.
    """
    summer = daily[daily['month'].isin(SUMMER_MONTHS)].copy()
    summer['period'] = summer['year'].apply(period_label)
    return summer


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a DataFrame and returns a DataFrame with the number and percent
    of missing values in each column that has at least one missing value.
    """
    counts = df.isna().sum()
    percents = counts / len(df) * 100
    summary = pd.DataFrame({'missing': counts, 'percent': percents})
    return summary[summary['missing'] > 0]


def seven_number_summary(values: pd.Series) -> pd.Series:
    """
    Takes a numeric Series and returns its mean, standard deviation,
    minimum, first quartile, median, third quartile and maximum.
    """
    return values.describe()[['mean', 'std', 'min', '25%', '50%', '75%',
                              'max']]


def plot_unhealthy_days(counts: pd.Series, path: str) -> None:
    """
    Takes the unhealthy days per year Series and saves a line plot of it
    to the given path.
    """
    plot_data = counts.reset_index()
    plot_data.columns = ['year', 'unhealthy_days']
    sns.relplot(data=plot_data, x='year', y='unhealthy_days', kind='line',
                marker='o')
    plt.title('Unhealthy PM2.5 days per year in King County')
    plt.xlabel('Year')
    plt.ylabel('Days with PM2.5 at or above 35.5 ug/m3')
    plt.savefig(path, bbox_inches='tight')
    plt.close('all')


def plot_monthly_means(means: pd.DataFrame, path: str) -> None:
    """
    Takes the monthly means DataFrame and saves a bar plot comparing the
    average PM2.5 in each month for the two periods to the given path.
    """
    sns.catplot(data=means, x='month', y='mean_pm25', hue='period',
                kind='bar')
    plt.title('Average daily PM2.5 by month in King County')
    plt.xlabel('Month')
    plt.ylabel('Average daily PM2.5 (ug/m3)')
    plt.savefig(path, bbox_inches='tight')
    plt.close('all')


def plot_summer_distribution(summer: pd.DataFrame, path: str) -> None:
    """
    Takes the summer days DataFrame and saves histograms of daily PM2.5
    for each period to the given path, to check the shape of the data
    before running the planned significance tests.
    """
    grid = sns.displot(data=summer, x='mean_pm25', col='period', bins=40)
    grid.set_axis_labels('County daily mean PM2.5 (ug/m3)', 'Number of days')
    grid.set_titles('Summer days, {col_name}')
    plt.savefig(path, bbox_inches='tight')
    plt.close('all')


def check_test_assumptions(summer: pd.DataFrame,
                           counts: pd.Series) -> None:
    """
    Prints checks of the assumptions for the planned tests: sample sizes,
    skew and spread of summer PM2.5 in each period (two-sample t-test and
    Mann-Whitney U test), and the residuals of a linear regression of
    unhealthy days on year.
    """
    early = summer[summer['period'] == '2000-2014']['mean_pm25']
    late = summer[summer['period'] == '2015-2025']['mean_pm25']
    print('Summer days per period:', len(early), len(late))
    print('Summer PM2.5 skew:', round(early.skew(), 2),
          round(late.skew(), 2))
    print('Summer PM2.5 standard deviation:', round(early.std(), 2),
          round(late.std(), 2))
    levene = stats.levene(early, late)
    print('Levene test for equal variance p-value:',
          round(levene.pvalue, 4))

    years = counts.index.to_series()
    fit = stats.linregress(years, counts)
    residuals = counts - (fit.intercept + fit.slope * years)
    shapiro = stats.shapiro(residuals)
    print('Regression residuals Shapiro-Wilk p-value:',
          round(shapiro.pvalue, 4))


def main():
    paths = []
    for name in sorted(os.listdir(DATA_FOLDER)):
        if name.startswith('daily_88101_'):
            paths.append(os.path.join(DATA_FOLDER, name))
    raw = load_king_county(paths)
    print('Raw King County rows and columns:', raw.shape)
    print()
    print('Missing values in the raw data:')
    print(missing_summary(raw))
    print()

    readings = clean_readings(raw)
    readings.to_csv(CLEAN_FILE, index=False)
    print('Cleaned readings rows and columns:', readings.shape)
    print('Missing values after cleaning:')
    print(missing_summary(readings))
    print()

    print('Readings by monitoring site:')
    print(readings['Local Site Name'].value_counts())
    print()
    print('Readings by sample duration:')
    print(readings['Sample Duration'].value_counts())
    print()
    print('Seven-number summary of PM2.5 per reading (ug/m3):')
    print(seven_number_summary(readings['Arithmetic Mean']))
    print()
    print('Seven-number summary of AQI per reading:')
    print(seven_number_summary(readings['AQI']))
    print()

    daily = daily_county(readings)
    print('First and last day:', daily.index.min(), daily.index.max())
    print('Days with at least one reading:', len(daily))
    print('Calendar days with no reading:', missing_days(daily))
    print('Days with at least one reading per year:')
    print(daily.groupby('year').size())
    print('Readings below 0 ug/m3:',
          len(readings[readings['Arithmetic Mean'] < 0]))
    print('Seven-number summary of county daily mean PM2.5:')
    print(seven_number_summary(daily['mean_pm25']))
    print()

    counts = unhealthy_days_per_year(daily)
    print('Unhealthy PM2.5 days per year:')
    print(counts)
    print('Seven-number summary of unhealthy days per year:')
    print(seven_number_summary(counts))
    print()
    print('Unhealthy PM2.5 days by month and period:')
    print(unhealthy_days_by_month(daily))
    print()

    means = monthly_means(daily)
    print('Average daily PM2.5 by month and period:')
    print(means)
    print()

    summer = summer_days(daily)
    print('Seven-number summary of summer daily PM2.5 by period:')
    print(summer.groupby('period')['mean_pm25'].describe())
    print()
    check_test_assumptions(summer, counts)

    plot_unhealthy_days(counts, 'unhealthy_days.png')
    plot_monthly_means(means, 'monthly_pm25.png')
    plot_summer_distribution(summer, 'summer_distribution.png')


if __name__ == '__main__':
    main()
