from src.jobs.etl import ETL
import os
import pandas as pd

etl = ETL()

def test_load():
    etl.input = os.getcwd()
    labels, orders = etl.load_data()
    assert 'is_returning_customer' in labels.columns
    assert 245455 == labels.shape[0]

def test_additional_columns(orders_df):
    res = etl.additional_cols(orders_df)
    assert 'Isnullrank' in res.columns
    assert 2015 in list(res['year'].values)

def test_separate_by_year(orders_df):
    orders_df1 = etl.additional_cols(orders_df)
    res = etl.separate_by_year(orders_df1)
    assert 'Is2017' in res.columns

def get_new_data():
    data = {'customer_id': {0: '000097eabfd9', 1: '0000e2c6d9be', 2: '000133bb597f'},
            'order_date': {0: '2015-06-20', 1: '2016-01-29', 2: '2017-02-26'},
            'order_hour': {0: 19, 1: 20, 2: 19},
            'customer_order_rank': {0: 1.0, 1: 1.0, 2: 1.0},
            'is_failed': {0: 0, 1: 0, 2: 0},
            'voucher_amount': {0: 0.0, 1: 0.0, 2: 0.0},
            'delivery_fee': {0: 0.0, 1: 0.0, 2: 0.493},
            'amount_paid': {0: 11.4696, 1: 9.558, 2: 5.93658},
            'restaurant_id': {0: 5803498, 1: 239303498, 2: 206463498},
            'city_id': {0: 20326, 1: 76547, 2: 33833},
            'payment_id': {0: 1779, 1: 1619, 2: 1619},
            'platform_id': {0: 30231, 1: 30359, 2: 30359},
            'transmission_id': {0: 4356, 1: 4356, 2: 4324},
            'Isnullrank': {0: 0, 1: 0, 2: 0},
            'Isnonzerovoucher': {0: 0, 1: 0, 2: 0},
            'year': {0: 2015, 1: 2016, 2: 2017},
            'month': {0: '2015-06', 1: '2016-01', 2: '2017-02'},
            'week_number': {0: 25, 1: 4, 2: 8},
            'week_day': {0: 'Saturday', 1: 'Friday', 2: 'Sunday'},
            'hour_class_evening': {0: 1, 1: 1, 2: 1},
            'hour_class_day': {0: 0, 1: 0, 2: 0},
            'hour_class_night': {0: 0, 1: 0, 2: 0},
            'Is2017': {0: 0, 1: 0, 2: 1},
            'Is2016': {0: 0, 1: 1, 2: 0},
            'Is2015': {0: 1, 1: 0, 2: 0}}

    return pd.DataFrame.from_dict(data)

def test_initial_etl(labels_df):
    newdf = get_new_data()
    res = etl.etl_initial(newdf, labels_df)
    assert 3 == res.shape[0]

def test_rename_cols(labels_df):
    newdf = get_new_data()
    df = etl.etl_initial(newdf, labels_df)
    res = etl.rename_cols(df)
    res.to_csv('/Users/vinodkumar/Documents/GitHub/mkt-casestudy-ds/tests/data/rename_cols.csv', index=False)
    assert 'customer_order_rank_min' in res.columns

def test_etl_phase2():
    df = pd.read_csv(os.path.join(os.getcwd(), 'tests/data/rename_cols.csv'))
    df['order_date_amax'] = pd.to_datetime(df['order_date_amax'])
    df['order_date_amin'] = pd.to_datetime(df['order_date_amin'])
    res = etl.etl_phase2(df)
    res.to_csv('/Users/vinodkumar/Documents/GitHub/mkt-casestudy-ds/tests/data/etl2.csv', index=False)
    assert 'recenencyscore' in res.columns

def test_split_by_year():
    df = pd.read_csv('/Users/vinodkumar/Documents/GitHub/mkt-casestudy-ds/tests/data/etl2.csv')
    res1, res2, res3 = etl.split_by_year(df)
    assert 1 == res1.shape[0]

def test_select_imp_cols():
    df = pd.read_csv('/Users/vinodkumar/Documents/GitHub/mkt-casestudy-ds/tests/data/etl2.csv')
    res = etl.select_imp_cols(df)
    assert 'recenencyscore' in res


