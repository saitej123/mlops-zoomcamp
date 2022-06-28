
import pickle
import pandas as pd
import sys

from datetime import datetime

with open('model.bin', 'rb') as f_in:
    dv, lr = pickle.load(f_in)


categorical = ['PUlocationID', 'DOlocationID']

def read_data(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df

def data_with_date(month,year):
    datetime_object1 = datetime.strptime(month,'%B')
    datetime_object2 = datetime.strptime(year,'%Y')

    filename= "https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_"+ str(datetime_object2.year) + "-" + str(datetime_object1.month).zfill(2)+".parquet"
    #print(filename)

    df = read_data(filename)
    return df


def mean_predicted_duration(df):
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    year= datetime.today().year
    month=datetime.today().month
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df["prediction"]=y_pred
    df_result=df[["ride_id","prediction"]]
    df_result.to_parquet(
                "week_4_df",
                engine='pyarrow',
                compression=None,
                index=False
                )
    return(y_pred.mean())

def main():
    month = sys.argv[1]
    year = sys.argv[2]
    print(month , year)
    df_result = data_with_date(month,year)
    print(mean_predicted_duration(df_result))

main()