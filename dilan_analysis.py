from datetime import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st

KNOWN_SOURCES = {'Reddit', 'SEO', 'AdWords'}

def load_data_local(file_path):
    with open(file_path, encoding='utf-8') as f:
        lines = f.readlines()
    parsed = [line.strip().split(';') for line in lines]
    max_len = max(len(row) for row in parsed)
    colnames = [f'col{i}' for i in range(max_len)]
    df = pd.DataFrame(parsed, columns=colnames)
    return df

def parse_read_event(fields, i):
    dt = fields[i]
    country = fields[i+2]
    user_id = fields[i+3]

    if fields[i+4] in KNOWN_SOURCES:
        # First-time visit (source known)
        source = fields[i+4]
        topic = fields[i+5]
        return {
            'datetime': dt,
            'event_type': 'read',
            'country': country,
            'user_id': user_id,
            'source': source,
            'topic': topic,
            'is_returning': False
        }, 6
    else:
        # Returning visit (no source provided)
        topic = fields[i+4]
        return {
            'datetime': dt,
            'event_type': 'read',
            'country': country,
            'user_id': user_id,
            'topic': topic,
            'is_returning': True           
        }, 5

def parse_subscribe_event(fields, i):
    return {
        'datetime': fields[i], 'event_type': 'subscribe', 'user_id': fields[i+2]
    }, 3

def parse_buy_event(fields, i):
    return {
        'datetime': fields[i], 'event_type': 'buy',
        'user_id': fields[i+2], 'price': fields[i+3]
    }, 4

def parse_log_file(file_path):
    read_rows, subscribe_rows, buy_rows = [], [], []
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            fields = line.strip().split(';')
            i = 0
            while i < len(fields):
                try:
                    event_type = fields[i+1]
                    if event_type == 'read':
                        record, step = parse_read_event(fields, i)
                        read_rows.append(record)
                    elif event_type == 'subscribe':
                        record, step = parse_subscribe_event(fields, i)
                        subscribe_rows.append(record)
                    elif event_type == 'buy':
                        record, step = parse_buy_event(fields, i)
                        buy_rows.append(record)
                    else:
                        step = 1
                    i += step
                except IndexError:
                    break
    return clean_read_df(read_rows), clean_subscribe_df(subscribe_rows), clean_buy_df(buy_rows)

def clean_read_df(rows):
    df = pd.DataFrame(rows)
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')

    first_time_df = df[df['is_returning'] == False].copy()
    returning_df = df[df['is_returning'] == True].copy()

    first_time_df.to_csv("clean_read_first_time.csv", index=False)
    returning_df.to_csv("clean_read_returning.csv", index=False)

    df.to_csv("clean_read.csv", index=False)
    return df

def clean_subscribe_df(rows):
    df = pd.DataFrame(rows)
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df.to_csv("clean_subscribe.csv", index=False)
    return df

def clean_buy_df(rows):
    df = pd.DataFrame(rows)
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df.to_csv("clean_buy.csv", index=False)
    return df

def validate_dataframes(read_df, subscribe_df, buy_df, log_file="validation_log.txt"):
    with open(log_file, "w", encoding="utf-8") as log:
        def w(s): log.write(s + "\n")

        w("=== BASIC STRUCTURE & NULLS ===")
        for df, name in zip([read_df, subscribe_df, buy_df], ['read', 'subscribe', 'buy']):
            w(f"\n{name.upper()} DF SHAPE: {df.shape}")
            w(f"{name} nulls:\n{df.isnull().sum().to_string()}")
            w(f"{name} dtypes:\n{df.dtypes.to_string()}")

        w("\n=== UNIQUE VALUES IN READ_DF ===")
        w("Countries: " + ", ".join(read_df['country'].unique()))
        w("Sources: " + ", ".join(str(s) for s in read_df['source'].dropna().unique()))
        w("Topics: " + ", ".join(read_df['topic'].unique()))

        w("\n=== VALUE CHECKS ===")
        invalid_countries = read_df[~read_df['country'].str.startswith("country_")]
        invalid_sources = read_df[~read_df['source'].isin(['Reddit', 'AdWords', 'SEO', 'Returning'])]
        w(f"Invalid countries: {len(invalid_countries)}")
        w(f"Invalid sources: {len(invalid_sources)}")

        w("\n=== DUPLICATE CHECKS ===")
        w(f"Duplicate read events: {read_df.duplicated().sum()}")
        w(f"Duplicate subscriptions: {subscribe_df.duplicated().sum()}")

        w("\n=== USER ACTIVITY ===")
        top_users = read_df['user_id'].value_counts().head(10).to_string()
        w("Reads per user (top 10):\n" + top_users)
        subscribers_not_readers = set(subscribe_df['user_id']) - set(read_df['user_id'])
        w(f"Subscribers not in read_df: {len(subscribers_not_readers)}")

        buyers = set(buy_df['user_id'])
        readers = set(read_df['user_id'])
        subscribers = set(subscribe_df['user_id'])
        w(f"Buyers not in read_df: {len(buyers - readers)}")
        w(f"Buyers not in subscribe_df: {len(buyers - subscribers)}")

        w("\n=== PRICE & TIMEFRAME ===")
        w("Price distribution:\n" + buy_df['price'].value_counts().to_string())
        w(f"Read timeframe: {read_df['datetime'].min()} -> {read_df['datetime'].max()}")
        w(f"Subscribe timeframe: {subscribe_df['datetime'].min()} -> {subscribe_df['datetime'].max()}")
        w(f"Buy timeframe: {buy_df['datetime'].min()} -> {buy_df['datetime'].max()}")

        w("\n=== UNIQUENESS IN READ_DF ===")
        w(f"Total reads: {len(read_df)}")
        w(f"Unique user-topic pairs: {read_df[['user_id', 'topic']].drop_duplicates().shape[0]}")
        w(f"Unique user-country pairs: {read_df[['user_id', 'country']].drop_duplicates().shape[0]}")

if __name__ == "__main__":
    raw_file_path = "dilans_data.csv"
    read_df, subscribe_df, buy_df = parse_log_file(raw_file_path)
    validate_dataframes(read_df, subscribe_df, buy_df)

    print("Preprocessing complete. Clean CSVs saved.")
    print(read_df.head())
    print(subscribe_df.head())
    print(buy_df.head())
