# Necessary imports for this notebook
import csv
import numpy as np
import pandas as pd
import time
import random

def generate_cql(customer_table: pd.DataFrame, terminal_table: pd.DataFrame, transaction_table: pd.DataFrame, dataSetSize: str):
    print("start to create csv")

    # make sure indexes pair with number of rows
    customer_table = customer_table.reset_index()
    data = []
    header = ["customerId", "x_geo", "y_geo", "meanAmount", "std_amount", "mean_tx_per_day"]

    with open('.\csv\\{}\\{}_customers.csv'.format(dataSetSize, dataSetSize), 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        print("composing customers")
        for index, row in customer_table.iterrows():
            tmp = [row['CUSTOMER_ID'], row['x_customer_id'], row['y_customer_id'], row['mean_amount'], row['std_amount'], row['mean_nb_tx_per_day']]
            data.append(tmp)
            
        writer.writerows(data)
    f.close()

    # make sure indexes pair with number of rows
    terminal_table = terminal_table.reset_index()
    data = []
    header = ["terminalId", "x_geo", "y_geo"]

    with open('.\csv\\{}\\{}_terminals.csv'.format(dataSetSize, dataSetSize), 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        print("composing terminals")
        for index, row in terminal_table.iterrows():
            # print("\t--- {}".format(index))
            terminalId = str(row['TERMINAL_ID']).replace(".0", "")
            tmp = [terminalId, row['x_terminal_id'], row['y_terminal_id']]
            data.append(tmp)
            
        writer.writerows(data)
    f.close()

    # make sure indexes pair with number of rows
    transaction_table = transaction_table.reset_index()
    data = []
    header = ["txId", "customerId", "terminalId", "amount", "date", "time", "semester", "isFraud"]

    fileIndex = 1
    
    f = open('.\csv\\{}\\{}_txs{}.csv'.format(dataSetSize, dataSetSize, fileIndex), 'w', encoding='UTF8', newline='')
    writer = csv.writer(f)
    writer.writerow(header)
    print("composing {} txs".format(dataSetSize))
    for index, row in transaction_table.iterrows():

        if(index > fileIndex*1000000):
            writer.writerows(data)
            f.close()
            fileIndex+=1
            data = []
            f = open('.\csv\\{}\\{}_txs{}.csv'.format(dataSetSize, dataSetSize, fileIndex), 'w', encoding='UTF8', newline='')
            writer = csv.writer(f)
            writer.writerow(header)

        print("\t--- {} of {} - {} {}".format(index, len(transaction_table), dataSetSize, fileIndex))
        is_fraud = True if row['IS_FRAUD'] == "1" else False
        date = str(row['TX_DATETIME']).split(" ")[0]
        time = str(row['TX_DATETIME']).split(" ")[1]
        semester = get_semester(date)
        tmp = [index, row["CUSTOMER_ID"], row['TERMINAL_ID'], row['TX_AMOUNT'], date, time, semester, str(is_fraud)]
    
    writer.writerows(data)
    f.close()
    
    return

def get_semester(date: str):
    month = date.split("-")[1]
    if month=="01" or month=="02" or month=="03" or month=="04" or month=="05" or month=="06":
        return 1
    elif month=="07" or month=="08" or month=="09" or month=="10" or month=="11" or month=="12":
        return 2

def generate_customer_profiles_table(n_customers, random_state=0):

    np.random.seed(random_state)

    customer_id_properties = []

    # Generate customer properties from random distributions
    for customer_id in range(n_customers):

        x_customer_id = np.random.uniform(0, 100)
        y_customer_id = np.random.uniform(0, 100)

        # Arbitrary (but sensible) value
        mean_amount = np.random.uniform(5, 100)
        std_amount = mean_amount/2  # Arbitrary (but sensible) value

        mean_nb_tx_per_day = np.random.uniform(
            0, 4)  # Arbitrary (but sensible) value

        customer_id_properties.append([customer_id,
                                      x_customer_id, y_customer_id,
                                      mean_amount, std_amount,
                                      mean_nb_tx_per_day])

    customer_profiles_table = pd.DataFrame(customer_id_properties, columns=['CUSTOMER_ID',
                                                                            'x_customer_id', 'y_customer_id',
                                                                            'mean_amount', 'std_amount',
                                                                            'mean_nb_tx_per_day'])

    return customer_profiles_table


def generate_terminal_profiles_table(n_terminals, random_state=0):

    np.random.seed(random_state)

    terminal_id_properties = []

    # Generate terminal properties from random distributions
    for terminal_id in range(n_terminals):

        x_terminal_id = np.random.uniform(0, 100)
        y_terminal_id = np.random.uniform(0, 100)

        terminal_id_properties.append([terminal_id,
                                      x_terminal_id, y_terminal_id])

    terminal_profiles_table = pd.DataFrame(terminal_id_properties, columns=['TERMINAL_ID',
                                                                            'x_terminal_id', 'y_terminal_id'])

    return terminal_profiles_table


def get_list_terminals_within_radius(customer_profile, x_y_terminals, r):

    # Use numpy arrays in the following to speed up computations

    # Location (x,y) of customer as numpy array
    x_y_customer = customer_profile[[
        'x_customer_id', 'y_customer_id']].values.astype(float)

    # Squared difference in coordinates between customer and terminal locations
    squared_diff_x_y = np.square(x_y_customer - x_y_terminals)

    # Sum along rows and compute suared root to get distance
    dist_x_y = np.sqrt(np.sum(squared_diff_x_y, axis=1))

    # Get the indices of terminals which are at a distance less than r
    available_terminals = list(np.where(dist_x_y < r)[0])

    # Return the list of terminal IDs
    return available_terminals


def generate_transactions_table(customer_profile, start_date="2023-01-01", nb_days=365):

    # print("Genero transazioni")
    customer_transactions = []

    random.seed(int(customer_profile.CUSTOMER_ID))
    np.random.seed(int(customer_profile.CUSTOMER_ID))

    # For all days
    for day in range(nb_days):

        # print("\t--- day {}".format(day))

        # Random number of transactions for that day
        nb_tx = np.random.poisson(customer_profile.mean_nb_tx_per_day)

        # If nb_tx positive, let us generate transactions
        if nb_tx > 0:

            for tx in range(nb_tx):

                isFraud = 0
                # Time of transaction: Around noon, std 20000 seconds. This choice aims at simulating the fact that
                # most transactions occur during the day.
                time_tx = int(np.random.normal(86400/2, 20000))

                # If transaction time between 0 and 86400, let us keep it, otherwise, let us discard it
                if (time_tx > 0) and (time_tx < 86400):

                    # Amount is drawn from a normal distribution
                    amount = np.random.normal(
                        customer_profile.mean_amount, customer_profile.std_amount)

                    # If amount negative, draw from a uniform distribution
                    if amount < 0:
                        amount = np.random.uniform(
                            0, customer_profile.mean_amount*2)

                    amount = np.round(amount, decimals=2)

                    if len(customer_profile.available_terminals) > 0:

                        terminal_id = random.choice(
                            customer_profile.available_terminals)

                        customer_transactions.append([time_tx+day*86400, day,
                                                      customer_profile.CUSTOMER_ID,
                                                      terminal_id, amount, isFraud])

    customer_transactions = pd.DataFrame(customer_transactions, columns=[
                                         'TX_TIME_SECONDS', 'TX_TIME_DAYS', 'CUSTOMER_ID', 'TERMINAL_ID', 'TX_AMOUNT', 'IS_FRAUD'])

    if len(customer_transactions) > 0:
        customer_transactions['TX_DATETIME'] = pd.to_datetime(
            customer_transactions["TX_TIME_SECONDS"], unit='s', origin=start_date)
        customer_transactions = customer_transactions[[
            'TX_DATETIME', 'CUSTOMER_ID', 'TERMINAL_ID', 'TX_AMOUNT', 'TX_TIME_SECONDS', 'TX_TIME_DAYS', 'IS_FRAUD']]

    return customer_transactions


def generate_dataset(n_customers=10000, n_terminals=1000000, nb_days=90, start_date="2023-01-01", r=5):

    start_time = time.time()
    customer_profiles_table = generate_customer_profiles_table(
        n_customers, random_state=0)
    print("Time to generate customer profiles table: {0:.2}s".format(
        time.time()-start_time))

    start_time = time.time()
    terminal_profiles_table = generate_terminal_profiles_table(
        n_terminals, random_state=1)
    print("Time to generate terminal profiles table: {0:.2}s".format(
        time.time()-start_time))

    start_time = time.time()
    x_y_terminals = terminal_profiles_table[[
        'x_terminal_id', 'y_terminal_id']].values.astype(float)
    customer_profiles_table['available_terminals'] = customer_profiles_table.apply(
        lambda x: get_list_terminals_within_radius(x, x_y_terminals=x_y_terminals, r=r), axis=1)
    # With Pandarallel
    # customer_profiles_table['available_terminals'] = customer_profiles_table.parallel_apply(lambda x : get_list_closest_terminals(x, x_y_terminals=x_y_terminals, r=r), axis=1)
    customer_profiles_table['nb_terminals'] = customer_profiles_table.available_terminals.apply(
        len)
    print("Time to associate terminals to customers: {0:.2}s".format(
        time.time()-start_time))

    start_time = time.time()
    transactions_df = customer_profiles_table.groupby('CUSTOMER_ID').apply(
        lambda x: generate_transactions_table(x.iloc[0], start_date, nb_days)).reset_index(drop=True)
    # With Pandarallel
    # transactions_df=customer_profiles_table.groupby('CUSTOMER_ID').parallel_apply(lambda x : generate_transactions_table(x.iloc[0], nb_days=nb_days)).reset_index(drop=True)
    print("Time to generate transactions: {0:.2}s".format(
        time.time()-start_time))

    # Sort transactions chronologically
    transactions_df = transactions_df.sort_values('TX_DATETIME')
    # Reset indices, starting from 0
    transactions_df.reset_index(inplace=True, drop=True)
    transactions_df.reset_index(inplace=True)
    # TRANSACTION_ID are the dataframe indices, starting from 0
    transactions_df.rename(columns={'index': 'TRANSACTION_ID'}, inplace=True)

    return (customer_profiles_table, terminal_profiles_table, transactions_df)

def launchMenu():

    small, med, big = False, False, False
    
    while True:
        print("Do you want to generate SMALL dataset? y/n")    
        line = input()
        line = line.rstrip()
        if(line!="y" and line!="n" and line!="Y" and line!="N"):
            print("Unrecognized input, please retry")
            continue
        else:
            small = line == "y"
            break

    
    while True:
        print("Do you want to generate MEDIUM dataset? y/n")    
        line = input()
        line = line.rstrip()
        if(line!="y" and line!="n" and line!="Y" and line!="N"):
            print("Unrecognized input, please retry")
            continue
        else:
            med = line == "y"
            break


    while True:
        print("Do you want to generate LARGE dataset? y/n")
        line = input()
        line = line.rstrip()
        if(line!="y" and line!="n" and line!="Y" and line!="N"):
            print("Unrecognized input, please retry")
            continue
        else:
            big = line == "y"
            break

    while True:
        print("Your settings:\n\tSMALL={}\n\tMEDIUM={}\n\tLARGE={}\nDo you confirm? y/n".format(small, med, big))
        line = input()
        line = line.rstrip()
        if(line!="y" and line!="n" and line!="Y" and line!="N"):
            print("Unrecognized input, please retry")
            continue
        elif(line=="y" or line=="Y"):
            return (small, med, big)
        else:
            return launchMenu()

# MAIN

TRY = not True

[SMALL, MED, BIG] = launchMenu()

nb_days = 365
start_date = "2023-01-01"
radius = 5

if(SMALL == True):

    n_customers = 3000
    n_terminals = 5500

    [customer_profiles_table, terminal_profiles_table, transactions_df] = generate_dataset(n_customers, n_terminals, nb_days, start_date, radius)

    generate_cql(customer_profiles_table, terminal_profiles_table, transactions_df, "small")

if(MED == True):

    n_customers = 6000
    n_terminals = 11000

    [customer_profiles_table, terminal_profiles_table, transactions_df] = generate_dataset(n_customers, n_terminals, nb_days, start_date, radius)

    generate_cql(customer_profiles_table, terminal_profiles_table, transactions_df, "med")

if(BIG == True):

    n_customers = 8500
    n_terminals = 16000

    [customer_profiles_table, terminal_profiles_table, transactions_df] = generate_dataset(n_customers, n_terminals, nb_days, start_date, radius)

    generate_cql(customer_profiles_table, terminal_profiles_table, transactions_df, "big")

if(TRY == True):

    n_customers = 5
    n_terminals = 10
    nb_days = 5
    radius = 30

    [customer_profiles_table, terminal_profiles_table, transactions_df] = generate_dataset(n_customers, n_terminals, nb_days, start_date, radius)

    generate_cql(customer_profiles_table, terminal_profiles_table, transactions_df, "try")