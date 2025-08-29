import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import csv
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Define feature columns
#feature_cols = ['unique_recipients', 'unique_senders', 'unique_activities', 
#                'unique_attachments', 'total_emails', 'avg_email_size', 'total_attachments']
feature_cols = ['unique_recipients', 'total_emails']

def preprocess_data(df):
    # Validate required columns
    required_cols = ['date', 'user', 'content_category']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    if df['date'].isna().any():
        raise ValueError("Invalid date values found in 'date' column")

    # Ensure all recipient fields are strings and handle missing values
    for col in ['to', 'cc', 'bcc']:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str)
        else:
            df[col] = ''

    features = []
    
    # Get date range for each user-category combination
    all_users = df['user'].unique()
    all_categories = df['content_category'].unique()
    all_dates = pd.date_range(start=df['date'].min(), end=df['date'].max())
    
    for user in all_users:
        for category in all_categories:
            # Get existing data for this user-category
            user_category_data = df[(df['user'] == user) & (df['content_category'] == category)]
            
            # Create entries for all dates in range
            for date in all_dates:
                date_data = user_category_data[user_category_data['date'].dt.date == date.date()]
                
                if len(date_data) > 0:
                    # Process existing data
                    all_recipients = []
                    
                    # Process 'to' field
                    if 'to' in date_data.columns:
                        all_recipients.extend([addr.strip() for addr in date_data['to'].str.split(',').explode() if addr.strip()])
                    
                    # Process 'cc' field
                    if 'cc' in date_data.columns:
                        all_recipients.extend([addr.strip() for addr in date_data['cc'].str.split(',').explode() if addr.strip()])
                    
                    # Process 'bcc' field
                    if 'bcc' in date_data.columns:
                        all_recipients.extend([addr.strip() for addr in date_data['bcc'].str.split(',').explode() if addr.strip()])
                    
                    daily_features = {
                        'user': user,
                        'content_category': category,
                        'date': date,
                        'unique_recipients': len(set(all_recipients)) if all_recipients else 0,
                        'total_emails': len(date_data),
                    }
                else:
                    # Create zero entry for missing date
                    daily_features = {
                        'user': user,
                        'content_category': category,
                        'date': date,
                        'unique_recipients': 0,
                        'total_emails': 0,
                    }
                
                features.append(daily_features)
    
    features_df = pd.DataFrame(features)
    
    # Validate feature columns
    missing_features = [col for col in feature_cols if col not in features_df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")
    
    # Ensure all feature columns are numeric
    for col in feature_cols:
        features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0)
    
    # Check for NaN values
    if features_df[feature_cols].isna().any().any():
        print("Warning: NaN values found in features_df. Filling with 0.")
        features_df[feature_cols] = features_df[feature_cols].fillna(0)
    
    # Normalize features
    scaler = MinMaxScaler()
    features_df[feature_cols] = scaler.fit_transform(features_df[feature_cols]).astype(np.float32)
    
    # Verify data types
    print("Feature dtypes after preprocessing:")
    print(features_df[feature_cols].dtypes)
    
    return features_df, scaler

def create_sequences(data, user, category, n_steps=1):
    user_category_data = data[(data['user'] == user) & (data['content_category'] == category)]
    user_category_data = user_category_data.sort_values('date')
    
    if len(user_category_data) < n_steps:
        return np.array([]), np.array([]), np.array([])
    
    sequences = []
    labels = []
    dates = []
    
    for i in range(len(user_category_data) - n_steps + 1):
        seq = user_category_data.iloc[i:i+n_steps][feature_cols].values
        sequences.append(seq)
        labels.append(seq)  # Autoencoder: labels are the same as input sequences
        dates.append(user_category_data.iloc[i+n_steps-1]['date'])
    
    sequences = np.array(sequences, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)
    dates = np.array(dates)
    
    # Verify sequence dtypes
    print(f"User: {user}, Category: {category}")
    print(f"Sequences shape: {sequences.shape}, dtype: {sequences.dtype}")
    print(f"Labels shape: {labels.shape}, dtype: {labels.dtype}")
    
    return sequences, labels, dates

def build_bilstm_model(input_shape):
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
        Dropout(0.2),
        Bidirectional(LSTM(32, return_sequences=True)),  # Keep return_sequences=True
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(input_shape[-1])  # Output shape matches input features
    ])
    
    model.compile(optimizer='adam', loss='mse')
    return model

def train_and_detect_anomalies(features_df, verbose=1):
    users = features_df['user'].unique()
    categories = features_df['content_category'].unique()
    
    results = []
    
    for user in users:
        for category in categories:
            if(category=="Miscellaneous"):
                continue
            # Create sequences for this user-category combination
            sequences, labels, dates = create_sequences(features_df, user, category)
            
            if len(sequences) < 10:  # Skip if not enough data
                print(f"Skipping user {user}, category {category}: insufficient sequences ({len(sequences)})")
                continue
                
            # Split into train/test
            split_idx = int(0.8 * len(sequences))
            if split_idx == 0 or len(sequences) - split_idx < 1:
                print(f"Skipping user {user}, category {category}: train/test split invalid")
                continue
            
            X_train, y_train = sequences[:split_idx], labels[:split_idx]
            X_test, y_test = sequences[split_idx:], labels[split_idx:]
            test_dates = dates[split_idx:]
            
            # Validate input shape
            if X_train.shape[1:] != (1, len(feature_cols)):
                print(f"Invalid shape for user {user}, category {category}: {X_train.shape}")
                continue
            
            # Build and train model
            model = build_bilstm_model((X_train.shape[1], X_train.shape[2]))
            early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
            
            print(f"Training model for user {user}, category {category}")
            print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            
            model.fit(X_train, y_train, 
                     epochs=50, 
                     batch_size=32,
                     validation_split=0.1,
                     callbacks=[early_stop],
                     verbose=verbose)
            
            # Calculate reconstruction error
            train_pred = model.predict(X_train, verbose=0)
            print(f"train_pred shape: {train_pred.shape}")
            train_mse = np.mean(np.power(X_train - train_pred, 2), axis=(1,2))
            threshold = np.percentile(train_mse, 97)  # 95th percentile as threshold
            
            test_pred = model.predict(X_test, verbose=0)
            print(f"test_pred shape: {test_pred.shape}")
            test_mse = np.mean(np.power(X_test - test_pred, 2), axis=(1,2))
            
            # Identify anomalies
            anomalies = test_mse > threshold
            
            # Store results
            for i, (date, is_anomaly, error) in enumerate(zip(test_dates, anomalies, test_mse)):
                results.append({
                    'user': user,
                    'category': category,
                    'date': date,
                    'is_anomaly': is_anomaly,
                    'reconstruction_error': float(error),
                    'threshold': float(threshold)
                })
    
    return pd.DataFrame(results)


def plot_anomaly_results(anomaly_results, categories_per_figure=6):
    # Ensure date is in datetime format
    anomaly_results['date'] = pd.to_datetime(anomaly_results['date'])
    
    # Get unique categories
    categories = anomaly_results['category'].unique()
    
    # Split categories into chunks of up to categories_per_figure
    category_chunks = [categories[i:i + categories_per_figure] 
                      for i in range(0, len(categories), categories_per_figure)]
    
    # Colors for categories within each figure (using tab10 for up to 3 distinct colors)
    colors = cm.get_cmap('tab10')(np.linspace(0, 1, categories_per_figure))
    
    # Create a figure for each chunk of categories
    for chunk_idx, category_chunk in enumerate(category_chunks):
        plt.figure(figsize=(12, 6))
        
        # Get all user-category combinations for the current chunk of categories
        user_categories = anomaly_results[anomaly_results['category'].isin(category_chunk)][['user', 'category']].drop_duplicates()
        
        # Plot lines and points for each user-category combination in the chunk
        for idx, (user, category) in enumerate(user_categories.itertuples(index=False)):
            # Assign color based on category (same color for same category across users)
            color_idx = list(category_chunk).index(category) % categories_per_figure
            color = colors[color_idx]
            
            # Filter data for this user and category
            subset = anomaly_results[(anomaly_results['user'] == user) & 
                                   (anomaly_results['category'] == category)]
            
            # Sort by date to ensure correct line connection
            subset = subset.sort_values('date')
            
            # Separate anomalies and non-anomalies
            anomalies = subset[subset['is_anomaly'] == True]
            non_anomalies = subset[subset['is_anomaly'] == False]
            
            # Plot line connecting all points in this user-category
            plt.plot(subset['date'], 
                     subset['reconstruction_error'], 
                     color=color, 
                     label=f'{user} - {category}', 
                     alpha=0.5, 
                     linewidth=1.5)
            
            # Plot non-anomaly points
            plt.scatter(non_anomalies['date'], 
                        non_anomalies['reconstruction_error'], 
                        c=[color], 
                        s=50, 
                        alpha=0.6, 
                        edgecolors='w')
            
            # Plot anomaly points
            plt.scatter(anomalies['date'], 
                        anomalies['reconstruction_error'], 
                        c='red', 
                        s=100, 
                        marker='x', 
                        label=None)  # Avoid duplicate legend entries for anomalies
        
        # Customize plot
        plt.xlabel('Date')
        plt.ylabel('Reconstruction Error (Anomaly Score)')
        plt.title(f'Anomaly Detection Results (Categories {", ".join(category_chunk)})')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.grid(True)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Show plot
        plt.show()

def calculateMetrics (anomaly_results):
    anomaly_per_user_day = anomaly_results.groupby(['user', 'date'])['is_anomaly'].all().reset_index()
    anomaly_per_user_day = anomaly_per_user_day.rename(columns={'is_anomaly': 'daily_anomaly'})
    result = anomaly_results.groupby(['user', 'date']).agg({
        'is_anomaly': 'any'
    }).reset_index()
    result = result.rename(columns={'is_anomaly': 'daily_anomaly'})

    new_column_name = 'find'
    new_column_values = 0  # Your logic to compute the new column values
    result[new_column_name] = new_column_values

    csv_file_path = 'UEBA/r6.2-2.csv'  # Replace with your actual file path
    TP=0
    FP=0
    TN=0
    FN=0
    with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            if(row[0] != "email"):
                continue
            dateAnswer = datetime.strptime(row[2], "%m/%d/%Y %H:%M").date()
            for index, anomaly in result.iterrows():
                dateAnomaly = pd.to_datetime(anomaly['date']).date()
                if(dateAnomaly != dateAnswer):
                    continue
                if(anomaly['find']==1 ):
                    break
                #print(f"anomaly time {dateAnomaly} and flag {anomaly['daily_anomaly']} AND FIND {anomaly['find']}")
                if(anomaly['daily_anomaly']):
                    print(f"******** {dateAnomaly}")
                    TP=TP+1
                else:
                    FN=FN+1
                result.at[index, 'find'] = 1 
                break
    FP = ((result['find'] != 1) & (result['daily_anomaly'])).sum()
    TN = ((result['find'] != 1) & (result['daily_anomaly']!=True)).sum()
    print(f"True Negatve: {TN} , True Positive: {TP}, False Positive: {FP}, False Negatve: {FN}")


# Load data with error handling
try:
    df = pd.read_csv('UEBA/Categorized_email_CMP2946.csv')
except FileNotFoundError:
    raise FileNotFoundError("Input file 'processed_email_CMP2946.csv' not found. Please check the file path.")

# Preprocess and engineer features
features_df, scaler = preprocess_data(df)
features_df.to_csv('email_features_CMP2946.csv')

# Train models and detect anomalies
anomaly_results = train_and_detect_anomalies(features_df, verbose=1)

plot_anomaly_results(anomaly_results)

calculateMetrics(anomaly_results)
# Save results
try:
    anomaly_results.to_csv('temail_anomaly_CMP2946.csv', index=False)
    print("Results saved to 'email_anomaly_CMP2946.csv'")
except Exception as e:
    print(f"Error saving results: {e}")

    import pandas as pd
import matplotlib.pyplot as plt

# Assuming anomaly_results is already generated from the provided code
# If running standalone, you would load it like this:
# anomaly_results = pd.read_csv('temp/email_anomaly_cmp2946.csv')
