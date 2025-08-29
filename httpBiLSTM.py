import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import csv 
from datetime import datetime
from urllib.parse import urlparse

# Define feature columns
feature_cols = ['unique_url','total_content_category']

def preprocess_data(df):
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Get all unique users, dates, and content categories
    all_users = df['user'].unique()
    all_dates = pd.date_range(start=df['date'].min().date(), end=df['date'].max().date(), freq='D')
    all_content_categories = df['content_category'].unique()
    
    # Create a complete multi-index of all possible combinations
    complete_index = pd.MultiIndex.from_product(
        [all_users, all_dates, all_content_categories],
        names=['user', 'date', 'content_category']
    )
    
    features = []
    
    for (user, day, content_category), group in df.groupby(['user', df['date'].dt.date, 'content_category']):
        unique_base_urls = set()
        
        print(f"user {user} day {day} content_category {content_category} size {len(group)}")
        
        for url in group['url'].dropna():
            try:
                parsed = urlparse(url)
                base_url = f"{parsed.scheme}://{parsed.netloc}"
                unique_base_urls.add(base_url)
            except:
                continue
        
        daily_features = {
            'user': user,
            'date': pd.Timestamp(day),
            'content_category': content_category,
            'unique_url': len(unique_base_urls), 
            'total_content_category': len(group)
        }
        features.append(daily_features)
    
    # Create initial features dataframe
    features_df = pd.DataFrame(features)
    
    # Create a complete dataframe with all combinations
    complete_df = pd.DataFrame(index=complete_index).reset_index()
    
    # Merge with our calculated features, filling missing values with 0
    features_df = complete_df.merge(
        features_df, 
        on=['user', 'date', 'content_category'], 
        how='left'
    ).fillna({
        'unique_url': 0,
        'total_content_category': 0
    })
    
    # Ensure all feature columns are numeric
    for col in feature_cols:
        features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0)

    # Check for NaN values
    if features_df[feature_cols].isna().any().any():
        print("Warning: NaN values found in features_df. Filling with 0.")
        features_df[feature_cols] = features_df[feature_cols].fillna(0)
    
    # Store original values before normalization
    features_df['original_unique_url'] = features_df['unique_url']
    features_df['original_total_content_category'] = features_df['total_content_category']
    
    # Normalize features
    scaler = MinMaxScaler()
    features_df[feature_cols] = scaler.fit_transform(features_df[feature_cols]).astype(np.float32)
    
    return features_df, scaler

def create_sequences(data, user, category, n_steps=1):
    user_category_data = data[(data['user'] == user) & (data['content_category'] == category)]
    user_category_data = user_category_data.sort_values('date')
    
    if len(user_category_data) < n_steps:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    sequences = []
    labels = []
    dates = []
    original_values = []  # Store original values
    
    for i in range(len(user_category_data) - n_steps + 1):
        seq = user_category_data.iloc[i:i+n_steps][feature_cols].values
        sequences.append(seq)
        labels.append(seq)  # Autoencoder: labels are the same as input sequences
        dates.append(user_category_data.iloc[i+n_steps-1]['date'])
        # Store original values for the last point in the sequence
        original_values.append({
            'unique_url': user_category_data.iloc[i+n_steps-1]['original_unique_url'],
            'total_content_category': user_category_data.iloc[i+n_steps-1]['original_total_content_category']
        })
    
    sequences = np.array(sequences, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)
    dates = np.array(dates)
    original_values = np.array(original_values)
    
    return sequences, labels, dates, original_values

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
            sequences, labels, dates, original_values = create_sequences(features_df, user, category)
            
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
            test_original_values = original_values[split_idx:]
            
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
            train_mse = np.mean(np.power(X_train - train_pred, 2), axis=(1,2))
            threshold = 0.001#np.percentile(train_mse, 95)  # 95th percentile as threshold

            test_pred = model.predict(X_test, verbose=0)
            test_mse = np.mean(np.power(X_test - test_pred, 2), axis=(1,2))
            
            # Identify anomalies
            anomalies = test_mse > threshold
            
            # Store results with original values
            for i, (date, is_anomaly, error, orig_vals) in enumerate(zip(test_dates, anomalies, test_mse, test_original_values)):
                results.append({
                    'user': user,
                    'category': category,
                    'date': date,
                    'is_anomaly': is_anomaly,
                    'reconstruction_error': float(error),
                    'threshold': float(threshold),
                    'unique_url': float(orig_vals['unique_url']),
                    'total_content_category': float(orig_vals['total_content_category'])
                })
    
    return pd.DataFrame(results)

def plot_anomaly_results(anomaly_results, categories_per_figure=7):
    # Ensure date is in datetime format
    anomaly_results['date'] = pd.to_datetime(anomaly_results['date'])
    
    # Get unique categories
    categories = anomaly_results['category'].unique()
    
    # Split categories into chunks of up to categories_per_figure
    category_chunks = [categories[i:i + categories_per_figure] 
                      for i in range(0, len(categories), categories_per_figure)]
    
    # Colors for categories within each figure
    colors = cm.get_cmap('tab10')(np.linspace(0, 1, categories_per_figure))
    
    # Create a figure for each chunk of categories
    for chunk_idx, category_chunk in enumerate(category_chunks):
        plt.figure(figsize=(14, 8))
        
        # Get all user-category combinations for the current chunk of categories
        user_categories = anomaly_results[anomaly_results['category'].isin(category_chunk)][[ 'user','category']].drop_duplicates()
        
        # Plot lines and points for each user-category combination in the chunk
        for idx, (user, category) in enumerate(user_categories.itertuples(index=False)):
            # Assign color based on category
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
                     subset['total_content_category'],  # Use actual value instead of reconstruction error
                     color=color, 
                     label=f'{category}', 
                     alpha=0.5, 
                     linewidth=1.5)
            
            # Plot normal points in blue
            plt.scatter(non_anomalies['date'], 
                        non_anomalies['total_content_category'], 
                        c='blue',  # Blue for normal points
                        s=50, 
                        alpha=0.7, 
                        edgecolors='w')
            
            # Plot anomaly points in red
            if len(anomalies) > 0:
                plt.scatter(anomalies['date'], 
                            anomalies['total_content_category'], 
                            c='red',  # Red for anomalies
                            s=100, 
                            marker='x', 
                            linewidth=2)
        
        # Customize plot
        plt.xlabel('Date')
        plt.ylabel('Total Content Category (Actual Value)')  # Show actual values on Y-axis
        plt.title(f'Anomaly Detection Results (Categories {", ".join(category_chunk)})')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='large')
        plt.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Show plot
        plt.show()

def calculateMetircs(anomaly_results):
    csv_file_path = 'UEBA/r6.2-2.csv'
    TP=0
    FP=0
    TN=0
    FN=0
    new_column_name = 'find'
    new_column_values = 0
    anomaly_results[new_column_name] = new_column_values
    count=0
    with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            if(row[0] != "http"):
                continue
            dateAnswer = datetime.strptime(row[2], "%m/%d/%Y %H:%M").date()
            for index, anomaly in anomaly_results.iterrows():
                dateAnomaly = pd.to_datetime(anomaly['date']).date()
                if(dateAnomaly != dateAnswer):
                    continue
                if(anomaly['find']==1 ):
                    break
                print(f"anomaly time {dateAnomaly} and flag {anomaly['is_anomaly']}")
                if(anomaly['is_anomaly']):
                    TP=TP+1
                else:
                    FN=FN+1
                anomaly_results.at[index, 'find'] = 1 
                break
    FP = ((anomaly_results['find'] != 1) & (anomaly_results['is_anomaly'])).sum()
    TN = ((anomaly_results['find'] != 1) & (anomaly_results['is_anomaly']!=True)).sum()
    print(f"True Negative: {TN} , True Positive: {TP}, False Positive: {FP}, False Negative: {FN}")

# Load data with error handling
try:
    df = pd.read_csv('UEBA/Categorized_http_CMP2946.csv')
except FileNotFoundError:
    raise FileNotFoundError("Input file 'Categorized_http_CMP2946' not found. Please check the file path.")

# Preprocess and engineer features
features_df, scaler = preprocess_data(df)
features_df.to_csv('http_features_CMP2946.csv')

# Train models and detect anomalies
anomaly_results = train_and_detect_anomalies(features_df, verbose=1)

plot_anomaly_results(anomaly_results)

calculateMetircs(anomaly_results)

try:
    anomaly_results.to_csv('anomaly_http_CMP2946.csv', index=False)
    print("Results saved to 'anomaly_http_CMP2946.csv'")
except Exception as e:
    print(f"Error saving results: {e}")