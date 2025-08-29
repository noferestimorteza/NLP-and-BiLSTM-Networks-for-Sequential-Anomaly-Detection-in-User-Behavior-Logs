import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import csv
from datetime import datetime
from urllib.parse import urlparse
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Define feature columns for each data type
email_feature_cols = ['unique_recipients', 'total_emails']
http_feature_cols = ['unique_url', 'total_content_category']
device_feature_cols = ['device_usage_count']  # Count of device usage events

def preprocess_device_data(df):
    """
    Preprocess device data to extract daily device usage count for user ACM2278
    """
    # Filter for user CMP2946 only
    df = df[df['user'] == 'ACM2278']
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Get all unique dates
    all_dates = pd.date_range(start=df['date'].min().date(), end=df['date'].max().date(), freq='D')
    
    features = []
    
    # Group by date to count device usage events
    for day, group in df.groupby(df['date'].dt.date):
        device_usage_count = len(group)  # Count all device usage events
        
        daily_features = {
            'data_type': 'device',
            'user': 'CMP2946',
            'date': pd.Timestamp(day),
            'device_usage_count': device_usage_count
        }
        features.append(daily_features)
    
    # Create a complete dataframe with all dates
    features_df = pd.DataFrame(features)
    complete_df = pd.DataFrame({'date': all_dates, 'user': 'CMP2946'})
    
    # Merge with our calculated features, filling missing values with 0
    features_df = complete_df.merge(
        features_df, 
        on=['user', 'date'], 
        how='left'
    ).fillna({
        'device_usage_count': 0,
        'data_type': 'device'
    })
    
    return features_df

def preprocess_email_data(df):
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
                        'data_type': 'email',
                        'user': user,
                        'content_category': category,
                        'date': date,
                        'unique_recipients': len(set(all_recipients)) if all_recipients else 0,
                        'total_emails': len(date_data),
                    }
                else:
                    # Create zero entry for missing date
                    daily_features = {
                        'data_type': 'email',
                        'user': user,
                        'content_category': category,
                        'date': date,
                        'unique_recipients': 0,
                        'total_emails': 0,
                    }
                
                features.append(daily_features)
    
    return pd.DataFrame(features)

def preprocess_http_data(df):
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
        
        for url in group['url'].dropna():
            try:
                parsed = urlparse(url)
                base_url = f"{parsed.scheme}://{parsed.netloc}"
                unique_base_urls.add(base_url)
            except:
                continue
        
        daily_features = {
            'data_type': 'http',
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
        'total_content_category': 0,
        'data_type': 'http'
    })
    
    return features_df

def merge_and_normalize_data(email_df, http_df, device_df):
    # Combine all datasets
    combined_df = pd.concat([email_df, http_df], ignore_index=True)
    
    # Create separate feature sets for each data type
    email_features = combined_df[combined_df['data_type'] == 'email'][['user', 'date', 'content_category'] + email_feature_cols].copy()
    http_features = combined_df[combined_df['data_type'] == 'http'][['user', 'date', 'content_category'] + http_feature_cols].copy()
    
    # Add device features (only for user CMP2946)
    device_features = device_df[['user', 'date'] + device_feature_cols].copy()
    
    # Normalize features separately
    email_scaler = MinMaxScaler()
    http_scaler = MinMaxScaler()
    device_scaler = MinMaxScaler()
    
    email_features[email_feature_cols] = email_scaler.fit_transform(email_features[email_feature_cols]).astype(np.float32)
    http_features[http_feature_cols] = http_scaler.fit_transform(http_features[http_feature_cols]).astype(np.float32)
    device_features[device_feature_cols] = device_scaler.fit_transform(device_features[device_feature_cols]).astype(np.float32)
    
    # Store original values before normalization
    for col in email_feature_cols:
        email_features[f'original_{col}'] = email_scaler.inverse_transform(email_features[email_feature_cols])[:, email_feature_cols.index(col)]
    
    for col in http_feature_cols:
        http_features[f'original_{col}'] = http_scaler.inverse_transform(http_features[http_feature_cols])[:, http_feature_cols.index(col)]
    
    for col in device_feature_cols:
        device_features[f'original_{col}'] = device_scaler.inverse_transform(device_features[device_feature_cols])[:, device_feature_cols.index(col)]
    
    # Merge the normalized features
    # First merge email and http
    combined_features = pd.merge(
        email_features, 
        http_features, 
        on=['user', 'date', 'content_category'], 
        how='outer'
    ).fillna(0)
    
    # Then merge with device data (only for user CMP2946)
    # For other users, device features will be 0
    combined_features = pd.merge(
        combined_features,
        device_features,
        on=['user', 'date'],
        how='left'
    ).fillna(0)
    
    # Add data_type column back
    combined_features['data_type'] = 'combined'
    
    return combined_features, email_scaler, http_scaler, device_scaler

def create_combined_sequences(data, user, category, n_steps=1):
    user_category_data = data[(data['user'] == user) & (data['content_category'] == category)]
    user_category_data = user_category_data.sort_values('date')
    
    if len(user_category_data) < n_steps:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    sequences = []
    labels = []
    dates = []
    original_values = []
    
    all_feature_cols = email_feature_cols + http_feature_cols + device_feature_cols
    
    for i in range(len(user_category_data) - n_steps + 1):
        seq = user_category_data.iloc[i:i+n_steps][all_feature_cols].values
        sequences.append(seq)
        labels.append(seq)  # Autoencoder: labels are the same as input sequences
        dates.append(user_category_data.iloc[i+n_steps-1]['date'])
        
        # Store original values for the last point in the sequence
        orig_vals = {}
        for col in email_feature_cols:
            orig_vals[col] = user_category_data.iloc[i+n_steps-1][f'original_{col}']
        for col in http_feature_cols:
            orig_vals[col] = user_category_data.iloc[i+n_steps-1][f'original_{col}']
        for col in device_feature_cols:
            orig_vals[col] = user_category_data.iloc[i+n_steps-1][f'original_{col}']
        
        original_values.append(orig_vals)
    
    sequences = np.array(sequences, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)
    dates = np.array(dates)
    
    return sequences, labels, dates, original_values

def build_combined_bilstm_model(input_shape):
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape),
        Dropout(0.3),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(32, return_sequences=True)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(input_shape[-1])  # Output shape matches input features
    ])
    
    model.compile(optimizer='adam', loss='mse')
    return model

def train_and_detect_combined_anomalies(features_df, verbose=1):
    users = features_df['user'].unique()
    categories = features_df['content_category'].unique()
    
    results = []
    
    for user in users:
        for category in categories:
            if category == "Miscellaneous":
                continue
                
            # Create sequences for this user-category combination
            sequences, labels, dates, original_values = create_combined_sequences(features_df, user, category)
            
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
            expected_features = len(email_feature_cols + http_feature_cols + device_feature_cols)
            if X_train.shape[1:] != (1, expected_features):
                print(f"Invalid shape for user {user}, category {category}: {X_train.shape}, expected (1, {expected_features})")
                continue
            
            # Build and train model
            model = build_combined_bilstm_model((X_train.shape[1], X_train.shape[2]))
            early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            
            print(f"Training model for user {user}, category {category}")
            print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            
            model.fit(X_train, y_train, 
                     epochs=100, 
                     batch_size=32,
                     validation_split=0.1,
                     callbacks=[early_stop],
                     verbose=verbose)
            
            # Calculate reconstruction error
            train_pred = model.predict(X_train, verbose=0)
            train_mse = np.mean(np.power(X_train - train_pred, 2), axis=(1,2))
            threshold = np.percentile(train_mse, 97)  # 97th percentile as threshold
            
            test_pred = model.predict(X_test, verbose=0)
            test_mse = np.mean(np.power(X_test - test_pred, 2), axis=(1,2))
            
            # Identify anomalies
            anomalies = test_mse > threshold
            
            # Store results with original values
            for i, (date, is_anomaly, error, orig_vals) in enumerate(zip(test_dates, anomalies, test_mse, test_original_values)):
                result = {
                    'data_type': 'combined',
                    'user': user,
                    'category': category,
                    'date': date,
                    'is_anomaly': is_anomaly,
                    'reconstruction_error': float(error),
                    'threshold': float(threshold)
                }
                
                # Add original values
                for col, val in orig_vals.items():
                    result[f'original_{col}'] = float(val)
                
                results.append(result)
    
    return pd.DataFrame(results)

def plot_combined_anomaly_results(anomaly_results, categories_per_figure=6):
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
        user_categories = anomaly_results[anomaly_results['category'].isin(category_chunk)][['user', 'category']].drop_duplicates()
        
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
                        linewidth=2)
        
        # Customize plot
        plt.xlabel('Date')
        plt.ylabel('Reconstruction Error (Anomaly Score)')
        plt.title(f'Combined Anomaly Detection Results')
        plt.legend(bbox_to_anchor=(1.05, 1), fontsize='large')
        plt.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Show plot
        plt.show()

def calculate_combined_metrics(anomaly_results):
    # Group by user and date to get daily anomalies
    anomaly_per_user_day = anomaly_results.groupby(['user', 'date'])['is_anomaly'].any().reset_index()
    anomaly_per_user_day = anomaly_per_user_day.rename(columns={'is_anomaly': 'daily_anomaly'})
    
    # Add a flag to mark if we've found a match
    anomaly_per_user_day['find'] = 0
    
    csv_file_path = 'UEBA/r6.2-2.csv'
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    
    with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            if row[0] not in ["email", "http", "device"]:
                continue
                
            date_answer = datetime.strptime(row[2], "%m/%d/%Y %H:%M").date()
            
            for index, anomaly in anomaly_per_user_day.iterrows():
                date_anomaly = pd.to_datetime(anomaly['date']).date()
                
                if date_anomaly != date_answer:
                    continue
                    
                if anomaly['find'] == 1:
                    break
                    
                if anomaly['daily_anomaly']:
                    TP += 1
                else:
                    FN += 1
                    
                anomaly_per_user_day.at[index, 'find'] = 1
                break
    
    # Calculate FP and TN
    FP = ((anomaly_per_user_day['find'] != 1) & (anomaly_per_user_day['daily_anomaly'])).sum()
    TN = ((anomaly_per_user_day['find'] != 1) & (~anomaly_per_user_day['daily_anomaly'])).sum()
    
    print(f"True Negative: {TN}, True Positive: {TP}, False Positive: {FP}, False Negative: {FN}")
    
    # Calculate precision, recall, and F1-score
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    
    return TP, FP, TN, FN, precision, recall, f1

# Main execution
if __name__ == "__main__":
    # Load data
    try:
        email_df = pd.read_csv('UEBA/Categorized_email_ACM2278.csv')
        http_df = pd.read_csv('UEBA/Categorized_http_ACM2278.csv')
        device_df = pd.read_csv('UEBA/device.csv')  # Device data file
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Input file not found: {e}")

    # Preprocess data
    print("Preprocessing email data...")
    email_features = preprocess_email_data(email_df)
    
    print("Preprocessing HTTP data...")
    http_features = preprocess_http_data(http_df)
    
    print("Preprocessing device data...")
    device_features = preprocess_device_data(device_df)
    
    print("Merging and normalizing data...")
    combined_features, email_scaler, http_scaler, device_scaler = merge_and_normalize_data(
        email_features, http_features, device_features
    )
    
    # Save features
    combined_features.to_csv('combined_features_with_device_ACM2278.csv', index=False)
    
    # Train models and detect anomalies
    print("Training combined model with device features and detecting anomalies...")
    anomaly_results = train_and_detect_combined_anomalies(combined_features, verbose=1)
    
    # Plot results
    print("Plotting results...")
    plot_combined_anomaly_results(anomaly_results)
    
    # Calculate metrics
    print("Calculating metrics...")
    calculate_combined_metrics(anomaly_results)
    
    # Save results
    try:
        anomaly_results.to_csv('combined_anomaly_with_device_ACM2278.csv', index=False)
        print("Results saved to 'combined_anomaly_with_device_ACM2278.csv'")
    except Exception as e:
        print(f"Error saving results: {e}")