#!/usr/bin/env python3
import sys
import pandas as pd
import numpy as np
import joblib

def engineer_features(df):
    """Enhanced feature engineering - EXACT same as enhanced training"""
    df = df.copy()
    
    # Basic derived features
    df['miles_per_day'] = df['miles_traveled'] / df['trip_duration_days']
    df['receipts_per_day'] = df['total_receipts_amount'] / df['trip_duration_days']
    
    # Trip duration categories
    df['is_1_day_trip'] = (df['trip_duration_days'] == 1).astype(int)
    df['is_2_day_trip'] = (df['trip_duration_days'] == 2).astype(int)
    df['is_3_day_trip'] = (df['trip_duration_days'] == 3).astype(int)
    df['is_4_day_trip'] = (df['trip_duration_days'] == 4).astype(int)
    df['is_5_day_trip'] = (df['trip_duration_days'] == 5).astype(int)
    df['is_6_day_trip'] = (df['trip_duration_days'] == 6).astype(int)
    df['is_7_day_trip'] = (df['trip_duration_days'] == 7).astype(int)
    df['is_8_day_trip'] = (df['trip_duration_days'] == 8).astype(int)
    df['is_9_day_trip'] = (df['trip_duration_days'] == 9).astype(int)
    df['is_10_plus_day_trip'] = (df['trip_duration_days'] >= 10).astype(int)
    
    # ENHANCED: Specific problem pattern features
    # Pattern 1: Long trips (8+ days) with high mileage and receipts
    df['long_trip_high_mileage'] = ((df['trip_duration_days'] >= 8) & (df['miles_traveled'] > 800)).astype(int)
    df['long_trip_high_receipts'] = ((df['trip_duration_days'] >= 8) & (df['total_receipts_amount'] > 1000)).astype(int)
    df['long_trip_combo'] = ((df['trip_duration_days'] >= 8) & (df['miles_traveled'] > 800) & (df['total_receipts_amount'] > 1000)).astype(int)
    
    # Pattern 2: 5-day trips with very high mileage (problematic case 35)
    df['five_day_high_mileage'] = ((df['trip_duration_days'] == 5) & (df['miles_traveled'] > 700)).astype(int)
    df['five_day_extreme_mileage'] = ((df['trip_duration_days'] == 5) & (df['miles_traveled'] > 800)).astype(int)
    
    # Pattern 3: Very high efficiency trips that seem to get bonuses
    df['very_high_efficiency'] = (df['miles_per_day'] > 140).astype(int)
    df['extreme_efficiency'] = (df['miles_per_day'] > 150).astype(int)
    
    # Pattern 4: 10+ day trips (special handling needed)
    df['very_long_trip'] = (df['trip_duration_days'] >= 10).astype(int)
    
    # Pattern 5: High mileage trips (900+ miles) 
    df['extreme_mileage'] = (df['miles_traveled'] >= 900).astype(int)
    df['ultra_high_mileage'] = (df['miles_traveled'] >= 1000).astype(int)
    
    # ENHANCED: More granular mileage and receipt categories
    df['miles_800_to_900'] = ((df['miles_traveled'] >= 800) & (df['miles_traveled'] < 900)).astype(int)
    df['miles_900_to_1000'] = ((df['miles_traveled'] >= 900) & (df['miles_traveled'] < 1000)).astype(int)
    df['miles_1000_plus'] = (df['miles_traveled'] >= 1000).astype(int)
    
    df['receipts_1000_to_1500'] = ((df['total_receipts_amount'] >= 1000) & (df['total_receipts_amount'] < 1500)).astype(int)
    df['receipts_1500_plus'] = (df['total_receipts_amount'] >= 1500).astype(int)
    
    # Original important features (kept from analysis)
    df['is_receipt_sweet_spot'] = ((df['total_receipts_amount'] >= 600) & (df['total_receipts_amount'] <= 800)).astype(int)
    df['has_small_receipts'] = ((df['total_receipts_amount'] > 0) & (df['total_receipts_amount'] < 100)).astype(int)
    df['has_no_receipts'] = (df['total_receipts_amount'] == 0).astype(int)
    df['has_moderate_receipts'] = ((df['total_receipts_amount'] >= 100) & (df['total_receipts_amount'] < 600)).astype(int)
    df['has_high_receipts'] = (df['total_receipts_amount'] >= 800).astype(int)
    
    # Mileage categories
    df['low_miles'] = (df['miles_traveled'] < 100).astype(int)
    df['medium_miles'] = ((df['miles_traveled'] >= 100) & (df['miles_traveled'] < 500)).astype(int)
    df['high_miles'] = (df['miles_traveled'] >= 500).astype(int)
    
    # Kevin's efficiency sweet spot
    df['is_efficiency_sweet_spot'] = ((df['miles_per_day'] >= 180) & (df['miles_per_day'] <= 220)).astype(int)
    
    # Receipt pattern analysis
    df['receipt_ends_in_49_or_99'] = (
        (df['total_receipts_amount'].astype(str).str.endswith('.49')) |
        (df['total_receipts_amount'].astype(str).str.endswith('.99'))
    ).astype(int)
    
    # ENHANCED: Complex interaction features
    df['miles_x_days'] = df['miles_traveled'] * df['trip_duration_days']
    df['receipts_x_days'] = df['total_receipts_amount'] * df['trip_duration_days']
    df['miles_x_receipts'] = df['miles_traveled'] * df['total_receipts_amount']
    df['efficiency_x_receipts'] = df['miles_per_day'] * df['total_receipts_amount']
    df['efficiency_x_days'] = df['miles_per_day'] * df['trip_duration_days']
    
    # ENHANCED: Polynomial features for complex relationships
    df['days_squared'] = df['trip_duration_days'] ** 2
    df['days_cubed'] = df['trip_duration_days'] ** 3
    df['miles_squared'] = df['miles_traveled'] ** 2
    df['receipts_squared'] = df['total_receipts_amount'] ** 2
    df['miles_per_day_squared'] = df['miles_per_day'] ** 2
    df['miles_per_day_cubed'] = df['miles_per_day'] ** 3
    
    # Log features
    df['log_miles'] = np.log1p(df['miles_traveled'])
    df['log_receipts'] = np.log1p(df['total_receipts_amount'])
    df['log_miles_per_day'] = np.log1p(df['miles_per_day'])
    df['log_days'] = np.log1p(df['trip_duration_days'])
    
    # Ratio features
    df['miles_to_receipts_ratio'] = df['miles_traveled'] / (df['total_receipts_amount'] + 1)
    df['receipts_to_miles_ratio'] = df['total_receipts_amount'] / (df['miles_traveled'] + 1)
    df['efficiency_ratio'] = df['miles_per_day'] / (df['receipts_per_day'] + 1)
    
    # Binned features for threshold effects
    df['miles_bin_100'] = (df['miles_traveled'] // 100).astype(int)
    df['receipts_bin_200'] = (df['total_receipts_amount'] // 200).astype(int)
    df['days_bin'] = df['trip_duration_days']
    df['efficiency_bin_50'] = (df['miles_per_day'] // 50).astype(int)
    
    # ENHANCED: Problem-specific combinations
    df['problematic_case_type_1'] = ((df['trip_duration_days'] == 8) & (df['miles_traveled'] > 800) & (df['total_receipts_amount'] > 1000)).astype(int)
    df['problematic_case_type_2'] = ((df['trip_duration_days'] == 5) & (df['miles_traveled'] > 750)).astype(int)
    df['problematic_case_type_3'] = ((df['trip_duration_days'] >= 10)).astype(int)
    df['problematic_case_type_4'] = ((df['trip_duration_days'] == 6) & (df['miles_traveled'] > 900)).astype(int)
    
    return df

# Check for proper command line arguments
if len(sys.argv) != 4:
    print(f"Error: Expected 3 arguments, got {len(sys.argv)-1}")
    print("Usage: python3 calculate_reimbursement.py <days> <miles> <receipts>")
    sys.exit(1)

# Validate arguments are not empty
for i, arg in enumerate(sys.argv[1:], 1):
    if not arg or arg.strip() == '':
        print(f"Error: Argument {i} is empty")
        print("Usage: python3 calculate_reimbursement.py <days> <miles> <receipts>")
        sys.exit(1)

try:
    # Load the enhanced pre-trained model and feature list
    model = joblib.load('enhanced_reimbursement_model.pkl')
    feature_list = joblib.load('enhanced_feature_list.pkl')

    # Get inputs from command line - with additional validation
    try:
        trip_duration = int(sys.argv[1])
        miles_traveled = float(sys.argv[2])
        receipts_amount = float(sys.argv[3])
    except ValueError as ve:
        print(f"Error: Invalid argument format - {ve}")
        print("Usage: python3 calculate_reimbursement.py <days> <miles> <receipts>")
        print("Arguments must be: <integer> <number> <number>")
        sys.exit(1)

    # Create a DataFrame from the inputs
    input_data = pd.DataFrame([{
        'trip_duration_days': trip_duration,
        'miles_traveled': miles_traveled,
        'total_receipts_amount': receipts_amount
    }])

    # Apply the EXACT same enhanced feature engineering as training
    input_data = engineer_features(input_data)

    # Ensure all features are present and in correct order
    for feature in feature_list:
        if feature not in input_data.columns:
            input_data[feature] = 0

    # Select features in the exact same order as training
    X = input_data[feature_list]

    # Predict the reimbursement
    prediction = model.predict(X)

    # Print the single numeric output, rounded to 2 decimal places
    print(f"{prediction[0]:.2f}")

except ValueError as e:
    print(f"Error: Invalid input - {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1) 