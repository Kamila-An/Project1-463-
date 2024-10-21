import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# DataLoader class for loading stock data from yfinance
class DataLoader:
    def __init__(self, ticker='AAPL', start_date='2022-01-01', end_date='2023-01-01'):
        # Initialize the data loader with a stock ticker, start date, and end date
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date

    def load_data(self):
        # Download stock data using yfinance within the given date range
        stock_data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        stock_data.reset_index(inplace=True)  # Reset index to convert Date from index to a column
        return stock_data

# Sorter class for performing merge sort on stock data by date
class Sorter:
    def merge_sort(self, arr, key=lambda x: x):
        # Base condition for recursion
        if len(arr) > 1:
            # Find the middle index to split the array
            mid = len(arr) // 2
            left_half = arr[:mid]
            right_half = arr[mid:]

            # Recursively sort each half
            self.merge_sort(left_half, key=key)
            self.merge_sort(right_half, key=key)

            # Merge the sorted halves
            i = j = k = 0
            while i < len(left_half) and j < len(right_half):
                # Compare elements and merge them back in sorted order
                if key(left_half[i]) < key(right_half[j]):
                    arr[k] = left_half[i]
                    i += 1
                else:
                    arr[k] = right_half[j]
                    j += 1
                k += 1

            # Copy any remaining elements from the left half
            while i < len(left_half):
                arr[k] = left_half[i]
                i += 1
                k += 1

            # Copy any remaining elements from the right half
            while j < len(right_half):
                arr[k] = right_half[j]
                j += 1
                k += 1
        # Return the sorted array
        return arr

# TrendAnalyzer class for detecting periods of maximum gain or loss
class TrendAnalyzer:
    def kadane_algorithm(self, arr):
        # Kadane's algorithm to find the maximum sum subarray
        max_sum = current_sum = arr[0]       # Initialize max_sum and current_sum with the first element
        start = end = s = 0        # Initialize indices to track the start and end of the max subarray

        # Iterate through the array starting from the second element
        for i in range(1, len(arr)):
            # If adding the current element drops the sum, reset the sum to the current element
            if arr[i] > current_sum + arr[i]:
                current_sum = arr[i]
                s = i           # Reset starting index
            else:
                current_sum += arr[i]            # Otherwise, add it to the current sum

            # Update max_sum if the current_sum is higher
            if current_sum > max_sum:
                max_sum = current_sum
                start = s             # Update the starting index
                end = i               # Update the ending index

        return max_sum, start, end

    def analyze_trends(self, stock_data):
        # Calculate daily changes in closing prices
        daily_changes = np.diff(stock_data['Close'])

        # Use Kadane's algorithm to find the subarray with the maximum sum
        max_gain, start_index, end_index = self.kadane_algorithm(daily_changes)

        # Map the indices back to the actual dates in the stock data
        start_date = stock_data.iloc[start_index]['Date']
        end_date = stock_data.iloc[end_index + 1]['Date']       # +1 because np.diff results in a shifted index
        return max_gain, start_date, end_date

# AnomalyDetector class for detecting anomalies using a closest pair approach
class AnomalyDetector:
    def closest_pair(self, points):
        # Sort the points based on their dates (first element of each tuple)
        points = sorted(points, key=lambda x: x[0])

        # Define a function to calculate the distance between two points
        def distance(p1, p2):
            return abs(p1[1] - p2[1])

        # Initialize minimum distance to infinity and placeholders for the closest points
        min_distance = float('inf')
        closest_points = (None, None)

        # Iterate through each pair of consecutive points to find the closest pair
        for i in range(len(points) - 1):
            dist = distance(points[i], points[i + 1])
            if dist < min_distance:
                min_distance = dist
                closest_points = (points[i], points[i + 1])

        return closest_points, min_distance

    def detect_anomalies(self, stock_data):
        # Convert stock data into a list of (Date, Close) tuples
        points = [(row['Date'], row['Close']) for index, row in stock_data.iterrows()]

        # Find the closest pair of points, which may indicate anomalies
        (anomaly1, anomaly2), anomaly_distance = self.closest_pair(points)
        return anomaly1, anomaly2, anomaly_distance

# ReportGenerator class for generating and visualizing results
class ReportGenerator:
    def generate_report(self, stock_data, max_gain_period, anomalies):
        # Create a plot for stock prices
        plt.figure(figsize=(12, 6))
        plt.plot(stock_data['Date'], stock_data['Close'], label='Stock Prices', color='blue')

        # Highlight the period of maximum gain with vertical lines
        plt.axvline(max_gain_period[0], color='green', linestyle='--', label='Start of Max Gain')
        plt.axvline(max_gain_period[1], color='red', linestyle='--', label='End of Max Gain')

        # Highlight detected anomalies with scatter points
        plt.scatter([anomalies[0][0], anomalies[1][0]], [anomalies[0][1], anomalies[1][1]], color='orange', label='Anomalies')

        # Add labels, title, and legend
        plt.xlabel('Date')
        plt.ylabel('Closing Price')
        plt.title('Stock Prices with Detected Trends and Anomalies')
        plt.legend()
        plt.grid(True)
        plt.show()

# Main function to integrate all components and perform analysis
def main():
    # Load stock data
    data_loader = DataLoader(ticker='AAPL', start_date='2022-01-01', end_date='2023-01-01')
    stock_data = data_loader.load_data()

    # Sort the stock data by date
    sorter = Sorter()
    stock_data_list = stock_data[['Date', 'Close']].values.tolist()
    sorter.merge_sort(stock_data_list, key=lambda x: x[0])
    sorted_stock_data = pd.DataFrame(stock_data_list, columns=['Date', 'Close'])

    # Analyze trends to find the period of maximum gain
    trend_analyzer = TrendAnalyzer()
    max_gain, start_date, end_date = trend_analyzer.analyze_trends(sorted_stock_data)
    print(f"Maximum gain of {max_gain} detected from {start_date} to {end_date}")

    # Detect anomalies in the stock data
    anomaly_detector = AnomalyDetector()
    anomaly1, anomaly2, anomaly_distance = anomaly_detector.detect_anomalies(sorted_stock_data)
    print(f"Closest pair of points (potential anomaly): {anomaly1} and {anomaly2} with distance {anomaly_distance}")

    # Generate a visual report of the analysis
    report_generator = ReportGenerator()
    report_generator.generate_report(sorted_stock_data, (start_date, end_date), (anomaly1, anomaly2))

# Entry point for running the script
if __name__ == '__main__':
    main()
