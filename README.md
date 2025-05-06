# Detection Analysis Tool

This Python script analyzes PCR detection test results and categorizes them as Invalid, False Positive, or False Negative according to specific rules.

## Requirements

- Python 3.6+
- pandas library

To install the required dependencies:
```
pip install pandas
```

## Usage

Basic usage:
```
python analyze_detection_results.py falseDetectionList.csv
```

To save the analysis results to a CSV file:
```
python analyze_detection_results.py falseDetectionList.csv -o results.csv
```

To include detailed channel information in the output:
```
python analyze_detection_results.py falseDetectionList.csv -d
```

To save detailed results to CSV:
```
python analyze_detection_results.py falseDetectionList.csv -d -o detailed_results.csv
```

## Rules for Analysis

The script analyzes the detection results according to these rules:

1. **Invalid Test**: If positive control (PC) is invalid, count as invalid test.
2. **False Positive**: If PC is valid and any channel for a sample ID shows a false positive curve, count this test as a false positive (each sample ID is counted only once).
3. **False Negative**: If PC is valid and ALL FOUR channels (ch2, ch3, ch4, and ch5) are false negatives, count this test as a false negative. The test must have data for all four channels to be classified as a false negative.

## Output

The script generates a report showing:
- Count and list of Invalid PC tests
- Count and list of False Positive tests
- Count and list of False Negative tests
- Summary of total counts

With the `-d` or `--detailed` flag, the report includes:
- Channel information for each test
- Statistics on which channels most commonly show false positives

When an output file is specified using `-o`, the results are saved to a CSV file with one row per test.

## Example

```
Analysis Results for falseDetectionList.csv
----------------------------------------

Invalid PC Tests: 2
  - FCFD4
  - 13.MB.40.S3

False Positive Tests: 12
  - 04.EN.189.S7
  - 04.MB.45.S1
  - 06.EN.206.S1.GL
  - 06.GF.146.NTC
  - 11.EN.204.S1.GL
  - 11.EN.212.S1.GL
  - 12.GF.186.NTC_CP
  - 13.EN.193.S7
  - 13.EN.207.S1.GL
  - 16.EN.208.S1..GL
  - 6.MB.46.S1
  - FCFD_FCSRB1_GF

False Negative Tests: 4
  - 04.EN.24.S2
  - 04.EN.214.S10
  - 12.EN.82.S1
  - 14.MB.106.S2

Summary:
  Invalid PC Tests: 2
  False Positive Tests: 12
  False Negative Tests: 4
  Total: 18
```

# Test information input
- Test samples related input should be fileed properly into the ###_testlog.csv file (Critical information: Used for, Input [C], test ID3)
- Export .db file into single test CSVs and rename accordingly based on the "Test ID#" in ###_testlog.csv
- Install 