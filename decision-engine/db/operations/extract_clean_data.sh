#! /bin/bash

set -e

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <db_file> <table_name> <output_csv_file>"
    echo "Example: $0 db/analytics.db orders data/orders.csv"
    exit 1
fi

DB_FILE="$1"
TABLE_NAME="$2"
OUTPUT_CSV_FILE="$3"

if [ ! -f "$DB_FILE" ]; then
    echo "Error: DB file '$DB_FILE' not found"
    exit 1
fi

echo "Extracting '$TABLE_NAME' of db '$DB_FILE' to '$OUTPUT_CSV_FILE'.."

duckdb "$DB_FILE" <<EOF
copy "$TABLE_NAME" to "$OUTPUT_CSV_FILE" (HEADER, DELIMITER ',');
EOF

echo "Successfully extracted data."
