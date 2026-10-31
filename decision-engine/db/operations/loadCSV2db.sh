#! /bin/bash

set -e

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <path_to_csv> <output_db_file> <target_table_name>"
    echo "Example: $0 ./data/orders_sample.csv analytics.db orders"
    exit 1
fi

CSV_FILE="$1"
DB_FILE="$2"
TABLE_NAME="$3"

if [ ! -f "$CSV_FILE" ]; then
    echo "Error: CSV file '$CSV_FILE' not found."
    exit 1
fi

echo "Converting '$CSV_FILE' into table '$TABLE_NAME' inside '$DB_FILE'..."

duckdb "$DB_FILE" <<EOF
create table "$TABLE_NAME" as select * from read_csv_auto('$CSV_FILE');
EOF

echo "Success! Db file '$DB_FILE' created."
