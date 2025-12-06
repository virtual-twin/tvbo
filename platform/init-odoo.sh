#!/bin/bash
set -euo pipefail

DB_HOST=${DB_HOST:-postgres}
DB_USER=${DB_USER:-odoo}
DB_PASSWORD=${DB_PASSWORD:-odoo}
DB_NAME=${DB_NAME:-tvbo}

export PGPASSWORD="$DB_PASSWORD"

# Wait for PostgreSQL to be ready (auth included so we do not loop forever)
until pg_isready -h "$DB_HOST" -U "$DB_USER" > /dev/null 2>&1; do
  echo "Waiting for PostgreSQL at $DB_HOST..."
  sleep 2
done

echo "PostgreSQL is ready!"

# Check if database exists
if psql -h "$DB_HOST" -U "$DB_USER" -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='${DB_NAME}'" | grep -q 1; then
  echo "Database exists, upgrading TVBO module..."
  odoo -d "$DB_NAME" -u tvbo --stop-after-init --without-demo=True --db_host="$DB_HOST" --db_user="$DB_USER" --db_password="$DB_PASSWORD"
else
  echo "Creating database and installing base modules..."
  odoo -d "$DB_NAME" -i base,website --stop-after-init --without-demo=True --db_host="$DB_HOST" --db_user="$DB_USER" --db_password="$DB_PASSWORD"

  echo "Installing TVBO module..."
  odoo -d "$DB_NAME" -i tvbo --stop-after-init --without-demo=True --db_host="$DB_HOST" --db_user="$DB_USER" --db_password="$DB_PASSWORD"
fi

echo "TVBO module ready!"
echo "Starting Odoo server in development mode..."
exec odoo -d "$DB_NAME" --db_host="$DB_HOST" --db_user="$DB_USER" --db_password="$DB_PASSWORD" --db-filter="^${DB_NAME}$" --dev=all
