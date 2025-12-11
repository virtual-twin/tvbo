# Docker Compose Setup

This setup runs three services that can communicate with each other:

## Services

1. **tvbo-api** (port 8000): TVBO API server
2. **odoo** (port 8069): Odoo ERP system
3. **postgres** (internal): PostgreSQL database for Odoo

All services are connected via the `tvbo-network` bridge network.

## Usage

Start all services:
```bash
docker-compose up -d
```

View logs:
```bash
docker-compose logs -f
```

Stop all services:
```bash
docker-compose down
```

Stop and remove volumes (⚠️ deletes all data):
```bash
docker-compose down -v
```

## Access

- **TVBO API**: http://localhost:8000/docs
- **Odoo**: http://localhost:8069
  - Default credentials on first setup:
    - Email: admin
    - Password: admin (you'll set this on first login)

## Odoo ↔ TVBO Communication

From within Odoo (Python code), you can call the TVBO API using:

```python
import requests

# TVBO API is accessible at http://tvbo-api:8000
response = requests.get('http://tvbo-api:8000/docs')
```

The hostname `tvbo-api` resolves to the TVBO container within the Docker network.

## Custom Odoo Addons

Place your custom Odoo modules in the `./odoo-addons` directory, and they will be available in Odoo.

## Configuration

To customize Odoo, create an `odoo.conf` file in `./odoo-config/` directory.
