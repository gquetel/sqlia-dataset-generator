# Dataset Schemas

This directory contains the database schemas for different datasets used in SQL injection attack generation.

**Important**: Each dataset gets its own MySQL database. The `init_db.sql` for each dataset must create and configure its own database.

## Structure

Each dataset should have its own directory with the following structure:

```
databases/
└── <dataset_name>/
    ├── init_db.sql          # Creates database and schema
    ├── dicts/               # Dictionary files for placeholders
    ├── queries/*.csv        # CSV templates (required)
    └── *.sql                # Optional: Raw SQL with -- template-ID annotations
```

## Adding a New Dataset

1. Create `databases/<dataset_name>/` with `init_db.sql`, `queries/*.csv`, `dicts/`
1. Optionally add `<statement_type>.sql` files with `-- template-ID` annotations (e.g., `INSERT INTO table VALUES (...); -- OHR-I1`)
1. Add `SOURCE databases/<dataset_name>/init_db.sql;` to `data/bootstrap.sql`
1. Add dataset to `config.toml` under `[[datasets]]`
1. Update `tests/test_database_schemas.py` parametrize decorators
1. Run `pytest tests/test_database_schemas.py -v` to verify

## Examples

**OurAirports**: CSV templates with placeholders

- `queries/select.csv`, `queries/insert.csv`, etc.
- Dictionary files in `dicts/`

**OHR**: Raw SQL script + CSV templates

- `insert.sql` with annotated statements: `INSERT INTO regions VALUES (10, 'Europe'); -- OHR-I1`
- `queries/insert.csv` with template definitions
- Dictionary files in `dicts/`
