# Dataset Schemas

This directory contains the database schemas for different datasets used in SQL injection attack generation.

**Important**: Each dataset gets its own MySQL database. The `init_db.sql` for each dataset must create and configure its own database.

## Structure

Each dataset should have its own directory with the following structure:

```
databases/
└── <dataset_name>/
    ├── init_db.sql          # Creates database and schema (tables)
    ├── dicts/               # Dictionary files for placeholder values
    │   ├── <placeholder1>
    │   ├── <placeholder2>
    │   └── ...
    └── queries/             # Query templates
        ├── select.csv
        ├── insert.csv
        ├── update.csv
        ├── delete.csv
        └── ...
```

## Adding a New Dataset

1. Create a directory: `databases/<dataset_name>/`
2. Create `init_db.sql` that:
   - Drops and creates its own database
   - Grants privileges to the `tata` user
   - Defines all tables and schemas
3. Add the schema to `data/bootstrap.sql`:
   ```sql
   SOURCE databases/<dataset_name>/init_db.sql;
   ```
4. Create query templates in `queries/` directory
5. Create dictionary files in `dicts/` directory
6. Update your TOML config to include the new dataset:
   ```toml
   [[datasets]]
   name = "<dataset_name>"  # This will also be the database name

   [datasets.statements]
   # ... statement proportions
   ```

## Example: Airport Dataset

See `databases/airport/init_db.sql` for a complete example that:
- Creates the `airport` database
- Grants privileges to `tata` user
- Defines 6 tables (airport, runways, navaids, countries, regions, airport_frequencies)

The directory also contains:
- Query templates for different SQL operations in `queries/`
- Dictionary files for placeholder values in `dicts/`
