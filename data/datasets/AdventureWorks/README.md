# AdventureWorks

Downloaded from: https://github.com/vishal180618/OLTP-AdventureWorks2019-MySQL/tree/3167d046f75cf2eb8eb0bd9768dc43ca9933378f.
The init_db.sql file was created by removing INSERT statements from the AdventureWorks.sql
We performed the following modifications on the schema:

- Replaced "." in table names by underscores so MySQL does not interpret this as database.table.
- Added a default value for rowguid

## Templates

- Select templates are derived from:
  - https://sqlzoo.net/wiki/AdventureWorks_easy_questions
  - https://www.w3resource.com/sql-exercises/adventureworks/adventureworks-exercises.php
  - And through human project analysis.
- Insert templates were build using LLMS using the database schema. 3 tables do not have INSERT statements: they are system / metadata tables or consist of auto-increment ID table.

## Notes

Our scripts to create dicts created a file named `dbo_AWBuildVersion_Database Version`. This lead to weird behaviour by treefmt, we renamed to `dbo_AWBuildVersion_Database Version`. We are not using this dict anyway as of now.
