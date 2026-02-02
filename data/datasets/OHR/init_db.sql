-- Oracle HR (Human Resources) database schema
-- Converted from Oracle to MySQL

DROP DATABASE IF EXISTS OHR;
CREATE DATABASE OHR;
USE OHR;

-- Grant privileges to tata user
GRANT ALL PRIVILEGES ON OHR.* TO 'tata'@'localhost';
FLUSH PRIVILEGES;

-- REGIONS table to hold region information for locations
CREATE TABLE regions (
    region_id INT NOT NULL PRIMARY KEY,
    region_name VARCHAR(25)
);

-- COUNTRIES table to hold country information
CREATE TABLE countries (
    country_id CHAR(2) NOT NULL PRIMARY KEY,
    country_name VARCHAR(60),
    region_id INT,
    FOREIGN KEY (region_id) REFERENCES regions(region_id)
);

-- LOCATIONS table to hold address information for company departments
CREATE TABLE locations (
    location_id INT NOT NULL PRIMARY KEY,
    street_address VARCHAR(40),
    postal_code VARCHAR(12),
    city VARCHAR(30) NOT NULL,
    state_province VARCHAR(25),
    country_id CHAR(2),
    FOREIGN KEY (country_id) REFERENCES countries(country_id)
);

-- DEPARTMENTS table to hold company department information
CREATE TABLE departments (
    department_id INT NOT NULL PRIMARY KEY,
    department_name VARCHAR(30) NOT NULL,
    manager_id INT,
    location_id INT,
    FOREIGN KEY (location_id) REFERENCES locations(location_id)
);

-- JOBS table to hold job roles within the company
CREATE TABLE jobs (
    job_id VARCHAR(10) NOT NULL PRIMARY KEY,
    job_title VARCHAR(35) NOT NULL,
    min_salary INT,
    max_salary INT
);

-- EMPLOYEES table to hold employee personnel information
CREATE TABLE employees (
    employee_id INT NOT NULL PRIMARY KEY,
    first_name VARCHAR(20),
    last_name VARCHAR(25) NOT NULL,
    email VARCHAR(25) NOT NULL UNIQUE,
    phone_number VARCHAR(20),
    hire_date DATE NOT NULL,
    job_id VARCHAR(10) NOT NULL,
    salary DECIMAL(8,2) CHECK (salary > 0),
    commission_pct DECIMAL(2,2),
    manager_id INT,
    department_id INT,
    FOREIGN KEY (department_id) REFERENCES departments(department_id),
    FOREIGN KEY (job_id) REFERENCES jobs(job_id),
    FOREIGN KEY (manager_id) REFERENCES employees(employee_id)
);

-- Add the circular foreign key constraint from departments to employees
ALTER TABLE departments
ADD CONSTRAINT dept_mgr_fk
    FOREIGN KEY (manager_id) REFERENCES employees(employee_id);

-- JOB_HISTORY table to hold the history of jobs that employees have held
CREATE TABLE job_history (
    employee_id INT NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    job_id VARCHAR(10) NOT NULL,
    department_id INT,
    PRIMARY KEY (employee_id, start_date),
    CHECK (end_date > start_date),
    FOREIGN KEY (job_id) REFERENCES jobs(job_id),
    FOREIGN KEY (employee_id) REFERENCES employees(employee_id),
    FOREIGN KEY (department_id) REFERENCES departments(department_id)
);

-- EMP_DETAILS_VIEW that joins employees, jobs, departments, countries, and locations
CREATE OR REPLACE VIEW emp_details_view AS
SELECT
    e.employee_id,
    e.job_id,
    e.manager_id,
    e.department_id,
    d.location_id,
    l.country_id,
    e.first_name,
    e.last_name,
    e.salary,
    e.commission_pct,
    d.department_name,
    j.job_title,
    l.city,
    l.state_province,
    c.country_name,
    r.region_name
FROM
    employees e
    JOIN departments d ON e.department_id = d.department_id
    JOIN jobs j ON e.job_id = j.job_id
    JOIN locations l ON d.location_id = l.location_id
    JOIN countries c ON l.country_id = c.country_id
    JOIN regions r ON c.region_id = r.region_id;

-- Create indexes for performance
CREATE INDEX emp_department_ix ON employees(department_id);
CREATE INDEX emp_job_ix ON employees(job_id);
CREATE INDEX emp_manager_ix ON employees(manager_id);
CREATE INDEX emp_name_ix ON employees(last_name, first_name);
CREATE INDEX dept_location_ix ON departments(location_id);
CREATE INDEX jhist_job_ix ON job_history(job_id);
CREATE INDEX jhist_employee_ix ON job_history(employee_id);
CREATE INDEX jhist_department_ix ON job_history(department_id);
CREATE INDEX loc_city_ix ON locations(city);
CREATE INDEX loc_state_province_ix ON locations(state_province);
CREATE INDEX loc_country_ix ON locations(country_id);
