---
title: SQL Summary
date: 2020-03-02
description: SQL cheat sheet and summary
category: summary
type: notes
---

```sql
select * from movies;
select title, year from movies where year <= 2010;
select * from movies where title like "Toy Story%" # begins query
select * from movies where title like "Toy Story _" # any single character
SELECT DISTINCT director FROM movies ORDER BY director;
SELECT * FROM movies ORDER BY title LIMIT 5 OFFSET 5;

SELECT * FROM movies INNER JOIN boxoffice ON movies.id = boxoffice.movie_id WHERE boxoffice.international_sales > boxoffice.domestic_sales;
SELECT * FROM movies INNER JOIN boxoffice ON movies.id = boxoffice.movie_id ORDER BY boxoffice.rating DESC;

# find all buildings that have employees
SELECT distinct(building_name) FROM buildings join employees on buildings.building_name = employees.building;

# list all buildings and distinct employee roles including empty buildings
SELECT distinct building_name, role FROM buildings left join employees on buildings.building_name = employees.building;

# find all employees not assigned a building
SELECT name, role FROM employees WHERE building IS NULL;

# find all buildings that hold no employees
SELECT building_name FROM buildings left join employees on employees.building = buildings.building_name where building is null;

# find the average years employed for each role
SELECT *, AVG(years_employed) FROM employees group by role;

# find the number of employees for each role
SELECT role, COUNT(*) FROM employees group by role;

# find total sales for each director
SELECT director, sum(domestic_sales + international_sales) FROM movies JOIN boxoffice ON boxoffice.movie_id = movies.id group by Director;
```

### Join Types

Inner join - matches rows from first table with the second table to create a row with combined columns from both tables. Only contains data that exists in both tables. Default join type.

Left join - include row from first table regardless of whether a matching row is found in second table

Right join - include row from second table regardless of whether a matching row is found in first table

full join - include rows from both tables regardless of whether a matching row exists in either.

sometimes left/right/full join is written as left outer/right outer/full outer but that is equivalent to short form.

### Aggregates

COUNT/MIN/MAX/AVG/SUM

use GROUP by agg col to apply aggregate function to groups of data by grouping rows that haev the same value as the column specified. Group by is executed after the where clause. can using HAVING to filter grouped rows.

### Order of Execution

- FROM/JOIN
- WHERE
- GROUP BY
- HAVING
- SELECT
- DISTINCT
- ORDER BY
- LIMIT/OFFSET