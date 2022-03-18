# SQL

```sql
select * from movies;
select title, year from movies where year <= 2010;
select * from movies where title like "Toy Story%" # begins query
select * from movies where title like "Toy Story _" # any single character
SELECT DISTINCT director FROM movies ORDER BY director;
```