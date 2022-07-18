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

### Order of Execution

- FROM/JOIN
- WHERE
- GROUP BY
- HAVING
- SELECT
- DISTINCT
- ORDER BY
- LIMIT/OFFSET

### Creating Tables

```sql
CREATE TABLE cities (
  id SERIAL PRIMARY KEY
  name VARCHAR(50),
  country VARCHAR(50),
  population INTEGER,
  area INTEGER
);
```

### Inserting Data

```sql
INSERT INTO cities (name, country, population, area)
  VALUES
    ('Delhi', 'India', 2000000, 2240),
    ('Shanghai', 'China', 22125000, 4015);
```

### Querying Data

```sql
SELECT * FROM cities;
SELECT name, country FROM cities;
SELECT name, population / area AS population_density FROM cities;
SELECT name || ', ' || country AS location FROM cities; // equivalent to CONCAT(name, ', ', country)

SELECT name, area FROM cities WHERE area BETWEEN 2000 AND 4000;
SELECT name, area FROM cities WHERE name IN ('Delhi', 'Shanghai');
SELECT name, population / area AS population_density FROM cities WHERE population / area > 6000;
```

### String functions

```sql
CONCAT // can also use ||
LOWER
UPPER
LENGTH
```

### Comparison Operators

```sql
-- can add ALL/ANY modifier
=, >, <, >=, <=, <>, !=
IN
NOT IN
BETWEEN X AND Y
```

### Updating Records

```sql
UPDATE cities SET population = 39505000 WHERE name = 'Tokyo';
```

### Deleting Rows

```sql
DELETE FROM cities WHERE name = 'Tokyo';
```

### Relationships

a user has many photos (one-to-many). Many photos belong to a user (many-to-one).

A company has one CEO. A CEO works at one company (one-to-one).

Students are enrolled in many classes. Classes have many students (many-to-many).

The "many" side of the relationship gets the foreign key column

```sql
CREATE TABLE users (
  id SERIAL PRIMARY KEY, -- postgres will autogenerate serial; doesn't need to be specified on INSERT
  username VARCHAR(50)
);

INSERT INTO users (username) VALUES
  ('foo'),
  ('bar');

CREATE TABLE photos (
  id SERIAL PRIMARY KEY, -- 1 to 2^31, auto increments
  url VARCHAR(200),
  user_id INTEGER REFERENCES users(id) -- attempting to insert a record with an invalid user_id will trigger fkey violation constraint
                                       -- NULL is allowed by default
                                       -- user_id INTEGER REFERENCES users(id) ON DELETE CASCADE - if user is deleted, photo associated with user is also deleted
);

INSERT INTO photos (url, user_id)
  VALUES('http://example.com', 1);

SELECT * from photos JOIN users ON users.id = photos.user_id;
```

### Delete Constraints

`ON DELETE RESTRICT` prevent deletion if fkey columns reference record. Default behavior.

`ON DELETE NO ACTION` prevent deletion if fkey columns reference record

`ON DELETE CASCADE` delete records that have fkey relationship too

`ON DELETE SET NULL`

`ON DELETE SET DEFAULT`

### Joins

```sql
-- find all comment author names and comment content
SELECT contents, username
FROM comments
JOIN users ON users.id = comments.user_id;

-- find all comment contents and photo urls associated with the comment
SELECT comments.id AS comment_id, contents, photo_url
FROM comments
JOIN photos p ON p.id = comments.photo_id;

-- find all users that comment on their own photos
SELECT username
FROM users
JOIN photos ON photos.user_id = users.id
JOIN comments ON comment.photo_id = photos.id AND comments.user_id = users.id;
```

FROM/JOIN order matters for left/right joins.

### Join Types

Inner join - matches rows from first table with the second table to create a row with combined columns from both tables. Only contains data that exists in both tables. Default join type.

Left join - include row from first table regardless of whether a matching row is found in second table

Right join - include row from second table regardless of whether a matching row is found in first table

full join - include rows from both tables regardless of whether a matching row exists in either.

sometimes left/right/full join is written as left outer/right outer/full outer but that is equivalent to short form.

### Aggregates

COUNT/MIN/MAX/AVG/SUM

use GROUP BY agg col to apply aggregate function to groups of data by grouping rows that have the same value as the column specified. Group by is executed after the where clause. can using HAVING to filter grouped rows.

null values are not included in COUNT. To include them, use COUNT(\*).

```sql
-- returns all unique user_ids
-- can only select grouped columns or aggregate values
SELECT user_id
FROM comments
GROUP BY comments.user_id;

-- returns number of comments a user has made
-- can apply aggregate function within a group
SELECT user_id, COUNT(*) AS num_comments_created
FROM comments
GROUP BY comments.user_id;

SELECT authors.name, COUNT(*)
FROM authors
JOIN books ON books.author_id = authors.id
GROUP BY authors.name;

-- find all photos with more than 2 comments where the photo_id < 3
SELECT photo_id, COUNT(*)
FROM comments
WHERE photo_id < 3
GROUP BY comments.photo_id HAVING COUNT(*) > 2;

-- find all manufacturers that sold more than 200000 total revenue
SELECT manufacturer, SUM(price * units_sold)
FROM phones
GROUP BY manfacturer HAVING SUM(price * units_sold) > 200000;
```

### Sorting

default is ascending. Add DESC to make descending.

```sql
-- sort by ascending price and descending weight
SELECT *
FROM products
ORDER BY price, weight DESC;
```

### Offset and Limit

Offset = skip the first N rows in the result set

Limit = return N records from result set

```sql
-- get the names of the second and third most expensive phones
SELECT name
FROM phones
ORDER BY price DESC
LIMIT 2
OFFSET 1;
```

### Unions and Intersections

UNION
UNION ALL
INTERSECT
INTERSECT ALL
EXCEPT
EXCEPT ALL

set methods require same columns in all queries

```sql
-- get the 4 most expensive products and the 4 highest price / weight products
-- if you want to get dups, do UNION ALL
(
  SELECT *
  FROM products
  ORDER BY price DESC
  LIMIT 4
)
UNION
-- INTERSECT will return products that are in the top 4 most expensive AND in the top 4 highest price / weight products
-- EXCEPT will return products that are in the top 4 most expensive but not in the top 4 highest price / weight products
(
  SELECT *
  FROM products
  ORDER BY price / weight DESC
  LIMIT 4
)
```

### Subqueries

inner query is executed first. Always think about shape of data.

Subquery in select statement should return a single value

```sql
-- list name of products that are more expensive than all products in the toys department
SELECT name
FROM products
WHERE price > (SELECT MAX(price) FROM products WHERE department = 'toys');

-- list the name, price, and price ratio relative to the most expensive product
SELECT name, price, price / (SELECT MAX(price) FROM products) AS max_price_ratio FROM products WHERE price > 867;

-- subquery in from clause must have alias
SELECT name, price_weight_ratio FROM (SELECT name, price / weight as price_weight_ratio FROM product) AS p WHERE price_weight_ratio > 5;

-- subquery in from clause can also return a single value
SELECT * FROM (SELECT MAX(price) as max_price FROM products) as p;

-- get average number of orders for all users
SELECT AVG(ct) FROM (SELECT COUNT(*) AS ct FROM orders GROUP BY user_id) AS o;

-- get the highest average phone price by manufacturer
SELECT MAX(avg_prc) AS max_average_price
FROM (
  SELECT AVG(price) as avg_prc
  FROM phones
  GROUP BY manufacturer
);

-- get the first name of all users that ordered a specific product
SELECT first_name
FROM users
JOIN (
  SELECT user_id FROM orders WHERE product_id = 3
) AS o
ON o.user_id = users.id;

-- get all orders that involve a product with price/weight ratio > 5
SELECT id
FROM orders
WHERE product_id IN (SELECT id FROM products WHERE price / weight > 5);

-- get all products that have a price greater than the average price of all products
SELECT name, price
FROM products
WHERE price > (
  SELECT AVG(price) FROM products
);

-- get name of products that are more expensive than all products in the 'Industrial' department
SELECT name
FROM products
WHERE price > ALL (SELECT price FROM products WHERE department = 'Industrial');

--- get name, department, and price of the most expensive product in each department
SELECT name, department, price
FROM products AS p1
WHERE price = (SELECT MAX(price) FROM products AS p2 WHERE p2.department = p1.department);

-- get number of orders for each product; join/group by would be more efficient
SELECT name, (SELECT COUNT(*) FROM orders AS o WHERE o.product_id = p1.id) AS num_orders
FROM products AS p1

-- get ratio of max price and avg price
SELECT (SELECT MAX(price) FROM products) / (SELECT AVG(price) FROM products);
```

### Distinct

```sql
-- get all unique departments
SELECT DISTINCT department FROM products;

-- get all unique department, product name pairs
SELECT DISTINCT department, name FROM products;

-- get number of unique departments
SELECT COUNT(DISTINCT department) FROM products;
```

### Utility Functions

```sql
-- get the shipping cost of a product where the cost is the larger of the weight * $2 or $30
SELECT name, weight, GREATEST(weight * 2, 30) FROM products;

-- get the sale price of a product
SELECT name, LEAST(price * 0.5, 400) FROM products;

SELECT name, price,
  CASE
    WHEN price > 600 THEN 'high'
    WHEN price > 300 THEN 'medium'
    ELSE 'cheap' -- if no ELSE, default is NULL
  END
FROM products;
```

### Data Types

data type categories: numbers, currency, binary, date/time, character, json, geometric, range, arrays, boolean, xml, uuid

Rules:

- id column should be SERIAL
- number without decimal - use INTEGER
- precise amounts with decimal - use NUMERIC
- numbers with decimal - use DOUBLE PRECISION

```sql
-- can cast with ::datatype
SELECT (3.14::INTEGER)
-- first three characters
SELECT ('fooobar'::VARCHAR(3))

SELECT ('NOV-20-1980 1:24 AM EST'::TIMESTAMP WITH TIME ZONE) - ('1 day'::INTERVAL);
```

CHAR(N) - store N characters; pad with space
VARCHAR - store arbitrary length string
VARCHAR(N) - store up to N characters, no space
TEXT - store arbitrary length string

boolean type:
true, 'yes', 'on', 1, 't', 'y' evaluate to true
false 'no', 'off', 0, 'f', 'n' evaluate to false
null

date/time
can provide almost any date/time format string and store it as yyyy-mm-dd and HH:MM:SS

DATE
TIME (WITH/WITHOUT TIME ZONE)
TIMESTAMP (WITH/WITHOUT TIME ZONE)
INTERVAL

### Validations

NULL

```sql
-- in create table
CREATE TABLE products(
  id SERIAL PRIMARY KEY
  price INTEGER NOT NULL
);

-- after table created; you need to ensure current state of table meets constraint
ALTER TABLE products
ALTER COLUMN price
SET NOT NULL;
```

DEFAULT

```sql
-- in create table
CREATE TABLE products(
  id SERIAL PRIMARY KEY
  price INTEGER DEFAULT 999
);

-- after table created
ALTER TABLE products
ALTER COLUMN price
SET DEFAULT 999;
```

UNIQUE

```sql
-- in create table
CREATE TABLE products(
  id SERIAL PRIMARY KEY
  name VARCHAR(30) UNIQUE
);

-- after table created; you need to ensure current state of table meets constraint
ALTER TABLE products
ADD UNIQUE (name);


ALTER TABLE products
DROP CONSTRAINT products_name_key;

-- in create table
CREATE TABLE products(
  id SERIAL PRIMARY KEY
  name VARCHAR(30),
  department VARCHAR(30),
  UNIQUE(name, department)
);

ALTER TABLE products
ADD UNIQUE (name, department);
```

comparison validation

```sql
-- in create table
CREATE TABLE products(
  id SERIAL PRIMARY KEY
  price INTEGER CHECK (price > 0)
);

-- after table created; you need to ensure current state of table meets constraint
ALTER TABLE products
ADD CHECK (price > 0);


CREATE TABLE orders(
  id SERIAL PRIMARY KEY,
  created_at TIMESTAMP NOT NULL,
  est_delivery TIMESTAMP NOT NULL,
  CHECK (created_at < est_delivery)
);

-- check for enum could be CHECK (color IN ('red', 'green', 'blue'))
```

### Instagram schema

```sql
CREATE TABLE users(
  id SERIAL PRIMARY KEY,
  created_at TIMESTAMP,
  updated_at TIMESTAMP,
  username VARCHAR(30)
);

CREATE TABLE posts (
  id SERIAL PRIMARY KEY
  created_at TIMESTAMP,
  updated_at TIMESTAMP,
  url VARCHAR(200),
  user_id INTEGER REFERENCES users(id)
);

CREATE TABLE comments (
  id SERIAL PRIMARY KEY
  created_at TIMESTAMP,
  updated_at TIMESTAMP,
  contents VARCHAR(240),
  post_id INTEGER REFERENCES posts(id),
  user_id INTEGER REFERENCES users(id)
);

CREATE TABLE post_likes (
  id SERIAL PRIMARY KEY
  user_id INTEGER REFERENCES users(id),
  post_id INTEGER REFERENCES posts(id),
  UNIQUE(user_id, post_id)
);

-- number of likes on a specific post
SELECT COUNT(*) FROM post_likes WHERE post_id = 3;

-- get the post ids of the top 5 liked posts
SELECT posts.id FROM posts JOIN post_likes ON post_likes.post_id = posts.id GROUP BY posts.id ORDER BY count(*) DESC LIMIT 5;

-- get the username of people whole like a specific post
SELECT username FROM users JOIN post_likes ON post_likes.user_id = users.id WHERE post_likes.post_id = 3;

-- get the urls of posts that user with a specific id liked
SELECT url FROM posts JOIN post_likes ON post_likes.post_id = posts.id WHERE post_likes.user_id = 4;

-- Polymorphic association. Generic like table for posts and comments (no fk; not recommended)
CREATE TABLE likes(
  id SERIAL PRIMARY KEY
  user_id INTEGER REFERENCES users(id),
  like_id INTEGER -- no foreign key
  liked_type VARCHAR(30), -- post/comment
  UNIQUE(user_id, post_id)
);

-- Polymorphic association alternative
CREATE TABLE likes(
  id SERIAL PRIMARY KEY
  user_id INTEGER REFERENCES users(id),
  post_id INTEGER REFERENCES posts(id),
  comment_id INTEGER REFERENCES comments(id),
  -- coalesce returns the first value that is not null
  CHECK (COALESCE((post_id)::BOOLEAN::INTEGER, 0) + COALESCE((comment_id)::BOOLEAN::INTEGER, 0)) = 1
);

-- simplest alternative is to create a separate table for each like type. Would require UNION queries for analytics
```
