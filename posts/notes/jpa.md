---
title: Hibernate and JPA
date: 2022-05-29
description: Udemy course notes
category: summary
type: notes
---

JDBC required writing your own query, connection management, and mapping logic. In spring boot, `JdbcTemplate` with `BeanPropertyRowMapper` does all of this.

The Java Persistence API uses annotated entity classes mapped to your database to automatically write queries for you, jpql, criteria queries, or native queries. Hibernate is the most popular implementation of JPA.

JPA buddy can autogenerate migrations from entities and provide diffs between migrations + entities. Also can help manage fields on entities. Also can generate query names for spring data repository methods and extract JPQL from method names.

### Spring Data JDBC

```java
public class PersonDao {
  @Autowired
  private final JdbcTemplate jdbcTemplate;

  public List<Person> findAll() {
    return jdbcTemplate.query("select * from person", new BeanPropertyRowMapper<Person>(Person.class));
  }

  public Person findById(int id) {
    return jdbcTemplate.queryForObject("select * from person where id=?", new BeanPropertyRowMapper<Person>(Person.class), id);
  }

  public int deleteById(int id) {
    return jdbcTemplate.update("delete from person where id=?", id);
  }

  public int insert(Person person) {
    return jdbcTemplate.update("insert into person (id, name, location, birth_date) VALUES(?, ?, ?, ?)", new Object[] { person.getId(), person.getName(), person.getLocation(), new Timestamp(person.getBirthDate().getTime())});
  }

  public int update(Person person) {
    return jdbcTemplate.update("update  person set name=?, location=?, birth_date=? where id = ?", new Object[] { person.getName(), person.getLocation(), new Timestamp(person.getBirthDate().getTime()), person.getId()})
  }
}
```

### Hibernate and JPA

Entity

```java
@Entity // hibernate triggers a schema update by default. CREATE TABLE not needed via schema.sql
@Table(name="people") // only needed if entity name doesnt match db table name - provided name is lower case + snake case
@NoArgsConstructor
@NamedQuery(name = "find_all_persons", query = "select p from Person p")
public class Person {
  @Id
  @GeneratedValue // adds a sequence to the db
  private int id;

  // insertable = false means JPA won't send this column on insert
  @Column(name="fullname", nullable = false, unique = true, insertable = false, updateable = false)
  private String name;
}
```

Repository

```java
@Repository
@Transactional // typically done at the service layer. Any methods that manipulate data require a transaction scope.
               // any JPA managed entity inside of a transaction is automatically synced at the end of the transaction (ie no need to call persist/merge)
public class PersonJpaRepository {
  @PersistenceContext
  EntityManager entityManager;

  public Person findById(int id) {
    return entityManager.find(Person.class, id);
  }

  public Person upsert(Person person) {
    // can be used to update or merge - if there is an id, it will update, else insert - issues a select followed by an insert/update
    return entityManager.merge(person);
  }

  public Person save(Person person) {
    if(person.getId() == null) {
      entityManager.persist(person); // persist only assigns an id to the user, it isn't committed until the transaction is committed on method complete
    } else {
      entityManager.merge(person);
    }
    return person;
  }

  public Person deleteById(int id) {
    Person person = findById(id);
    return entityManager.remove(person);
  }

  public List<Person> findAll() {
    TypedQuery<Person> namedQuery = entityManager.createNamedQuery("find_all_persons", Person.class);
    return namedQuery.getResultList();
  }

  public List<Person> findAllJPQL() {
    return entityManager.createQuery("SELECT p from Person p", Person.class).getResultList();
  }

  public List<Person> findAllNative() {
    return entityManager.createNativeQuery("SELECT * from person", Person.class).getResultList();
  }
}
```

application.properties:

```
# load data.sql AFTER jpa creates all tables
spring.jpa.defer-datasource-initialization=true
```

### Entity Manager

```java
em.flush() // synchronize all changes with db - requires a transaction; a flush can still be rolled back on failure
em.clear() // make all managed entities unmanaged - any changes made to entity object are not synchronized after clear is called
em.detach(entity) // detach specific entity
em.refresh(entity) // repace entity contents with values in db
```

### JPQL

Query using entities instead of db tables.

```java
List<Course> courses = em.createQuery("SELECT c from Course c where name like '%foo'", Course.class).getResultList()
```

### Relationships

```java
@Entity
public class Course {
  @Id
  @GeneratedValue
  private Long id;

  @OneToMany(mappedBy = "course") // lazy fetch by default
  private List<Review> reviews = new ArrayList<>();

  @ManyToMany(mappedBy = "courses") // ensures 1 m2m table called student_courses is created. If mappedBy wasn't specified, you'd end up with two join tables.
  private List<Student> students = new ArrayList<>();
}

@Entity
public class Review {
  @Id
  @GeneratedValue
  private Long id;

  @ManyToOne // adds course_id field to Review table; eager fetched by default
  private Course course;
}

@Entity
public class Student {
  @Id
  @GeneratedValue
  private Long id;

  @OneToOne // the student table will have a passport_id field; by default this is nullable and eager fetched with a left outer join; if optional is false, then inner join
  private Passport passport;

  @ManyToMany
  @JoinTable(name = "student_course", joinColumns = @JoinColumn(name="student_id"), inverseJoinColumns = @JoinColumn(name="course_id"))
  private List<Course> courses = new ArrayList<>();
}

@Entity
public class Passport {
  @Id
  @GeneratedValue
  private Long id;

  @OneToOne(fetch=FetchType.LAZY, mappedBy="passport") // mappedBy ensures passport won't have a student_id field
  private Student student;
}
```

To add a `Review`:

```java
Course c = em.getReference(Course.class, <primaryKey>); // getReference doesn't actually execute a query; if it doesnt exist, the persist will through a referential integrity contraint exception
Review r = new Review();
r.setCourse(c);
em.persist(r);
```

To enroll a student in a course

```java
Course c = em.getReference(Course.class, <primaryKey>);
student.addCourse(c);
em.persist(student); // this triggers select * on all courses, delete on each course, the insert of each course
```

### Lazy Loading

set `fetch = FetchType.LAZY` on any association to lazy load it. Be sure to exclude it from `toString()`. If the entity is fetched from within a transaction, calling `getAssociation()` will fetch the association on-demand (ie a second query) as opposed to in a single query with a join. `*ToOne` is always eager fetched.

### Transactions

`@Transactional` creates a persistence context (all entities in scope are managed; lazy joins are fetched on-demand). The transaction is committed after the method completes. Calling `setField()` will be persisted without explicitly calling `persist()`. Transactions are required for inserts/updates. Read only methods need transactions for lazy joins (the entitymanager has a default transaction, but calling `getAssociation()` isn't using the entity manager - results in `LazyInitializationException`).

Atomicity - changes are all-or-nothing
Consistency - system remains in a consistent state (ie if you withdraw 50, you need to deposit 50 somewhere else)
Isolation - changes in one transaction don't impact another transaction
Durability - data is persisted/rolled back on failure

Dirty reads - one transaction reading a value that is modified by another incomplete transaction (account with 300, two transactions, t1 withdraws 100 (200), t2 withdraws 50 (150), t1 fails and rolls back to 300 even though t2 was successful)

Non-repeatable reads - reading the same record multiple times can result in different values if another transaction updates it

Phantom reads - running a range query can return a different number of records if another transaction inserts/deletes a record

Use springs transactional over javax transactional to ensure the transaction spans across multiple db's/mq's

##### Isolation levels

Read uncommitted - can read regardless of transaction commit - no locks - dirty reads, non-repeatable reads, and phantom reads are possible

Read committed - modified rows are locked - dirty reads not possible <- default for postgresql

Repeatable read - modified rows and read rows are locked - dirty reads and non-repeatable reads not possible <- default for mysql

Serializable - all records that match the where clause on a select are locked (including future records) - dirty reads, non-repeatable reads, and phantom reads are not possible

### Spring Data JPA

```java
// provides basic crud operations - JpaRepository extends Crud/SortPageable/QueryByExample Repos
public interface CourseRepository extends JpaRepository<Course, Long> {}
```

Sorting:

```java
Sort s = new Sort(Sort.Direction.DESC, "name");
repository.findAll(s);
```

Pagination:

```java
PageRequest pageRequest = PageRequest.of(<zero-based-page>, <size>);
Page<Course> firstPage = repository.findAll(pageRequest);
firstPage.getContent();
```

Custom queries:

add methods to the repository interface

```java
List<Course> findByName(String name);
int countByName(String name);
List<Course> findByNameAndDescription(String name, String description);
List<Course> findByNameOrderByIdDesc(String name);
List<Course> deleteByName(String name);

@Query("jpql or native query or named query")
List<Course> someCustomName(String namedParam1, String namedParam2);
```

### Spring Data REST

Add `@RepositoryRestResource(path="courses")` to the repository. Add `@JsonIgnore` to entity fields you don't want to expose/return.

### H2 Setup

in application.properties:

```
spring.datasource.url=jdbc:h2:mem:testdb
```

When running, goto `/h2-console` to connect to an admin panel

Spring boot will automatically run `src/main/resources/schema.sql` to create the schema for JPA-managed entites and `src/main/resources/data.sql` to populate it. If using a migration tool, do not use these files.

### Questions

How to remove h2 for prod build

jsonb for h2

spring data jdbc joins + transactions

why does jpa need to find entity before deleting it?

flush in a transaction that throws an exception? it undoes it

review join types

implication of collection type in java on jpa

if you add addToCollection method on entity, will it allow persisting the parent-child entity together in one save call?

m2m with data (studentcourse.grade) - how to get courses
