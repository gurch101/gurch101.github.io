---
title: Software Architecture & Technology of Large-Scale Systems
date: 2022-05-29
description: Udemy course notes
category: summary
type: notes
---

Traditional three-tier architecture - frontend/web tier, application server/business logic, database server

### Performance

A measure of how fast or responsive a system is under a given workload and given hardware.

Every performance problem is the result of some queue building up somewhere. This queue builds up due to inefficient/slow processing, serial resource access (only one thing can access resource at a time ie low concurrency), or limited resource capacity.

Efficiency

- resource utilization (IO/CPU)
- logic (algorithms, db queries)
- storage (db schema/indices, data structures)
- caching

Concurrency

- hardware
- software

Capacity

- CPU
- RAM
- Disk
- Network

##### Performance Objectives

Minimize request-response latency

- affects single user experience
- wait/idle time + processing time

Maximize throughput

- rate of request processing
- affects number of users that a system can support
- decreasing latency and increasing capacity can increase throughput

To visualize performance, plot request time (X) vs number of requests (Y). If there is a positive skew, the tail latency indicates requests are being queued.

### Network transfer latency

- network hops between potentially distant/unreliable servers
- TCP overhead - connection creation request/acknowledgement (SYN/ACK) is a round trip
- SSL overhead - TCP overhead + SSL hello/key exchange (2 more round trips)

Approaches to minimize network latencies

- between client + web server (persistent connections - HTTP 1.1, static data caching)
- data format (gRPC/thrift for intranet/microservice communication)
- data compression
- SSL session caching
- session/data caching
- connection pooling

### Memory access latency

- GC execution
- Memory swaps when app memory exceeds physical memory
- finite buffer memory for db

Approaches to minimize memory latencies

- avoid memory bloat
- use smallest heap as possible to minimize GC runtime
- multiple smaller processes are better than one large process
- use a purpose built GC algorithm (real-time GC for server, blocking but more efficient GC for batch processing)
- maximize buffer memory, normalize data, consider computation over storage (if value can be computed, compute it rather than store it)

### Disk access latency

- logging
- database
- web content (static files)

Approaches to minimize disk latency

- web content caching (static files)
- async/batch logging (disadvantage - data loss)
- batch I/O
- query optimization
- db cache
- db indexes
- schema denormalization
- faster disk, using RAID to allow parallel reads

### CPU latency

- inefficient algorithms
- context switching - process state is saved/restored to/from memory/cpu when multiple processes are running on a computer. Also caused by I/O

Approaches to minimize CPU latency

- async/batch I/O
- use a single threaded model (nginx, node.js - one thread handles request/response, other threads do I/O)
- thread pool should be sized appropriately to number of cores
- run processes in their own virtual env to ensure each process has dedicated CPU

### Common Latency Costs

CPU latency - < 10 ns

mutex lock - 25 ns

main memory - 100 ns

1kb compression - 3 micro seconds

send 1kb over 1 gbps network - 10 micro seconds

read 4k randomly from SSD - 150 micro seconds

round trip within one datacenter - 500 micro seconds

read 1 mb sequentially from SSD - 1 ms

send packet from California to Netherlands and back - 150 ms

### Concurrency related latency

If you have work that can be executed in parallel, adding CPU/threads can increase throughput. For serially executed code, adding CPU/threads does not increase throughput. If there is a serial portion of code in your app, the limit for max throughput is the % of your code is parallel (log curve). This bottleneck will be reached with fewer CPUs with more serial code. Locks on shared resources introduce serially executed code.

##### Amdahl's law for concurrent tasks

C(N) = N/(1 + alpha(N-1))

C is capacity, N is scaling dimension (CPU or load), alpha is resource contention (0 for linear performance). Serial code/queueing limits scalability.

##### Gunther's Universal Scalability Law

C(N) = N/(1 + alpha(N-1) + betaN(N-1))

beta = coherence delay - volatile variables updated in one thread leads to updates across all threads. If you increase processes/CPU, the throughput can actually decrease.

### Shared Resource Contention

- Listen/accept network queue on web server
- Thread pool is limited by number of processors. Can lead to context switching. Throughput decreases if thread pool size is too low or too high. If threads are sitting around waiting, you can increase the size. If requests are fast, thread pool size should be the same number of CPUs. Generally slightly bigger than num cores to account for idle time.
- Threads get a connection from the connection pool. One thread takes one connection.
- CPU/disk/network contention
- locks around serial parts of code

Approaches to minimize shared resource contention

- vertical scaling
- RAID to allow parallel access to disk
- find optimal thread pool size
- minimize lock scope

Approaches to minimize lock contention

- reduce the duration the lock is held. Move out code that doesnt require a lock (don't log inside a lock scope), split locks into lower granularity, split locks for each partition of data
- replace exclusive locks with coordination mechanisms (readwritelock/stamped locks or atomic variables)

### Pessimistic Locking

- Threads must wait to acquire a lock
- Lock is held for a long time
- Used when contention is high
- May result in deadlocks

```java
connection.setAutoCommit(false);
connection.statement.executeQuery("SELECT * FROM inventory WHERE product_id = 'xyz' FOR UPDATE");

// do work

connection.statement.executeUpdate("UPDATE inventory SET quantity=499 WHERE product_id = 'xyz'");

connection.commit();
```

### Optimistic Locking

- Threads do not wait for a lock
- Threads backup when they discover contention
- Used when contention is low
- May result in starvation if many threads need to retry

```java
connection.statement.executeQuery("SELECT * FROM inventory WHERE product_id = 'xyz'");

// do work

connection.setAutoCommit(false);
boolean success = connection.statement.executeUpdate("UPDATE inventory SET quantity=(quantity - 1) WHERE product_id = 'xyz' AND (quantity - 1) >= 0");

if(!success) {
  // stale data - retry
} else {
  connection.commit();
}
```

### Compare & Swap

CAS is an optimistic locking mechanism (implemented through java.util.concurrent.atomic.\* in java). Also in nosql databases. Good performance because it is generally implemented at the hardware level.

```java
AtomicInteger ai = new AtomicInteger(100);

// returns true if the value was 100 and sets 200 as the new value
// returns false if the value was not 100 as a result of a race condition with another thread
ai.compareAndSet(100, 200);
```

### Deadlocks

- Lock ordering related. Ex simultaneous money transfer from accounts X and Y by threads t1 and t2. T1 from X to Y and T2 from Y to X. Acquire locks in a fixed global order.
- Request load related. Threads waiting for connections to multiple databases or threads waiting for other threads to be spawned to perform some work. Ex API gateway with 10 threads receives 10 requests which call service 1. Service 1 in turn uses the gateway to call service 2 but can't because threads are in use.

### Coherence Delays

java guarantees that a volatile object is always read from main memory into CPU cache and written back to main memory when updated in a processor. Value in a CPU cache isn't visible by other CPU thread (two L1 caches).

all variables accessed inside a synchronized block are read from main memory at the start of the sync block and flushed to main memory when the associated thread exits the sync block. Changes to the value marks the CPU cache value as dirty and forces reading from main memory.

synchronized = locking + visibility

volatile = visibility

### Caching

- db cache: buffer/page cache
- http cache: cache in the browser itself
- web content cache: cache static assets
- SSL session caching
- session caching: user-specific data
- object cache: prevent db query

##### Static Data Caching

- GET method responses are idempotent and are good candidates for caching
- headers:
  - cache-control
    - no-cache: don't cache without validating with origin server
    - must-revalidate: validate only after max-age
    - no-store: don't cache at all
    - public: any shared cache can cache
    - private: only a client cache can cache
    - max-age: max age of a resource in cache relative to resource request time
  - etag: hash code that indicates version of a resource

##### Dynamic Data Caching

- exclusive (local) cache. Low latency but can lead to duplication (same data cached on multiple nodes) and uneven load balancing (intelligent routing based on user/session).
- shared cache. Higher latency due to an extra hop but can scale out and handle large datasets.

##### Caching Challenges

- limited cache space results in early evictions. Only cache small, frequently accessed, rarely changing objects
- cache invalidation & inconsistency. Requires update/deletion of cached value upon update. High TTL results in more cache hits but inconsistency increases as interval increases. Low TTL decreases inconsistency interval but cache hits go down.

### Scalability

Performance is focused on fixed load. Goal is to minimize latency and maximize throughput.

Scalability is focused on variable load. Goal is to maximize throughput.

Vertical scalability: increase resources on current hardware. Easier to achieve but it is limited scalability.

Horizontal scalability: increase hardware of the same specifications. Harder to achieve but offers unlimited scalability. Lets you scale down/repurpose hardware.

Reverse proxy: client only knows the address of the reverse proxy. The reverse proxy acts as a load balancer.

Have more specialized, modular components/workers/threads/instances that can work independently of one another. Independence is impeded by shared resources/data.

###### Stateful replication

- sticky sessions where user data is stored in the web app node. Cookie pins requests to a specific node through a cookie. Requests are routed to the appropriate node.
- if node goes down, requests are routed to other nodes but latency will be high since data needs to be fetched from db.
- has lower latency vs stateless replication at the expense of lower throughput.

###### Stateless replication

- higher scalability at the expense of higher latency. Store sessions in a shared cache (redis/memcached) or in cookies.

###### Database replication

- increase read scalability by adding read replicas
- allows for high availability
- Primary/secondary setup:
  - async: low latency writes, eventually consistent, can be data loss. Good for read replicas.
  - sync: consistent, high latency writes, low write availability. If primary goes down, you can't write at all. Good for backup.
- Peer-to-peer: async, highly available, has write conflicts/transaction ordering issues. Bidirectional replication. Can be used for global data replication.

##### App server scalability

- use specialized modules that run as microservices
- scale services that are under higher load indpendently of others
- microservices increases complexity - deployments, coordinated changes
- reduce request/response times by leveraging async services that communicate via mq. Async also be used to control db load by requiring infra for average load as opposed to peak load. MQ simply queues up backlog during peak periods. Reduces write load.
- cache frequently read, infrequently changed data into an object cache. Reduces read load on db.
- Can use separate DBs for each service to increase scalability. Can no longerdo interservice ACID transaction and need to deal with eventual consistency.

##### Database Partitioning

Vertical partitioning - split db by domains/services

Horizontal partitioning - split db by range/hash on id (id 1..100 goes to node 1, hash(id) % N used to compute node). Usually done with a nosql database.

- range partitioning supports range queries (id > 1 and id < 100 - goes to multiple nodes if range spans nodes) and exact matches. Hash partitioning only supports exact matches. Hash partitioning is generally faster.
- works through cluster-aware client/router/proxy libraries
  Disadvantages - can no longer do ACID transactions across db nodes. Data will be eventually consistent.

### Load Balancing

Provides a single IP address for a component and distributes calls to underlying service instances.

Discovery service is a registry for IPs of healthy instances. The API gateway service queries the discovery service to get the IP of the app instance based on load.

External clients use DNS to discover the external load balancer. Internal clients use a local registry/config to discover an internal load balancer.

Hardware LB can handle L4 (transport layer - TCP/UDP) and L7 (application layer - HTTP/FTP/SMTP) traffic and can handle many more connections. Software load balancers can only handle L7, fewer connections, but are free.

L7 LBs act as a reverse proxy (takes a connection, creates connections to underlying services), route based on content (mime type), SSL termination.

DNS can act as a load balancer where one DNS record has multiple A records. Used for routing in multi-region systems (GeoDNS) to ensure user is routed to closest datacenter. DNS does health checks against each region for HA.

### Auto Scaling

1. Configure load thresholds
2. Monitor CPU/network/disk load and health
3. Launch new instances from VM/container image and assign IP
4. Register new instance with LB

### Service-Oriented Architecture

each service can have its own tech stack and can be scaled independently but shared a common interface and database schema.

### Microservices

goal is high scalability through independence, frequent/independent development and deployment

- shared nothing architecture. Services are fully vertically partitioned and developed/deployed independently
- db/schema are independent
- service interface is loosely coupled through an ETL layer
- challenges: duplicate code, transaction failures, transaction rollbacks

##### Microservice Transactions

Since transactions involve multiple machines, local transactions are not possible.

- distributed ACID transactions (2 or 3 phase commit). Does not provide scalability since locks/transactions are open for a long time.
  - 1. coordinator service creates transaction in each service
  - 2. coordinator service commits transaction in each service
- compensating transactions - SAGA pattern; eventually consistent

ACID within a service, compensating transactions across services

##### SAGA Pattern

- logical undo of partially committed transactions

Example:
Order, inventory, and shipment services

1. Create Order
2. Reserve Inventory
3. Create Shipment

undo service actions on failure of any step. Need to handle cases where undo of any step fails. Sagas are executed asynchronously. Atomic, eventually consistent, not isolated, durable.

### Microservice Communication Model

- synchronous processing - immediate response; for read/query loads
- async processing - deferred response; for write/transaction loads; higher scalability/reliability

### Event-Driven Transactions

Order request goes to order orchestrator which add events onto an MQ

Insert events on MQ. Orchestrator waits until each response event is published to MQ before sending proceeding:
Create Order Event (orchestrator)/Order Created Event(Order service)
Reserve Inventory Event (orchestrator)/Inventory Reserved Event (Inventory service)
Create Shipment Event (orchestrator)/Shipment Created Event (Shipment service)

Service tracks whether it sent out a created event in case the node goes down after the action is done but before event is sent out. When node is back online, it polls its db to see which actions were done where events weren't sent out.

If any step fails, add an "undo \<X\> event" to

The MQ can be horizontally partitioned by topic

### TODO

Read Amdahl's law wiki page
