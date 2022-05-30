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
