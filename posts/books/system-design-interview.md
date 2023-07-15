---
title: System Design Interview
date: 2021-10-10
description: System Design Interview Summary
category: book summary
type: books
---

### Chapter 1: Scaling from 1 to a million users

- start with a single server to host static assets, app server, database server
- vertically scale until you can't - simple but has hard limits and doesn't provide failover/redundancy
- split db and app server so they can be scaled independently
  - nosql = low latency, unstructured, massive amounts of data
- horizontally scale - use a load balancer (lb has 1 public ip and N private ip's)
  - improved availability
  - stateful architecture requires sticky sessions - harder to scale/add servers/deal with failures
  - stateless - more scalable -> store user state in redis/nosql db/relational db
- failover/redundancy - db replication with one master db for writes and many read nodes
  - most systems have more writes vs reads
  - better performance because more reads can happen in parallel
  - nodes can be added/promoted to ensure high availability
- add a cache for data that is read frequently but updated infrequently or data that is expensive to compute
  - read-through: read from cache first, if miss then read from db and update cache
  - write-through: write to both cache and db
  - considerations: expiration, what to cache, consistency, eviction (what to do when full - LRU)
- use a CDN
  - considerations: cost (only cache frequently used content), expiry, failover, invalidation
- use multiple data centers and a geoDNS to deal with international users and improve availability - routes to server closest to user
  - considerations: automated deployments, monitoring, replication, immutable infrastructure
- use message queues for long running tasks that can be done asynchronously
  - producer can publish messages even when there are no consumers, consumers can read messages even when there are no producers
  - consumers can be scaled based on workload
- sharding the db to scale horizontally
  - considerations: resharding data when shards need to be split further, uneven data distribution, celebrity (hotspot key) problem (certain users are more popular/active than others), leads to denormalization to prevent joins
- split tiers into separate services

![System Architecture](/images/sysdesign1.png)

### Chapter 2: Back-Of-The-Envelope Estimation

2^10 = 1024 = 1 kb - thousands
2^20 = 1048576 = 1 mb - millions
2^30 = 1gb - billions
2^40 = 1 tb - trillions
2^50 = 1 pb - quadrillions

• Memory is fast but the disk is slow.
• Avoid disk seeks if possible.
• Simple compression algorithms are fast.
• Compress data before sending it over the internet if possible.
• Data centers are usually in different regions, and it takes time to send data between them.

99% availability = down 3 days/year
99.9% availability = down 8 hours/year

Example: Estimate Twitter QPS and storage requirements
Please note the following numbers are for this exercise only as they are not real numbers
from Twitter.
Assumptions:
• 300 million monthly active users.
• 50% of users use Twitter daily.
• Users post 2 tweets per day on average.
• 10% of tweets contain media.
• Data is stored for 5 years.
Estimations:
Query per second (QPS) estimate:
• Daily active users (DAU) = 300 million _ 50% = 150 million
• Tweets QPS = 150 million _ 2 tweets / 24 hour / 3600 seconds = ~3500
• Peek QPS = 2 _ QPS = ~7000
We will only estimate media storage here.
• Average tweet size:
• tweet_id 64 bytes
• text
140 bytes
• media
1 MB
• Media storage: 150 million _ 2 _ 10% _ 1 MB = 30 TB per day
• 5-year media storage: 30 TB _ 365 _ 5 = ~55 PB

### Chapter 3: A Framework For System Design Interviews

- deliberately ambiguous
- purpose is to demonstrate design skill, defend design choices, and respond to feedback
- interviewer is looking for ability to collaborate, resolve ambiguity, work under pressure

step 1: understand the problem and establish design scope

web/mobile/both?
what features do we need to build?
what are the goals?
how many users does the product have?
how fast does the company anticipate the scale up?
what is the tech stack?
what existing services can you leverage to simplify the design

step 2: propose high-level design

ask for feedback - treat interviewer as a teammate
think out loud
go through use cases
sketch out lines and boxes

step 3: design deep dive

ask about which areas to prioritize

step 4: answer questions/wrap up

identify bottlenecks, potential improvements
talk about errors - server failure, network loss
monitoring/metrics/roll out

### Chapter 4: Design a Rate Limiter

Used to control rate of traffic sent by a client or service by limit the number of client requests allowed to be sent over a specified period.

Goals: prevent DOS, reduce cost

Step 1: Understand the problem and establish design scope.

client side or server side?
throttle mechanism - IP, user id, other properties? system wide?
scale? distributed?
different limiters for different endpoints?

Step 2: Propose high-level design

Start with client/server, then add separate services.

Algorithms:

- Token bucket algorithm: put tokens into a bucket periodically up to some capacity. Remove a token for each request.

  - pros: easy to implement, memory efficient, allows bursty traffic. Cons: difficult to tune bucket size/refill rate

- Leaking bucket algorithm: put requests on a queue - if queue is full, drop request, else process request at a fixed rate

  - pros: memory efficient, good for cases where a stable outflow rate is needed
  - cons: not good for bursty traffic

- Fixed window counter algorithm: divide time into fix-sized time windows and assign a counter to each window. Increment counter by one for each request. Once the counter reaches a threshold, drop requests until new window starts.

- Sliding window algorithm: remove all timestamps older than start of current time window, add timestamp for new request, if the log size is the same or lower than allowed count, accept request.
  - pros: very accurate, cons: consumes memory because timestamps are kept
- redis has INCR/EXPIRE to increment and timeout a counter.

Step 3: Design Deep-Dive

maintain rules in configuration files, return HTTP 429 if rate is exceeded. Return headers to clients to provide number of remaining requests within windows. Possibly queue excess requests for later processing on message queue.

Possible problems

- race conditions when reading/updating counters - use locks
- synchronization issues in distributed environments where multiple caches are involved. Use sticky sessions or centralized data stores.
- performance - make service geographically distrubted and route to nearest server. Synchronize with eventual consistency.
- monitoring - gather analytics data to check whether limiter is effective

Step 4: Wrap up

- hard vs soft rate limiting where you allow some requests to exceed threshold for a short period
- rate limit via iptables instead of app level
- design client with best practices - use client cache, graceful recovery from exceptions

### Chapter 5: Design Consistent Hashing

If you have n cache servers, a common way to balance the load is to use the following hash method:

```serverIndex = hash(key) % N, where N is the size of the server pool

```

Problem - if number of servers change, we get different server indices

Consistent hashing: when hash table is resized, only k/n keys need to be remapped where k is the number of keys, and n is the number of slots. In contrast, in most traditional hash tables, a change in the number of array slots causes nearly all keys to be remapped.

map servers and keys on to a ring using a uniformly distributed hash function. To find out which server a key is mapped to, go clockwise from the key position until the first server on the ring is found.

problems: partition sizes can be different when servers are added/removed. Key distribution can be non-uniform.

Solutions:

- represent each server with multiple virtual nodes so that each server is responsible for multiple partitions. More virtual nodes = more balanced distribution but this requires more storage to store data about each virtual node.
- new node: redistribute data by going anti-clockwise from new node until a server is found.
- removed node: go anti-clockwise from removed node until server is found and redistribute to next node

used to deal with celebrity problem by making sure celebrity data is more evenly distributed.

### Chapter 6: Design a Key-Value Store

Understand the problem - read/write/memory usage, size of key-value pair, HA, automatic scaling

Single server - use an in-memory hash table - optimize by compressing data and only storing frequently used data in memory and the rest on disk. Limitations on capacity.

distributed hash table:
CAP theorem: System can only have two of the following.

- Consistency: all clients see the same data at the same time no matter which node they connect to
- Availability: any client which requests data gets a response even if some of the nodes are down
- Partition Tolerance: System continues to operate despite network partition (three nodes - data is written to two but not replicated to third, data written to third isn't replicated to other two)

If consistency is chosen over availability, must block all write operations to avoid data inconsistency.

if availability is chosen over consistency, the system keeps accepting reads even though it might return stale data and n1/n2 keep accepting writes. Writes will be replicated to n3 once its back up.

Data is partitioned using consistent hashing using a hash ring. Pros: automatic scaling depending on load, the number of virtual nodes for a server is proportional to the server capacity.

Data is replicated asynchronously over N servers - chose first N servers from key placement.

Quorum consistency:

- For a write operation to be considered as successful, write operation must be acknowledged by W replicas
- For a read operation, read operations must wait for responses from R replicas.

Configuration of W, R, N is a tradeoff between latency and consistency.

If R = 1, W = N, system optimized for fast reads
If W = 1, R = N, system optimized for fast writes
Strong consistency is guaranteed if W + R > N (usually N = 3, W = R = 2).

Strong consistency - reads always return most up-to-date val (force replica to not accept new reads/writes until every replica has agreed on current write).
Weak consistency - reads might not always return up-to-date val.
Eventual consistency - Given enough time, all updates are propagated and all replicas are consistent.

Inconsistency resolution using versioning
vector clock - [server, version] pair associated with a data item that can be used to check if one version precedes, succeeds, or is in conflict with others.
When data item D is written to server Si, vi needs to be incremented in [Si, vi] or [Si, 1] needs to be created

Failure handling

- need at least two independent sources of info to mark a server as down. All-to-all multicasting makes all servers talk to one another. Gossip protocol - each node mas a list of members and heartbeat counters. Each node sends heartbeats to random nodes. Updated list is sent to a set of random nodes. If heartbeat has not increased for more than predefined periods, the member is considered offline.
