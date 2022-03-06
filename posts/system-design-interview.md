---
title: System Design Interview
date: 2021-10-10
description: System Design Interview Summary
category: book summary
type: notes
---

### Chapter 1: Scaling from 1 to a million users

- start with a single server to host static assets, app server, database server
- vertically scale until you can't
- split db and app server so they can be scaled independently
    - nosql = low latency, unstructured, massive amounts of data
- horizontally scale - use a load balancer 
    - improved availability
    - stateful architecture requires sticky sessions
    - stateless - more scalable -> store user state in redis/nosql db/relational db
- failover/redundancy - db replication with one master db for writes and many read nodes
- add a cache for data that is read frequently but updated infrequently
    - considerations: expiration, what to cache, consistency, eviction (what to do when full - LRU)
- use a CDN
    - considerations: cost (only cache frequently used content), expiry, failover, invalidation
- use multiple data centers and a geoDNS - routes to server closest to user
    - considerations: automated deployments, monitoring, replication, immutable infrastructure
- use message queues for long running tasks
- sharding the db to scale horizontally
    - considerations: resharding data when shards need to be split further, celebrity problem (certain users are more popular/active than others), leads to denormalization to prevent joins
- split tiers into separate services


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
• Daily active users (DAU) = 300 million * 50% = 150 million
• Tweets QPS = 150 million * 2 tweets / 24 hour / 3600 seconds = ~3500
• Peek QPS = 2 * QPS = ~7000
We will only estimate media storage here.
• Average tweet size:
• tweet_id 64 bytes
• text
140 bytes
• media
1 MB
• Media storage: 150 million * 2 * 10% * 1 MB = 30 TB per day
• 5-year media storage: 30 TB * 365 * 5 = ~55 PB

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
``` serverIndex = hash(key) % N, where N is the size of the server pool
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