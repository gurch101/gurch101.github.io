---
title: Cloud Native Patterns
date: 2021-10-03
description: Cloud Native Patterns Summary
category: book summary
type: books
---

Cloud = where it runs; Cloud-native = how it runs

Modern apps need rapid iteration, fast releases, zero downtime and an increase in the volume and variety of devices that connect to it.

Cloud-native applications remain stable even when the infra they're running on is constantly changing/failing

### Chapter 2: Running Cloud-Native Applications in Production

In most orgs, getting software deployed to prod is challenging - process designed to reduce risk and improve efficiency has the opposite effect

inconsistencies in artifacts, configuration, and environments lead to system instability

goal should be easy and frequent releases into production

continuous delivery = the newest possible version of the software is deployable at any time (trunk is always deployable)

- advantages: can deploy at any time (first mover advantage), allows you to gather feedback early. If you miss dates,
  you can release less features

traditional delivery = SDLC is front-loaded with development, followed by extensive testing and packaging

- disadvantages: if you miss dates, testing is compressed (at the expense of quality) or you push dates

Before: tested extensively before going to production and were left scrambling on go-live.

After: Plan for failure and intentionally create a retreat path to make failures non-events. Monitor for crashes, latency changes, changes in click-through rates, etc.

### Chapter 3: The Platform for Cloud-Native Software

AWS uses commodity hardware to offer services at a lower price but with a higher rate of failure. Exposes primitives as a service.

Google app engine doesnt provide raw access to resources.

For cloud-native apps, operators are interested in the application and _not_ the hosts the app is running on/directories involved

The popularity of containers was driven by the need to support cloud-native applications. Multiple containers run on a single host sharing the host's operating system.

The platform creates app instances, monitors application health, distributes app instances across the infrastructure, assigns IP addresses to the containers, dynamically routes to app instances, and injects configuration

challenges of highly distributed architectures:

- coordinating configuration changes
- tracing/monitoring execution flows
- retry logic - circuit breakers prevent inadvertent internal DDoS
- service discovery - each component needs to know URLs/IPs of all components it calls
- rolling upgrades

each team is responsible for deployment, configuration, monitoring, scaling, and upgrading its products

### Chapter 4: Event-Driven Microservices

Request-response = procedural/top-down - service request waits for responses from each dependent service and does some sort of aggregation to respond. Aggregation occurs in response to request.

Event-driven = fire and forget - code execution has an effect and the outcome may cause other things to happen but the entity that triggered the execution doesnt expect a response. Aggregation occurs whenever data in the system changes - its asynchronous.

Even-driven is less coupled and does not depend on other systems working correctly. Dependent services execute events to the aggregate service which maintains its own data store. Each service is independent of the other.

Command Query Responsibility Segregation (CQRS) - separates write logic (commands) rom read logic (queries). Splitting controllers by read/write allows for different models on read/write.

Events need message queues (RabbitMQ/Kafka) to store events through network partitions and handle downtimes in services

https://github.com/cdavisafc/cloudnative-abundantsunshine

### Chapter 5: App Redundancy: Scale-Out and Statelessness

Core tenant of cloud-native software is to have redundancy to avoid single points of failure. Apps should always have multiple instances deployed.

Multiple instances should behave like a single logical entity.

Multiple instances allow for flexible horizontal scaling, high availability, reliability, and operational efficiency.

Kubernetes is a platform for running applications that includes capabilities that allow you to deploy, monitor, and scale your apps. Apps must be containerized to run in kubernetes.

Instances must be stateless to allow for redundant deployment.

CAP theorem - only two attributes of consistency, availability, and partition tolerance can be maintained.

### Chapter 6: Application Configuration: Not just environment variables

Need a way to apply config to dynamic number of instances where infrastructure changes can cause config changes.

Store config in the environment when possible. With spring, you can use the property file for defaults:

```
ipaddress=${INSTANCE_IP:127.0.0.1}
```

Configuration data stores persists key/value pairs, maintains version history, and has access control mechanisms. Spring cloud config exposes config from a source code control system over an HTTP API. Sensitive values should be encrypted and stored by a service like Hashicorp Vault.

### Chapter 7: The Application Lifecyle: Accounting for Constant Change

Management functions should be automatable, efficient, and reliable

Platform needs a fail-safe way of detecting when an app failed

App scales depending on load to safe cost - new apps are started/stopped all the time.

Zero downtime deployments:

- blue/green deployment - stand up second set of instances, then switch traffic from first set to second set. All traffic is on the old or new version exclusively.
- rolling upgrade - replace a subset of instances with new instances. Different versions of the app will serve traffic at the same time temporarily.
- parallel deploys - different versions of the app serve traffic for as long as needed. Supports experimentation.

For credential rotation across dependent services, make the provider accept two credentials temporarily (old + new), restart the app, then add the new credential to the client, restart the app, then remove the old credential from the provider.

You should _not_ be able to SSH into a production environment because doing so allows a way to make instances non reproducible.

Treat logs as event streams

- write directly to stderr/stdout
- avoids need for log rotations or locating files in directories

each service needs to broadcast events about its lifecycle - ip/port, etc

- publish health endpoints, kubernetes will call it continuously to detect crashed apps/app lifecycle. If its down, kubernetes will start a new instance. Can also be event-driven where you publish events.

Serverless takes cloud-native to the extreme

- each call takes app through entire lifecycle
- dev needs to focus on making startup and execution as fast as possible

### Chapter 8: Accessing Apps: Services, routing and service discovery
