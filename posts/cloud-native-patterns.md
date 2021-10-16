---
title: Cloud Native Patterns
date: 2021-10-03
description: Cloud Native Patterns Summary
category: book summary
type: notes
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

For cloud-native apps, operators are interested in the application and *not* the hosts the app is running on/directories involved

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