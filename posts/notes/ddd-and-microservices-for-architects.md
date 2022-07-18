---
title: Domain-Driven Design and Microservices for Architects
date: 2022-06-12
description: Udemy course notes
category: summary
type: notes
---

Monolithic architectures

- large, modular code bases with tight coupling between components
- teams are organized by technology and business
- changes require coordination between teams and span across multiple business functions

Microservice architectures

- collection of find grained, loosely coupled services built to realize a specific business capability
- interactions are via well-defined contracts over the network

Business transformation examples:

- microsoft: packaged software to subscription model
- amazon: online bookstore to marketplace
- apple: computers to ipod, iphones, mac, music
- netflix: DVD mail service to streaming service

Businesses change because of regulatory changes, competitive pressure, new opportunities, and customer demands

Transformation is not a one-time initiative. Needs to be rapid and continuous.

Each service is built and operated by a small team (two-pizza team) which evolves independently. Each team focuses on the business capabilities provided by their microservice. Leads to decentralized governance and well-defined business scope.

Microservice advantages:

- requires minimal coordination with other teams
- regression testing is needed only for the changed service
- independent deployments
- failure in one service will not bring down entire system
- each service can scale independently

Microservice disadvantages:

- poor network performance
- complexity managing data integrity
- harder to monitor and debug

Creating a business case:

- quantify the business value
- have a roadmap
- describe requirements for success
- present a PoC

### Introduction to Domain-Driven Design

A domain = sphere of knowledge, influence, or activity. A field or industry in which the business operates. The problem space.

Each domain consists of sub-domains. Three types: generic, core, supporting

No one expert know everything about the domain. Multiple domain experts are aligned to sub-domains.

A domain is made up of vocabulary, entities/relationships, key concepts, and workflow activities.

Teams central focus should be on the core domain and domain logic.

Each sub-domain has a different level of complexity that is driven by the business rules, compliance, process/handovers, dynamicity

##### Modeling

Think about purpose of model and perspective of shareholders before constructing a model.

conveys idea to stakeholders and provides a point of reference to create detailed specifications.

Define common terminology for all domain concepts.

Define relationships between concepts.

4 + 1 modeling

Logical view = functionality provided to end-user/business value

Process view = workflow/interaction

Development view = modules/subsystems

Physical view = servers/db design

Taken together, these views form scenarios/requirements

knowledge crunching = convert SME knowledge into structured domain models

Goal of a software development team is to help the business achieve its business goals, hence the team MUST understand the business model.

##### Business Model Canvas

A tool that helps discuss, communicate, design, and understand the organization's business model

9 areas:
key partners - who supplies resources?
key activities
key resources - what does the value prop need?
value propositions
customer relationships
channels
customer segments
cost structure
revenue streams

### Domain-Driven Design Patterns

Generic subdomain: known solutions with best practices exist. No business advantage exists in re-inventing the wheel.

Core subdomain: the business differentiator which gives the company a competitive advantage. Fast paced and ever evolving.

Supporting subdomain: no business advantage but core depends on it.

Known solutions available? Generic
Adds business value? Core
Core depends on it? Supporting

Business should focus on core sub-domain, buy generic sub-domains, and outsource supporting domains.

##### Ubiquitous language

A common, evolving, language within each business context that is used by _all_ stakeholders - business and technology experts. Used in code/tests, communication, documentation.

##### Bounded Context

Entities in one sub-domain are independent of other sub-domains, even if they share the same language (ie retail banking customer = employer + demographics, credit cards customer = income + credit history, compliance = kyc/fraud).

Each bounded context has its own ubquitous language and is translated into one or more microservices.

Discover bounded contexts by drawing boundaries around business functions

A context map is a visual representation of the systems bounded contexts and the relationships between them

- separate ways pattern - each bounded context is completely independent of one another

- Symmetric relationship - bounded contexts depend on one another.

  - Partnership pattern. Teams must coordinate changes.
  - Shared kernel pattern - use a shared library which contains shared models. Coordination is only needed on changes on the shared models.

- asymmetric relationship - one bounded context depends on another.

  - customer-supplier pattern. Upstream bounded context fulfills a specific need of the downstream bounded context. Upstream BC adjusts to meet requirements of downstream BC.
  - Conformist pattern. Upstream BC exposes models with no regard to ANY douwnstream BC. Downstream BC conforms to models exposed by upstream BC.
  - Anti-corruption layer pattern. Downstream BC isolates translation logic in a separate layer.

- one-to-many relationships.
  - Open host service. Upstream provider offers common services to other BC's. Published language is used by all downstream BCs.

##### Entity Object

A business object that encapsulates attributes and well-defined domain logic (business rules, validations, calculations) that is within a bounded context. Entities are persisted in long term storage and have IDs to uniquely identify them. IE accounts, orders, etc

##### Value Object

An immutable object that doesnt map directly into the core concepts of the bounded context. Data types the encapsulate validation/format for non core concepts - IE EmailAddress. Do not have IDs - comparison requires full attribute-level comparison. Not persisted as an independent object but may be part of an entity.

##### Aggregate

An agregate object is a cluster of entities and value objects that are viewed as a unified whole from the domain concepts and data perspective. Has a unique identity. Example: Account is aggregate route, has transactions.

The aggregate provides functions to operate on the inner objects. Do not interact with inner objects directly.

Inner objects are only meaningful in the context of the root.

##### Repository

A repository object acts as a collection of aggregate objects in memory. It hides the storage level details of the aggregate.

One repo per aggregate. Persistence operations are atomic.

Used to keep the domain model indpeendent of the storage layer.

##### Anemic vs Rich Models

anemic model - a domain model composed of entities that do not exhibit operations applicable to the domain concepts

rich model - business model is implement as an inherent part of the entity

Anemic models are acceptable for apps with minimal business logic, simple CRUD, or shared logic that doesn't belong in a single model entity.

##### Domain Service

Implements domain functionality that may not be modeled as a behavior in any domain entity or value object. Does not maintain state between calls. Should be highly cohesive and do only one thing. May call other domain services.

##### Application Service

Does not implement domain functionality but depends on other domain objects/services. Stateless, define an external interface, exposed over network. Orchestrates execution of domain logic and transforms into response object. Can be used to consolidate results from several domain objects.

##### Infrastructure Service

A service that interacts with an external resource to address a concern that is not part of the primary problem domain.
Examples - logging, notifications, persistence, external APIs. No dependency on any domain object.

### TODO

look for practical DDD examples with persistence. How to do db-backed validation in an entity object?
