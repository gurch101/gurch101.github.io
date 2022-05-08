---
title: Get your Hands Dirty On Clean Architecture
date: 2021-10-10
description: Get Your Hands Dirty on Clean Architecture Summary
category: book summary
type: notes
---

### What's Wrong with Layers

Traditional app - controllers, domain/service, persistence

Problems:
- It promotes database-driven design. We should be trying to model behaviour, not state. Also leads to tight coupling between the repository/entities and the service.
- It's prone to shortcuts. In a typical layered architecture, you can only access things in the same layer or lower. If you need to access something in a higher layer, we move the thing to the lower layer. IE utils/helpers in repo layer.
- It grows hard to test. In simple cases, devs make the controller talk to repo directly - makes test setup difficult. More dependencies = harder test setup. A complex test setup is the first step towards no tests at all because we don't have time for them.
- It hides the use cases. Layered architectures don't impose rules on the "width" of domain services making things hard to find/maintain.
- It makes parallel work difficult.

### Inverting Dependencies

Single Responsibility Principle - a component should only have one reason to change.

You shouldn't need to change a component if its dependencies change.

Dependency inversion principle - invert the direction of any dependency in our code base.

The domain code shouldn't have dependencies; instead, all depdencies point towards the domain code.

![Dependency Inversion](/images/di.png)
![Hexagonal Architecture](/images/hexarch.png)

### Organizing Code

Traditional package structure is just a nice-looking facade for an unstructured mess of code - classes in one package import classes from other packages that should not be imported.
```
app
    /domain
    /persistence
    /web
```

Organizing by feature - allows enforcing access via package-private visibility for classes. Don't allow broad services (ie use SendMoneyService vs AccountService).

```
app
    /feature1
        /adapter
            /in
                /web
                    /AccountsController (package private)
            /out
                /persistence
                    / AcoountPersistenceAdapter (package private - implements UpdateAccountStatePort)
        /domain - domain is public
            Account
        /application
            SendMoneyService (package private - implements SendMoneyUseCase)
            /port - all ports are public
                /in
                    SendMoneyUseCase
                    SendMoneyCommand
                    GetAccountBalanceQuery
                /out
                    LoadAccountPort
    /feature2
```

![Package structure](/images/hexarchpackages.png)

### Implementing a Use Case

- don't put input validation in input adapters since multiple adapters may consume use case. Instead, put validation in the input model. Each use case should have a dedicated input model.
- business rule validation should go in the domain entities (rich domain model) or in the use case code itself (anemic domain model).
- use cases should return as little data as possible. Return types should be isolated from other use cases.

![Use Case](/images/usecase.png)

### Implementing a Web Adapter

Web adapters take requests from the outside and translates them into calls to our application core. Control flow goes from controllers in the web adapter to the services in the app layer.

![Web Adapter](/images/webadapter.png)

Responsibilities:
- Map HTTP request to native objects
- Perform auth checks
- Validate input (ensure input model can be translated to use case model)
- Map input to the input model of the use case
- Call use case
- Map output of the use case back to HTTP
- Return HTTP response

Create a separate controller for each operation, don't reuse input objects (ie create endpoint/update endpoint probably differ by an id attribute only but that is okay). Consider putting each controller and its input objects into a package and making the input objects private to discourage re-use.

### Implementing a Persistence Layer

Services call port interfaces to access persistence functionality. These interfaces are implemented by persistence adapters that are responsible for talking to the actual database. This layer of indirection lets you evolve domain code without thinking about persistence problems

![Persistence Adapter](/images/persistence.png)

Responsibilites:
- take input
- map input into database format (in java, domain objects to JPA entities)
- send input to the database
- map database output into application format
- return output

The repository should only expose methods that are needed by the service. Interface segregation principle: broad interfaces should be split into specific ones so that clients only know about methods they need. Bob Martin: "Depending on something that carries baggage you don't need can cause you troubles you didn't expect". You can have one persistence adapter that implements all persistence ports for each entity.

![Persistence Context](/images/persistence-context.png)

Transactions should be managed by the service since the persistence layer doesn't know which operations are part of the same use case.

### Testing Architecture Elements

Testing pyramid tells us that system/integration tests shouldn't focus on coverage because it will lead to too much time spent building tests.

Domain entities should be tested with unit tests.

Use cases should also be unit tests with mocked dependencies

Web adapters should be be tested with `@WebMvcTest(controllers = ControllerUnderTest.class)` using `MockMvc`

Persistence adapters should be tested with `@DataJpaTest` and explicitly `@Import({Adapter.class, Mapper.class})`.

System tests should be done with `@SpringBootTest` with a `TestRestTemplate`. Only cover the most important/common parts.

Test coverage alone is meaningless - test success should be measured in how comfortable the team feels shipping the software. The more often you ship, the more you trust your tests. For each production bug, ask "why didnt our tests catch this bug?", document the answer, and then add a test that covers it.

### Mapping between Boundaries

there are tradeoffs in using the same model in two layers of the app vs implementing a mapper.

No Mapping Strategy:

![No Mapping](/images/nomapping.png)

Pros: no mapping needed. Good for simple CRUD use cases.

Cons: Single entity will have annotations to deal with JSON serialization/request validation/database mapping. Meanwhile, the actual service layers cares about none of these things. Violates single responsibility principle.

Two-Way Mapping Strategy:

![Two-Way Mapping](/images/twowaymapping.png)

Outer layers map into the inner domain model and back. The inner layers concentrate on domain logic and aren't responsible for mapping.

Pros: Single responsibility principle is maintained since each layer has its own model which may have a structure that is completely different from the domain model.

Cons: Lots of boilerplate. Debugging mapping code can be a pain especially if its hidden behind a framework of generics/reflection. Since the domain object communicates across boundaries, it is vulnerable to change required by the outer layers.

Full Mapping Strategy:

![Full Mapping](/images/fullmapping.png)

Web layer maps to command object of app layer. Each use case has its own command.

Pros: there is no guessing involved as to which fields should be filled and which fields should be left empty. The application layer maps the command object to the domain model.

Cons: Even more mapping code since you are mapping into many different command objects (one per use case).


One-Way Mapping Strategy:

![One-Way Mapping](/images/onewaymapping.png)

Models in all layers implement the same interface that encapsulates the state by providing getters on the relevant attributes.

Domain model itself can implement rich behavior which is accessible only within the service layer.

Domain object can be passed to the outer layers without mapping since the domain object implements the same state interface.

Layers can then decide if they work with the interface or if they need to map it to their own model. They cannot modify the state of the domain object since the modifying behavior is not exposed by the state interface. Mapping is unnecessary at the web layer if we're dealing with a "read" operation.

Pros: clear mapping responsibility - if a layer receives an object from another layer, we map it to something that layer can work with. Thus each layer only maps one way. Best if the models across the layers are similar.

Cons: doesn't work if models across layers are not similar.


Which to use:

If working on a modifying use case, use full mapping between web and application layers. This gives clear per-use-case validation rules and we don't need to deal with fields we don't need in a certain use case. Use no mapping between the application and persistence layer in order to be able to quickly evolve the code without mapping overhead. Move to two-way mapping once persistence issues creep into application layer.

If working on a query, start with the no mapping strategy between all layers. Move to two-way mapping once we need to deal with web/persistence issues in the app layer.

### Assembling the Application

We want to keep the code dependencies pointed in the right direction - all dependencies should point inwards towards the domain code so that the domain code doesn't need to change when something in the outer layers changes. Nice side effect of this is testability - all use cases only know about interfaces which are injected.

Should be the responsibility of configuration components to construct concrete implementations at runtime.

With Spring, use `@Component` and `@RequiredArgsConstructor` with private final dependencies on interfaces. Alternative that doesn't involve classpath scanning, use `@Configuration` classes where `@Bean` is exposed - this has the benefit of keeping spring specific annotations outside of application code. Use `@EnableJpaRepositories` to instantiate spring data repository interfaces. Keep "feature" annotations on specific config classes rather than the main application to keep test start up fast.

### Enforcing Architecture Boundaries

There is a boundary between each layer and its next inward/outward neighbor - dependencies that cross a layer boundary must always point inwards. Java visiblity modifiers don't scale to big packages since sub-packages are treeated as different packages so package-private doesn't always work.

Use ArchUnit to do post-compile checks at build time.


```java
@Test
void domainLayerDoesNotDependOnAppLayer() {
    noClasses()
        .that()
        .resideIn("buckpal.domain")
        .should()
        .dependOnClassesThat()
        .resideInPackage("buckpal.application")
        .check(new ClassFileImporter().importPackages("buckpal.."));
}
```

Build separate artifacts.

![Multi-module project](/images/multimodule.png)

### Taking Shortcuts Consciously

Broken window theory - as soon as something looks rundown or damaged, people feel that it's ok to make it more rundown or damaged.

Maintain [architecture decision records](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions) to consciously document added shortcuts.

Common shortcuts:
- sharing input/output models between use cases when the use cases are functionally bound and we actually want both use cases to be affected if we change a certain detail. As soon as they evolve separately from one other, separate the models even if it means to duplicate classes.
- using the domain entity  as the input/output model for a use case.
- skipping incoming ports to remove a layer of abstraction. Adapters are forced to know more about the internals of the application.
- skipping the serivce and communicating directly with the persistence layer
