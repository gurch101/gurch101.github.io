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
            /out
                /persistence
        /domain
            Account
        /application
            SendMoneyService
            GetAccountBalanceService
            /port
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