---
title: Dive into Design Pattern
date: 2021-10-10
description: Dive into Design Pattern Summary
category: book summary
type: books
---

# Dive Into Design Patterns

### Chapter 1 - Intro to OOP

wrapping data and behavior related to that data into objects that are constructed from blueprints called classes - class has methods - fields on class are called members of the class - members = state, methods = behavior

4 Pillars

- Abstraction: objects only model attributes and behaviors of real objects in a specific context. IE Airplane would be represented differently in a flight simulator vs a flight booking application
- Encapsulation: each object has an interface that is open to interaction with other objects. IE a car engine can have a start method but all the details are hidden. Object hides parts of its state and behaviors from other objects.
- Inheritance: ability to build new classes on top of existing ones.
- Polymorphism: ability of a program to detect the real class of an object and call its implementation even when its type is unknown in the current context.

Object relationships

- Association: one object uses or interacts with another. Solid line with arrow in UML.
- Dependency: object accepts another object as a method parameter, instantiates, or uses another object. Dotted line with arrow in UML. If changes in to a class definition results in modifications in another class, you have a dependency.
- Composition: One object is composed of one or more instances of another. Filled in diamond arrow in UML. Container controls lifecycle of dependents. (University is composed of departments)
- Aggregation: one object contains reference to one or more instances of another. Empty diamond arrow in UML.

### Chapter 2 - Intro to Design Patterns

pattern = typical solution to commonly occuring problems in software design that descirbe how a couple classes relate and interact with each other.

### Chapter 3 - Software Design Principles

#### Features of Good Design

Code reuse: reuse code rather than develop something over and over again to reduce development cost. To do so requires effort to ensure loose coupling (coding against interfaces, no hardcoded operations) at the expense of making the software more complicated.

Framework: don't call us, we'll call you (JUnit manages test lifecycle, Spring manages calls to things)

Extensibility: only constant thing is change - things change because: we understand the problem better once we start to solve it, the goal posts move after the clients see what your app can do. It's important to provide for possible future changes when designing an application's architecture.

Design Pattern Principles:

- Encapsulate what varies: identify aspects of your app that vary and separate them from what stays the same. Main goal is to minimize the effect caused by changes.
  - on a method level, move frequently changing stuff into a separate method (ie computing taxes for an order total)
  - on a class level, move frequently changing stuff into a separate class and delgate work to that class
- Program to an interface, not an implementation:
  - allows for extensibility in the future
- Favor composition over inheritance
  - subclasses can't reduce the interface of the superclass
  - overridden methods must be compatible with the base one
  - breaks encapsulation of the superclass
  - tight coupling between superclass and base classes

![Inheritance](/images/inheritance.png)
![Composition](/images/composition.png)

#### SOLID Principles

Single responsibility principle - a class should only do one thing. You should only need to change a class if that one thing changes.

Open/Closed principle - classes should be open for extension but closed for modification. Goal is to keep existing code from breaking when you implement new features. A class is open if you can add new fields/methods, override base behavior. Ie if you use the strategy pattern, you can override behavior (open) without changing the original class (closed)

Liskov substitution principle - when extending a class, you should be able to pass objects of the subclass in place of objects of the parent class without breaking the client code. The subclass should remain compatible with the behavior of the superclass; when you override a method, you should extend the base behavior rather than replace it. A subclass shouldn't strengthen pre-conditions or weaken post-conditions.

Interface segregation principle - clients shouldnt be forced to depend on methods they don't use. Break down "fat" interfaces into more specific ones.

Dependency inversion principle - high-level classes shouldn't depend on low-level classes. Both should depend on abstractions.

### Factory Method

replace direct object construction calls using the new operator with calls to a special factory method.

![Factory](/images/factory.png)
![Factory Example](/images/factory-example.png)

### Abstract Factory

produce families of related classes without specifying their concrete classes. Benefit is that you don't need to modify the client code each time you add a new variation of components. Use the pattern when you have a set of factory methods.

![Abstract Factory](/images/abstractfactory.png)
![Abstract Factory Example](/images/abstractfactory-example.png)
![Abstract Factory Example 2](/images/abstractfactory-example2.png)

### Builder
