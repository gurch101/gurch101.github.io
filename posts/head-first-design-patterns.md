---
title: Head First Design Patterns
date: 2021-10-10
description: Head First Design Patterns Summary
category: book summary
type: notes
---

### Chapter 1: Intro to Design Patterns

best way to use design patterns is to load your brain with them then recognize places in your apps where you can apply them.

Problem with inheritance - changes can unintentionally affect all subclasses

Problem with interfaces - can lead to code duplication if you have multiple classes that implement the same interface in the same way

identify aspects of your app that vary and separate them from what stays the same.

Program to an interface, not an implementation.

Use delgation - have a base class delegate behavior to other interfaces

favor composition over inheritance

strive for loosely coupled designs between objects that interact.

The strategy pattern: a family of algorithms that are encapsulated and interchangeable. Allows the algorithm vary independently from the client that uses it.

![Strategy Pattern](/images/strategy.png)

### Chapter 2: The Observer Pattern

one to many dependency between objects so that when one object changes state, all of its dependents are notified automatically.

pub/sub - publisher is the subject, subscribers are the observers

![Observer Pattern](/images/observer.png)

rather than the subject sending out all datapoints to all observers, update() can take no params and it could be the responsibility of the observer to call get on the subject.