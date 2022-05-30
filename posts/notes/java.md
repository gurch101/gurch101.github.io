---
title: Java Summary
date: 2021-09-18
description: Java Summary
category: summary
type: notes
---

### Primitive Types

integer types:
byte - -2^7 to 2^7 - 1
short - -2^15 to 2^15 - 1
int - -2^31 to 2^31 - 1 (Integer.MIN_VALUE, Integer.MAX_VALUE)
long - 2^63 to 2^63 - 1 (should be suffixed with L)

floating point types:
float (32 bit)
double (64 bit)

doubles and ints are the default type for numeric literals.

Can get min/max values by calling `<DataType>.MIN_VALUE/MAX_VALUE`

char - single character in quotes - 16 bits - can store unicode (65k characters) - `'\u<charCode>'`
boolean

### Modules

Introduced in Java 9.

A module is a container of packages.

Has a name, inputs (module dependencies), and outputs (package exports)

Every module has a module descriptor file at the module root folder and is named `module-info.java`

```
[open] module <my.module.name> {
  requires // other modules this module depends on
  exports // packages exported by the current module
  provides // service implementations that the current modules provide
  uses // the services the current module consumes
}
```

A normal module grants access at compile and run time to types in only those packages which are explicitly exported.

An open module grants access at compile time to types in only those packages which are explicitly exported but grants access at run time to types in all its packages.

Automatic modules are not explicitly declared in module descriptor files. It is automatically created when a JAR file is placed into the module path. By default, it `requires` all platform modules, all our own modules, and all other automatic modules and exports all packages by default.

Libraries that use reflection like Hibernate or Spring will only work with open modules.
