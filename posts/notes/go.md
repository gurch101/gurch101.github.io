---
title: Go Summary
date: 2021-09-18
description: Golang Summary
category: summary
type: notes
---

## Hello World

```go
package main

import "fmt"

func main() {
    fmt.Println("Hello, World")
}
```

## Variables

```go
// type is specified last just like function parameters
// can be defined at package or function level
var c, python, go bool

// type is not needed if variable is initialized
var c, python, java = true, false, "no!"

// can use short declarations inside functions
c, python, java := true, false, "no!"

const line = "Hello, World!"
```

## Types

bool, string, (u)int(8/16/32/64), byte, rune(alias for int32 used for unicode), float32/64, complex64/128

## Functions

```go
// When two or more consecutive named function parameters share a type, you can omit the type from all but the last
func funcName(param1 int, param2 int) int {
    return param1 + param2
}

// functions can return any number of results
func swap(x, y string) (string, string) {
	return y, x
}

a, b := swap("hello", "world")

```

### Iteration

```go
// could use same syntax to iterate over a string or a map
for idx, val := range arr {

}
for i := 0; i < len(arr); i++ {

}
```

### Maps

```go
// iterate over all keys
for k := range mymap {

}
```

## Visibility

names are exported if they begin with a capital letter. Otherwise, names are private.

## Modules


Create a module

```bash
go mod init gurch101.com/modname
```

## CLI

Compile and run go program

```bash
go run <prog.go>
```

// todo control flow, structs/slices/maps, methods/interfaces, concurrency

// how does defer close work?