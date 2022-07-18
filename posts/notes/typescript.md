---
title: Typescript
date: 2022-05-29
description: Udemy course notes
category: summary
type: notes
---

### Types

core types: number, string, boolean, object, array

typescript types: tuple, enum, any, void (will return undefined in js - callbacks that have a void return "can" return something but the return will be unused), Function, unknown, never

```ts
function add(n1: number, n2: number): number {
  return n1 + n2;
}

const n1 = 5; // don't need to specify type since typescript infers it
let someVar: string; // specify type if variable declared but not initialized

let otherAdd: (n1: number, n2: number) => number;
otherAdd = add;

// types are inferred
const person = {
  name: "John",
  age: 30,
};

// person.foobar will show error

// starts at 0, can assign own number/string
enum Permission {
  READ_ONLY,
  ALL_ACCESS,
}

// above is preferred; let typescript infer the type
const explicitlyTypedPerson: {
  name: string;
  age: number;
  details: {
    addr1: string;
  };
  hobbies: string[];
  role: [number, string]; // tuple - fixed length where each element is of a specific type
  permission: Permission;
} = {
  name: "Jane",
  age: 30,
  details: {
    addr1: "510 Burrard",
  },
  hobbies: ["Sports", "Cooking"],
  role: [2, "author"],
  permission: Permission.READ_ONLY,
};

let favoriteActivities: string[];
favoriteActivities = ["foo", "bar"];

function inlineObjFunc(user: { name: string; age: number }) {}
```

### Union Types

```ts
function combine(a1: number | string, a2: number | string) {
  if (typeof n1 === "number" && typeof a2 === "number") return a1 + a2;
  else return n1.toString() + n2.toString();
}

console.log(combine(1, 1)); // 2
console.log(combine("a", "b")); // 'ab'
```

### Literal Types

```ts
// can only provide val1 or val2 as param
function myFunc(someLiteral: "val1" | "val2") {}
```

### Type Aliases

```ts
type MyType = number | string;
type MyOtherType = "val1" | "val2";
type User = {
  name: string;
  age: number;
};
function myFunc(myVal: MyType) {}
```

### Unknown Type

```ts
let userInput: unknown;
let userName: string;

userInput = 5;
userInput = "Hello";
// unknown type requires typechecking; any does not
if (typeof userInput === "string") {
  userName = userInput;
}
```

### Never type

Use never for functions that always throw, or functions with infinite loops

```ts
// never doesn't ever return anything; void returns undefined
function generateError(message: string, errorCode: number): never {
  throw { message, errorCode };
}
```

### Classes

```ts
class Employee {
  private someOtherProp: string;
  static someConstant = 20;
  // don't need to explicitly list properties separate
  constructor(private readonly id: string, public name: string) {
    this.someOtherProp = "foobar";
  }

  // call via Employee.create();
  static create(id: string, name: string) {
    return {
      id,
      name,
      someOtherProp: "foo",
    };
  }
}

abstract class Describeable {
  abstract describe(): void;
}

class Department implements Describeable {
  // default is public
  private name: string;
  withDefault: string = "foo";
  protected employees: string[] = []; // protected can be changed from subclasses

  constructor(n: string) {
    this.name = n;
  }

  addEmployee(employee: string) {
    if (this.validateEmployee(employee)) {
      this.employees.push(employee);
    }
  }

  describe() {
    console.log(`Department: ${this.name}`);
  }

  private validateEmployee(employee: string): boolean {
    return employee.length > 1;
  }

  // access as property via dept.name;
  get name() {
    return this.name;
  }

  set name(name: string) {
    this.name = name;
  }
}

const dept = new Department("my department");

class ITDepartment extends Department {
  constructor(name: string, public admins: string[]) {
    super(name); // super must be called first
  }

  // overrides base class method
  addEmployee(employee: string) {
    if (employee !== "Foo") {
      this.employees.push(employee);
    }
  }
}

const itDepartment = new ITDepartment("my it department", ["admin1"]);

class MySingleton {
  private static instance: MySingleton;
  // can't call new
  private constructor(id: string) {}

  static getInstance() {
    if (MySingleton.instance) {
      return MySingleton.instance;
    }
    MySingleton.instance = new MySingleton("0");
    return MySingleton.instance;
  }
}
```

### Interfaces

only exists in typescript. Transpilation removes it.

```ts
// type Person = {} will work but not common - interfaces can be implemented; types can't; types are used for aliases/union types
// interface can extend another interface
interface Person {
  // can't use public/private but can add readonly
  name: string;
  age: number;
  // optional property
  foo?: number;
  // optional method
  greet?(phrase: string): void;
}

// can use interface as a function type

interface AddFn {
  // anonymously defined
  (a: number, b: number): number;
}

let add: AddFn;
add = (a: number, b: number) => {
  return a + b;
};

class PersonImpl implements Person {
  constructor(private name: string, private age: number) {}
  greet(phrase: string) {}
}
let user: Person;

user = {
  name: "Max",
  age: 12,
  greet(phrase: string) {
    console.log(phrase);
  },
};
```

### Intersection Types

```ts
type Admin = {
  name: string;
  privileges: string[];
};

type Employee = {
  name: string;
  startDate: Date;
};

// similar to interface inheritance. Intersections also work with interfaces. Combines all properties into new type.
type ElevatedEmployee = Admin & Employee;

type Combinable = string | number;
type Numeric = number | boolean;

// Universal is of type number.  Combines all common types.
type Universal = Combinable & Numeric;

const e: ElevatedEmployee = {
  name: "Max",
  privileges: ["create-server"],
  startDate: new Date(),
};
```

### Type Guards

```ts
class Car {
  drive() {}
}
class Truck {
  loadCargo() {}
}

type UnknownEmployee = Employee | Admin;
type Vehicle = Car | Truck;

function print(e: UnknownEmployee) {
  // type guard for interfaces
  if ("startDate" in e) {
    console.log(e.startDate);
  }
}

function useVehicle(v: Vehicle) {
  // for classes you can use instanceof; in also works
  if (v instanceof Truck) {
    v.loadCargo();
  }
}
```

### Discriminated Unions

```ts
interface Bird {
  type: "bird";
  flyingSpeed: number;
}

interface Horse {
  type: "horse";
  runningSpeed: number;
}

type Animal = Bird | Horse;

function moveAnimal(a: Animal) {
  let speed;

  switch (a.type) {
    case "bird":
      speed = a.flyingSpeed;
      break;
    case "horse":
      speed = a.runningSpeed;
      break;
  }

  console.log(speed);
}
```

### Type Casting

```ts
// ! means expression will never return null
const el = document.getElementById("user-input")! as HTMLInputElement;
// won't work with jsx files
const el2 = <HTMLInputElement>document.getElementById("user-input");
```

### Index Properties

```ts
interface ErrorContainer {
  // object key can be of any string
  [prop: string]: string;
}

const errorBag: ErrorContainer = {
  email: "oops",
  username: "oops2",
};
```

### Optional Chaining

```ts
const fetchedData = {
  id: 1,
  person: { name: "foo" },
};

console.log(fetchedData?.job?.title);
```

### Nullish Coelescing

```ts
const userInput = null;
// set to default if user input is null or undefined
const storedData = userInput ?? "default";
```

### Generics

```ts
const names: Array<String> = []; // eq: string[]

// resolve will be passed a string
const promise = new Promise<string>((resolve, reject) => {});

// data will be a string
promise.then((data) => data.split(""));

function merge<T extends object, U extends object>(objA: T, objB: U): T & U {
  return Object.assign(objA, objB);
}

interface Lengthy {
  length: number;
}

function countAndDescribe<T extends Lengthy>(element: T): [T, string] {
  let descriptionText = "Got no elements";
  if (element.length > 0) {
    descriptionText = `Got ${element.length} elements`;
  }
  return [element, descriptionText];
}

function extractAndConvert<T extends object, U extends keyof T>(obj: T, key: U) {
  return obj[key];
}

class Storage<T> {
  private data: T[] = [];

  addItem(item: T) {
    this.data.push(item);
  }

  removeItem(item: T) {
    this.data.splce(this.data.indexOf(item), 1);
  }

  getItems() {
    return [...this.data];
  }
}

const s = new Storage<string>();

interface Course {
  title: string;
  description: string;
}

function createCourse(title: string, desciption: string): Course {
  let course = Partial<Course> = {};
  course.title = title;
  course.description = description;
  return course as Course;
}

const names: Readonly<string[]> = ['Max', 'Anna'];
```

### Decorators

add `experimentalDecorators: true` to tsconfig to enable ES7 decorators

Decorators are executed on _definition_ of the class, not instantiation

```ts
function Logger(constructor: Function) {
  console.log("Logging...");
}

@Logger // class decorators are called when the class is defined
class Person {
  name = "Max";
}
```

```ts
function Logger(logString: string) {
  return function (constructor: Function) {
    console.log(logString);
  };
}

function PropertyLogger(target: any, propertyName: string | Symbol) {}

function MethodLogger(
  target: any,
  name: string | Symbol,
  descriptor: PropertyDescriptor
) {}

function AccessorLogger(
  target: any,
  name: string,
  descriptor: PropertyDescriptor
) {}

// name is the method name
function ParamLogger(target: any, name: string | symbol, position: number) {}

// multiple decorators can be added; they run bottom up
@Logger("Person")
class Person {
  @PropertyLogger //property name will be "name", target will be the class definition
  name = "Max";

  @AccessorLogger // target will be class definiton, name will be "name", descriptor will be accessor definition
  set name(val: string) {
    this.name = val;
  }

  @MethodLogger
  printName(@ParamLogger someVal: string) {
    console.log(this.name + someVal);
  }
}
```
