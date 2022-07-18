---
title: Gradle
date: 2022-05-29
description: Udemy course notes
category: summary
type: notes
---

gradle is an incremental build system. It checks if a task needs to be executed before it executes it.

### build.gradle

Each block in a gradle file is a closure that receives a delegate object which has properties/functions in it

```
// Plugins add new tasks to your project by extending the gradle core. Promote re-use across multiple projects.
plugins {
  id 'java' // lets you build/compile java source code - adds clean, compileJava, testClasses, jar tasks; requires project layout with src/main/java, src/main/resource, src/test/java
}

// servers that have dependencies from which you can download dependencies
repository {
  mavenCentral()
}

// 3rd party dependencies that your code needs
dependencies {
  // compile is deprecated, use implementation instead
  implementation 'org.apache.commons:commons-math3:3.6.1'

  testImplementation 'junit:junit:4.12'
}

jar {
  baseName = "$project-all"
  println "$basename"

  manifest {
    attributes 'Main-Class': 'com.package.MainClass'
  }

  // bundle all dependencies into jar
  from {
    project.configurations.runtimeClasspath.collect {File file -> project.zipTree(file)}
  }
}

task myTask(type: ParentTask) {
  // has all properties of ParentTask
  // can access other task properties by using otherTask.propName
}

```

### Dependency Management

run `dependencies` task to see all dependencies. To exclude transitive dependencies,

```
dependencies {
  implementation('some.package') {
    exclude group 'some.transtive.dep', module: 'dep-core'
  }
}
```

To get a dependency report, add `project-report` plugin which gives you an `htmlDependencyReport` task

### Multi-Project Builds

Allows clear isolation between modules which leads to more efficient builds and testing

Each project/subproject has its own gradle build configuration.

The root project will have a `settings.gradle` file

```
rootProject.name = 'myRootProjectName'

include 'subProject1, 'subProject2', 'subProject3'
```

in `build.gradle`:

```
subprojects {
  apply plugin: 'java'

  sourceCompatibility = 1.8
  targetCompatibility = 1.8

  repositories {
    mavenCentral()
  }
}

project(':subProject1') {
  dependencies {
    // define dependency on another subproject
    implementation project(':subproject2')
  }
}

project(':subProject2') {
  // gives access to implementation and api dependencies
  apply plugin: 'java-library'
  dependencies {

  }
}
```

### TODO

use git commit hash as project version

```
    version 'git rev-parse --short HEAD'.execute().getText().trim()
```
