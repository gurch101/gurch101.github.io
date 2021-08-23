---
title: Type Hinting For Legacy Javascript Applications
date: 2020-02-06
description: Getting the benefits of typescript without adding a compilation step to your toolchain
category: javascript
---

If you're in the (un)fortunate position of maintaining a legacy javascript application, you may find yourself longing for the benefits of a modern javascript toolchain. 

In this post, we'll look at how we can get some of the benefits of typescript without adding a compilation step to your development workflow.


### Add type definitions to your dev dependencies

Installing type definitions for your dependencies can give you intellisense in editors like vs code regardless of whether you're actually using typescript.

Before - no autocomplete is available for jquery:
![VS code without type definitions](/images/packagejsonbefore.gif)

After - we get intellisense just by installing the type definitions for our dependencies:
![VS code with type definitions](/images/packagejsonafter.gif)



### Add a jsconfig.json file

Adding a jsconfig.json file gives you some basic type checking which should prevent trivial bugs which would otherwise fly under the radar.

Before - no method overload checking or variable type re-assignment checking:
![VS code without jsconfig.json](/images/jsconfigbefore.gif)

After - we see errors when trying to call methods that don't exist or if we're calling functions with the incorrect number of arguments. Variable type re-assignment also triggers an error:
![VS code without jsconfig.json](/images/jsconfigafter.gif)


### Add jsdoc for improved typing

If you want to specify types without having to compile your code through `tsc`, you can use jsdoc.

Before - function defaults to accepting an argument with `any` type
![VS code without jsdoc](/images/jsdocbefore.gif)

After - function can only accept a string and returns a string array
![VS code with jsodc](/images/jsdocafter.gif)

### Run tsc in your ci/cd pipeline

Up until now, we were able to reap a bunch of the benefits of typescript without actually using typescript but most of the behavior above comes from vs code. Technically, all of the "errors" above are still valid javascript.
Even if you don't have or want a build step impacting your day-to-day development, you can prevent the above errors from entering your repositories main/trunk by running your code through the typescript compiler via your ci/cd pipeline at merge time.

Simply run `tsc --project jsconfig.json`.
