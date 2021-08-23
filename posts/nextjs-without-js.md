---
title: Building a Next.js Site Without Any Client-Side Javascript
date: 2020-01-02
description: Using React without your users *running* React
category: javascript
---

Next.js is a React framework that lets you build static and server rendered websites by adopting a few simple conventions to hydrate your components. In doing so, you get the flexibility and familiarity of building a site with React & Javascript while also getting the performance and SEO benefits of statically pre-rendering as much of your site as possible.

By default, Next.js preloads a bunch of javascript regardless of whether your site actually needs it. Really simple sites (like this blog), generally don't need any client-side javascript magic. Statically generated emails would also benefit from avoiding client-side javascript since email clients don't execute javascript.

Turns out it's really easy to disable *all* javascript in your Next.js app (provided you are using the canary build and are comfortable using an experimental feature).

First,
```sh
npm install next@canary
```

Then, add the following to your page:
```js
export const config = {
  unstable_runtimeJS: false
}
```

That's it!