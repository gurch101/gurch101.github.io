---
title: Mithril.js is great
date: 2022-07-02
description: Why mithril should be more popular than it is
category: javascript draft
type: blog
---

I wish mithril.js was more popular. Not only is it smaller and more performant than comparative view libraries like react/angular/vue, it also has a far simpler (and, imo, more pleasant-to-use) API.

### Mithril is decidedly state-unaware

Unlike other view libraries, mithril is decidedly state *un*aware. Mithril doesn't care about your state and it does not need to be explicitly notified when your state changes. Instead, in mithril, re-renders are automatically triggered on every event and ajax call.

This is _notably_ different than other view libraries:

React:

```jsx
function Button() {
  // react needs a hook/call to setState/react-aware state management library
  const [count, setCount] = useState(0);
  return (
    <button onClick={() => setCount((count) => count + 1)}>
      You clicked {count} times
    </button>
  );
}
```

Mithril

```jsx
function Button() {
  // mithril doesn't care where state comes from, instead it just re-renders whenever a dom event is fired
  let count = 0;
  return {
    view: () => (
      <button onclick={() => count++}>You clicked {count} times</button>
    ),
  };
}
```

Mithril has all the benefits of hooks without any of the drawbacks. Mithril is succinct and doesn't require an understanding of `this` without relying on transpilation tricks to make concepts like hooks to work. Mithril's lifecycle methods also strike a good balance between React's class component lifecycle methods and the `useEffect` hook.

React:

Mithril:
