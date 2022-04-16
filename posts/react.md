---
title: React
date: 2021-10-03
description: React Summary
category: summary
type: notes
---

- favor functional components since they have no lifecycle methods, constructors, or boilerplate.
- always name components/functions since it helps when reading the error stack
- rather than hardcoding markup, use const config objects and loops
- use `ErrorBoundary` to avoid cascading failures
- the more props a component takes, the more reasons to re-render
- if a component has multiple pieces of state, use `useReducer` instead of multiple `useState` calls

```js
const TYPES = {
  SMALL: 'small',
  MEDIUM: 'medium',
  LARGE: 'large'
}

const initialState = {
  isOpen: false,
  type: TYPES.LARGE,
  phone: '',
  email: '',
  error: null
}

const reducer = (state, action) => {
  switch (action.type) {
    ...
    default:
      return state
  }
}

function Component() {
  const [state, dispatch] = useReducer(reducer, initialState)

  return (
    ...
  )
}

```
- group components by route/module rather than container/component. State should live close to the component using it. Container/component model encourages few components to hold most state. Put UI components in a common module.
- use absolute paths instead of relative paths to enable easy project structure changes.
- watch the bundle size. Don't ship a single JS bundle. Split by route.

### SWR

Data is bound to the components that need the data, components are independent of one another. Only 1 request is sent to the API - SWR caches by key, request is deduped, cached, and shared automatically.

```js
const fetcher = (...args) => fetch(...args).then(res => res.json())

function useUser (id) {
  const { data, error } = useSWR(`/api/user/${id}`, fetcher)

  return {
    user: data,
    isLoading: !error && !data,
    isError: error
  }
}

// page component

function Page () {
  return <div>
    <Navbar />
    <Content />
  </div>
}

// child components

function Navbar () {
  return <div>
    ...
    <Avatar />
  </div>
}

function Content () {
  const { user, isLoading } = useUser()
  if (isLoading) return <Spinner />
  return <h1>Welcome back, {user.name}</h1>
}

function Avatar () {
  const { user, isLoading } = useUser()
  if (isLoading) return <Spinner />
  return <img src={user.avatar} alt={user.name} />
}

/*
if fetcher needs multiple args,
useSWR(['/api/user', token], fetchWithToken)
const { data: orders } = useSWR({ url: '/api/orders', args: user }, fetcher)
*/

import useSWR from 'swr'

function Profile () {
  const { data, mutate } = useSWR('/api/user', fetcher)

  return (
    <div>
      <h1>My name is {data.name}.</h1>
      <button onClick={async () => {
        const newName = data.name.toUpperCase()
        // send a request to the API to update the data
        await requestUpdateUsername(newName)
        // update the local data immediately and revalidate (refetch)
        // NOTE: key is not required when using useSWR's mutate as it's pre-bound
        mutate({ ...data, name: newName })
      }}>Uppercase my name!</button>
    </div>
  )
}
```

### Prefetching Data

Prefetch when the HTML loads.

```html
<head>
<link rel="preload" href="/api/data" as="fetch" crossorigin="anonymous">
</head>
```