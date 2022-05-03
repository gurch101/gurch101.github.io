---
title: React
date: 2021-10-03
description: React Summary
category: summary
type: notes
---

### Components

A component that accepts props and returns a description of its UI

### Tips

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

### React Router

```js
<Router>
  <Link to="/foo"></Link>
  <Routes>
    <Route path="/foo" element={<FooPage />}>
  </Routes>
</Router>
```


### useState

When state changes, react needs to run the component functions that use that state. This differs from the class-based setState - in classes, setState accepts an object with just the properties you want to update and then merges them with the rest. The useState hook replaces the previous state value with the new value passed to the function.


```js
const [state, setState] = useState(initialVal);


setState(someVal);

setState(latestState => ({
  ...latestState,
  someVal
}))
```

If its expensive to set `initialVal`, pass a function to `useState`

```js
const [state, setState] = useState(() => someExpensiveFunc());
```

If you need to compute next state val from current state val, pass a function to `setState`.  We need to do this because React batches updates together to ignore redundant updates.

```js
setState(state => (state + 1));
```

### useReducer

When multiple state values change together, use a reducer (ie ajax - loading/error/data). A reducer function accepts a state value and an action value and returns the new state value.


```js
function reducer(state, action) {
  switch(action.type) {
    case "SET_GROUP":
      return {
        ...state,
        group: action.payload,
        bookableIndex: 0
      }
    case "SET_BOOKABLE":
      return {
        ...state,
        bookable: action.payload
      }
    case "NEXT_BOOKABLE":
      const count = state.bookables.filter(b => b.group === state.group).length;
      return {
        ...state,
        bookableIndex: (state.bookableIndex + 1) % count
      }
    default:
      return state;
  }
}


const initialState = {
  group: "Rooms",
  bookableIndex: 0,
  hasDetails: true,
  bookables
};

// alt: const { {group, bookableIndex, bookables, hasDetails }, dispatch } = useReducer(reducer, initialState);
const [state, dispatch] = useReducer(reducer, initialState);
const {group, bookableIndex, bookables, hasDetails} = state;

function changeGroup(e) {
  dispatch({
    type: "SET_GROUP",
    payload: e.target.value
  })
}

// if initialState needs expensive computation, getInitialState is passed someVal and should return initialState
const [state, dispatch] = useReducer(reducer, someVal, getInitialState)
```

#### useEffect

When your component has a side effect, use the `useEffect` hook. Called after rendering. Common side effects:
- interacting with dom directly
- working with timers
- using local storage
- ajax


```js
// run on every render
useEffect(() => {

})

// run on component mount only
useEffect(() => {
  function handleResize() {
    setSize({
      width: window.innerWidth,
      height: window.innerHeight
    });
  }

  window.addEventListener('resize', handleResize);

  // called on unmount or if the effect runs again
  return () => window.removeEventListener('resize', handleResize);
}, []);

useEffect(() => {
  fetch("/api/users")
    .then(resp => resp.json())
    .then(data => setUsers(data));
}, []);

// since react expects a cleanup function as a return value, use async internally
useEffect(() => {
  async function getUsers() {
    const resp = await fetch("/api/users");
    const data = await resp.json();
    setUsers(data);
  }

  getUsers();
}, [])

// run when dependencies change - deps can be props, state
useEffect(() => {

}, [dep1, dep2]);
```

### useLayoutEffect

Runs synchronously after React updates the DOM but before the browser repaints. Can be used to avoid flashes of changing state. Use in place of `useEffect` if you see FOUC.

### useRef

if you need to update state without causing a re-render, `useRef`. Useful to store timer id's, DOM element references.

```js
// ref.current will equal initialVal
const ref = useRef(initialVal);

const incRef = () => ref.current++;


const timerRef = useRef(null);

useEffect(() => {
  timerRef.current = setInterval(() => {
    dispatch({ type: 'NEXT_BOOKABLE' });
  }, 3000);

  return () => clearInterval(timerRef.current);
}, []);


const nextButtonRef = useRef();

function doStuff() {
  nextButtonRef.current.focus();
}

return (
  <button ref={nextButtonRef}>NEXT</button>
)
```

### State Management

When different components use the same data to build their UI, the most explicit way to share that data is to pass it as props from parent to child. It is common to lift state up to a common ancestor to make it more widely available.

### useCallback

avoid redefining callbacks with the `useCallback` hook. Functions from useState/useReducer are guaranteed to have stable identity. By default, our own functions will be re-defined on every call to the component.

```js
const stableFunction = useCallback(functionToCache, [depList]);
```

### useMemo