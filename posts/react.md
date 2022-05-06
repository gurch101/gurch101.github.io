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

### React Query

```js
const queryClient = new QueryClient();

<QueryClientProvider client={queryClient}>
  <App/>
</QueryClientProvider>

// if initialData is not undefined, don't call server and just use initialData instead
const { data, status, error, isLoading } = useQuery([keys], () => fetch(url), { initialData: queryClient.getQueryData([keys])});
const queryClient = useQueryClient();
const data = queryClient.getQueryData([keys]);

const {mutate: createBookable, status, error} = useMutation(item => createItem("http://foo.com/bookables", item), { onSuccess: bookable => queryClient.setQueryData("bookables", old => [...(old || []), bookable] )})
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
  <Link to="/foo" replace={true} className="btn"><span>New</span></Link>
  <Routes>
    <Route path="/foo1" element={<FooPage />}>
    <Route path="/foo2">
      <FooPage />
    </Route>
    <Route path="/bar/*" element={<BarPage />}>
  </Routes>
</Router>

// BarPage.js

<Routes>
  <Route path="/:id">
    <BarDetail />
  </Route>
  <Route path="/:id/edit">
    <BarEditDetail />
  </Route>
</Routes>


// BarDetail.js
const = {id} = useParams();
const {searchParams, setSearchParams} = useSearchParams();
const navigate = useNavigate();
searchParams.get("date");

// replace true prevents browsers back button from cycling through query params
setSearchParams({id, date}, {replace: true});
function onButtonClick() {
  navigate(`/bar/${id}/edit`);
}

// RequireAuth component
function RequireAuth({ children }: { children: JSX.Element }) {
  let auth = useAuth();
  let location = useLocation();

  if (!auth.user) {
    // Redirect them to the /login page, but save the current location they were
    // trying to go to when they were redirected. This allows us to send them
    // along to that page after they login, which is a nicer user experience
    // than dropping them off on the home page.
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  return children;
}
```

### Hooks

hooks let you use state, access context, and hook into lifecycle events

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

useEffect(() => {
  let doUpdate = true;
  fetch("").then(resp => if(doUpdate) {})
  return () => doUpdate = false;
}, [])
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

avoid redoing unnecessary work if the inputs to a function are unchanged.

```js
const [sourceText, setSourceText] = useState("ball");
// passing an empty list will mean the function only runs once
const anagrams = useMemo(() => getAnagrams(sourceText), [sourceText]);
const distict = useMemo(() => getDistinct(anagrams), [anagrams]);
```

### useContext

share state like themes, localization, or user details across many components using `useContext`. Should be used for rarely changing values used by many components.

```js

const { createContext } from "react";

// can take defaultContext as a param - default will be returned even if there is no provider
const UserContext = createContext();

export default UserContext;

// in the provider component (ie app.js/router.js)

const [user, setUser] = useState();

<UserContext.Provider value={{user, setUser}}>
// user can now be made available to all components in the tree - any time the user changes, the entire tree is re-rendered
</UserContext.Provider>


// in the consuming component
const {user: loggedInUser} = userContext(UserContext)
```

To avoid full re-renders, do this:

```js
export function UserProvider({children}) {
  const [user, setUser] = useState(null);
  return (
    <UserContext.Provider value={{user, setUser}}>
      {children} // react wraps children in <Wrapper></Wrapper> - this component doesn't change when setUser is called thus avoiding a re-render
    </UserContext.Provider>
  )
}

// in the provider component
<UserProvider>
  // components
</UserProvider>
```

Above works because UserProvider accesses children as a prop and updating the state within the component doesn't change props. Since the identity doesn't change, react doesn't re-render children except for context consumers.

Best practice - use separate context providers for different pieces of info to avoid unnecessary re-renders. Put the providers as close as possible to the components using them. If there are many providers, consider creating an `<AppProvider>` that wraps them all.

Use separate contexts for a state value and its updater. When you provide `{user, setUser}`, all consumers re-render every time since its a fresh object every time. In the above case, its fine since setUser never changes but if you had a Logout button that only needed `setUser` it would make sense to split.

### Custom Hooks

Lets you re-use functionality in multiple components

```js
function useDocumentTitle(title) {
  useEffect(() => {
    document.title = title;
  }, [title]);
}

// in component

useDocumentTitle('foobar');
```

Hooks can call other hooks.

```js
const getRandomIndex = length => Math.floor(Math.random() & length);

function useRandomTitle(titles = ["Hello"]) {
  const [index, setIndex] = useState(() => getRandomIndex(titles.length));

  useDocumentTitle(title);

  return () => setIndex(getRandomIndex(titles.length));
}

// in component

const nextTitle = useRandomTitle(titles);

return (
  <button onClick={nextTitle}></button>
)
```


```js
function getSize() {
  return {
    width: window.innerWidth,
    height: window.innerHeight
  };
}

function useWindowSize() {
  const [size, setSize] = useState(getSize());
  
  useEffect(() => {
    function updateSize() {
      setSize(getSize());
    }

    window.addEventListener('resize', updateSize);

    return () => window.removeEventListener('resize', updateSize);
  }, []);
}


const { width, height } = useWindowSize();
```


```js
function useLocalStorage(key, val) {
  const [value, setValue] = useState(val);

  useEffect(() => {
    const storedValue = window.localStorage.getItem(key);

    if(storedValue) {
      window.localStorage.setItem(key, value);
    }
  }, [key]);

  useEffect(() => {
    window.localStorage.setItem(key, value);
  }, [key, val]);

  return [value, setValue];
}
```

Avoid context consumers from caring about where data comes from

```js
function useUser() {
  const user = useContext(UserContext);

  return user;
}

```


data fetching hook

```js
function useFetch(url) {
  const [data, setData] = useState();
  const [error, setError] = useState();
  const [status, setStatus] = useState();

  useEffect(() => {
    let doUpdate = true;

    setStatus("loading");
    setData(undefined);
    setError(null);

    getData(url)
      .then(data => {
        if(doUpdate) {
          setData(data);
          setStatus("success");
        }
      })
      .catch(error => {
        if(doUpdate) {
          setError(error);
          setStatus("error");
        }
      }


      return () => doUpdate = false;
  }, [url]);

  return {data: bookables = [], error, status};
}

const {data, status, error} = useFetch(url);
```

Rules
- names must start with `use`
- hooks can only be called at the top level - not inside conditions, loops, or nested functions (put the logic *inside* the hook)
- hooks can only be called from React functions (components or other hooks)


### Concurrent Mode

lets react work on multiple versions of your UI simultaneously - pausing, restarting, and discarding rendering tasks to make apps seem as responsive as possible.

### Code Splitting with Suspense

For big components that don't take part in initial user interactions, should do the following:
- load component code only when we try to render the component
- show a placeholder while the component loads
- continue rendering the rest of the app
- replace the placeholder with the component after it's loaded

Default dynamic imports:

```js
function onClick() {
  import("./myBigModule")
    .then(({ default: showMessage, sayHi }) => {
      // default exported function
      showMessage("foo", "Bar");
      // named exports
      sayHi("hello world");
    })
}
```

React dynamic import

```js
function CalendarWrapper() {
  const [isOn, setIsOn] = useState(false);
  return isOn ? <LazyCalendar> : <button onClick={() => setIsOn(true)}>Show Calendar</button>
}

// lazy is passed a function that returns a promise that resolves to a default component export. The promise is thrown and caught by the Suspense
const LazyCalendar = lazy(() => import("./Calendar.js"));

// suspense components handle pending promises that are thrown
<Suspense fallback={<div>Loading...</div>}>
  <CalendarWrapper/>
</Suspense>
```

Code splitting on routes. Routes that arent lazy are just rendered.

```js
const BookablesPage = lazy(() => import("./BookablesPage"));

return (
  <Suspense fallback={<div>Loading</div>}>
    <Routes>
      <Route path="/foo" element={<div>Foo</div>} />
      <Route path="/bookables/*" element={<BookablesPage/>} />
    </Routes>
  </Suspense>
)
```

Error boundary components can be used to catch thrown errors

```js
class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError() {
    return {hasError: true}
  }

  componentDidCatch(error, errorInfo) {
    logErrorToMyService(error, errorInfo);
  }

  render() {
    const { children, fallback = <h1>Something Went Wrong</h1> } = this.props;
    return this.state.hasError ? fallback : children;
  }
}
```

Suspense and error boundary let you decouple loading/error UI from the actual component.

### Data Fetching with Suspense

```js
const { data: bookables = [] } = useQuery("bookables", () => getData("/bookables"), { suspense: true });

function Img({src, alt, ...props}) {
  const { data: imgObject } = useQuery(src, () => new Promise(resolve => (
    const img = new Image();
    img.onload = () => resolve(img);
    img.src = src;
  ), { suspense: true}));

  return <img src={imgObject.src} alt={alt} {...props}/>;
}

<Suspense fallback={<img src={fallbackSrc} alt="fallback" />}>
  <Img src={src} alt={alt} />
</Suspense>
```

### useTransition, useDeferredValue, SuspenseList


TODO:
- notes on react-testing-library
- testing components
- testing custom hooks
- testing components using context
- read Next.js docs
- https://react2025.com/
- https://masteringnextjs.com/