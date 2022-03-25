---
title: Modern Full Stack Development
date: 2021-10-10
description: Modern Full Stack Development Summary
category: book summary
type: notes
---

# Node

- single-threaded, event-driven, with background workers. Node spawns a thread for I/O, event queue on main thread handles responses.
- Uses V8 JS engine


### NPM

```bash
npm init

npm install # install stuff in package.json
npm install <packagename>
npm install --save <packagename> # install and update package.json
npm install --save <packagename> # update devDependencies in package.json
# updates respect semver version
npm update <packagename>
npm update # update all
npm uninstall <packagename>

# report dependencies with vulnerabilities
npm audit
npm audit fix
npm audit fix --dry-run

# reorg node_modules to remove dup shared packages
npm dedupe

npm ls --depth=0
```

### SemVer

major.minor.patch

- major: backwards-incompatible changes
- minor: backwards-compatible + bug fixes
- patch: bug fixes
 
- for "~" (ie ~1.0.1), npm will grab latest patch
- for "^" (ie ^1.0.1), npm will grab latest minor
- for "*", npm will grab latest version

### Axios Tips

request is sent in application/json. If you need form-urlencoded, do this in a browser:

```js
const params = new URLSearchParams();
params.append('param1', 'value1');
params.append('param2', 'value2');
axios.post('/foo', params);
```

Cancel tokens:

```js
const CancelToken = axios.CancelToken;
const source = CancelToken.source();

axios.get('/user/12345', {
  cancelToken: source.token
}).catch(function (thrown) {
  if (axios.isCancel(thrown)) {
    console.log('Request canceled', thrown.message);
  } else {
    // handle error
  }
});

axios.post('/user/12345', {
  name: 'new name'
}, {
  cancelToken: source.token
})

// cancel the request (the message parameter is optional)
source.cancel('Operation canceled by the user.');
```

Client with shared props
```js
// Set config defaults when creating the instance
const instance = axios.create({
  baseURL: 'https://api.example.com'
});

// Alter defaults after instance has been created
instance.defaults.headers.common['Authorization'] = AUTH_TOKEN;
```

Questions:
does npm update/uninstall require --save flag to update package.json?
what is the default behavior of npm install for versioning? does it at ~ or ^?
serve file from filesystem without letting user traverse directories - create read/write file templates
http 1 to 2

CHAPTER 3