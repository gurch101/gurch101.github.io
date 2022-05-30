---
title: Express.js
date: 2021-10-03
description: Express.js Summary
category: summary
type: notes
---

### Routing

```js
app.METHOD(PATH, (req, res) => {
  res.send("hello world");
});

app.get("/users/:userId", (req, res) => {
  // req.params == path params
  // req.query == query params
  // req.body == req body
  res.send(req.params.userId);
});

app.get("/users/:userId(\\d+)", (req, res) => {});

const router = express.Router();

router.get("/", (req, res) => {});

app.use("/users", router);
```

### Response Methods

```js
// prompt file download
res.download();

// send json response
res.json();

res.jsonp();

res.redirect();

// send response of various types
res.send();

// send octet stream
res.sendFile();
```

### Static Content

```js
// accessible via http://localhost:3000/css/style.css
// lookup is relative to static directory
// can add multiple by multiple calls to app.use
// use a reverse proxy for better results
// call to express.static is relative to directory from where node is launched
const path = require("path");
app.use(express.static(path.join(__dirname, "public")));

// alt
// accessible via http://localhost:3000/static/css/style.css
app.use("/static", express.static("public"));
```

### Error Handling

Default error handling middleware function is added at the end of the middleware function stack. If you pass an error to `next()` and do not handle it in a custom error handler, the default error handler will write the error with a stacktrace in dev environments, sets res.status to err.statusCode, res.statusMessage based on the status code, body will be err.stack in dev/statusMessage in prod.

If anything is passed to next, besides 'route', express goes straight to error handling

Custom error handler

```js
function errHandler(err, req, res, next) {
  if (res.headersSent) {
    return next(err);
  } else if (req.xhr) {
    res.status(500).send({ error: "some error" });
  }
  res.status(500).send("Error");
}

// define after all other app.use/route setup
app.use(errHandler);
```

### Middleware

functions that have access to the request, response, and next function in the apps request-response cycle.
Middleware functions loaded first are executed first.

Configurable middleware

```js
module.exports = function (options) {
  return function (req, res, next) {
    // do stuff based on options
    next();
  };
};
app.use(customMiddleware({ option1: "foo" }));
```

Route-specific middleware

```js
app.use("/api", (req, res, next) => {
  const key = req.query["api-key"];
  // validate key
  if (!valid(key)) return next({ status: 401 });
  req.key = key;
  next();
});
```

### Production Best Practices

Use gzip. Better to put gzip in reverse proxy for scale.

```js
const compression = require("compression");
app.use(compression());
```

Dont use synchronous functions. Use `--trace-sync-io` to print a warning whenever your app uses a synchronous API

Use a logging library

set NODE_ENV to "production"

Ensure your app auto restarts - PM2, Forever

Run app in a cluster - node cluster module, node-pm, cluster-service, PM2

### Security Best Practices

Use secure, http only cookies

Use a rate limiter around auth endpoints

use csurf for cookies

use npm audit
