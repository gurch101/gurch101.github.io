---
title: HTTP Header Review
date: 2021-10-10
description: HTTP Header Summary
category: summary
type: notes
---


### Content-Security-Policy

`Content-Security-Policy: \<policy-directive\>; \<policy-directive\>`

control resources the user agent is allowed to load for a given page by specifying server origins and script endpoints. Helps guard against cross-site scripting attacks.

Can control image/script/style/font sources, iframes. Can also upgrade-insecure-requests to instruct the user agent to use https for all http urls.

connect-src: what ajax/websockets can do
font-src: @font-face
default-src/script-src/style-src/img-src/media-src/frame-src
form-action (form actions) /child-src (iframe)

frame-ancestors 'none' is roughly equivalent to X-Frame-Options: DENY

navigate-to to control where a website can go via links or window.location

Use Content-Security-Policy-Report-Only: report-to /some-endpoint so that policy failures are logged

Possible values:
- none: prevent loading resource from any source
- self: allow from same scheme + host + port
- data: allow loading via data urls
- <domain.com>: allow loading from given domain
- *.domain.com: allow loading from any subdomain
- https: allow loading from https only
- unsafe-inline: allow running inline source

example:

default-src 'self' = allow everything but only from same origin

```http
Content-Security-Policy: default-src 'self' cdn.example.com; connect-src 'self'
```

### Strict-Transport-Security: max-age=31536000; includeSubDomains

Ensure site is only accessed through https. http is auto upgraded.