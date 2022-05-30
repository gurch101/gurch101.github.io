---
title: Nginx Summary
date: 2020-03-02
description: Ngnix cheat sheet and summary
category: summary
type: notes
---

Nginx has one master process and several worker processes. The master process reads/evaluates config and maintains workers.
Workers process requests.

### CLI

`nginx -s [stop|quit|reload|reopen]`

stop - fast shutdown
quit - graceful shutdown

### Configuration

Config lives in `/etc/ngnix`

```
http {
    server {
        location / {
            root /data/www
        }
    }
}
```

config changes can be applied with `nginx -s reload`