---
title: Cloud Native Patterns
date: 2021-10-03
description: Cloud Native Patterns Summary
category: book summary
type: notes
---

Cloud = where it runs; Cloud-native = how it runs

Modern apps need rapid iteration, fast releases, zero downtime and an increase in the volume and variety of devices that connect to it.

Cloud-native applications remain stable even when the infra they're running on is constantly changing/failing

### Chapter 2: Running Cloud-Native Applications in Production

In most orgs, getting software deployed to prod is challenging - process designed to reduce risk and improve efficiency has the opposite effect

inconsistencies in artifacts, configuration, and environments lead to system instability

goal should be easy and frequent releases into production

continuous delivery = the newest possible version of the software is deployable at any time (trunk is always deployable)

- advantages: can deploy at any time (first mover advantage), allows you to gather feedback early. If you miss dates,
                you can release less features

traditional delivery = SDLC is front-loaded with development, followed by extensive testing and packaging
    
- disadvantages: if you miss dates, testing is compressed (at the expense of quality) or you push dates

Before: tested extensively before going to production and were left scrambling on go-live.
After: Plan for failure and intentionally create a retreat path to make failures non-events. Monitor for crashes, latency changes, changes in click-through rates, etc.