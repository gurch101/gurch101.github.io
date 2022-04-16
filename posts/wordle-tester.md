---
title: Solving Wordle
description: A site to try out algorithms to solve Wordle puzzles
date: 2022-03-22
category: typescript
type: post 
---

<div style="font-size: 1.25rem; font-weight: 300; border: 1px solid rgba(0, 0, 0, 0.2); border-radius: 3px; padding: 5px">
tl;dr? Try out the tester here: <a href="/apps/wt/index.html">Wordle Tester</a>
</div>

I used to play a game called Mastermind when I was in elementary school. The rules are simple:

1. One player is the code breaker, the other is the code maker
2. The code maker chooses a pattern with 4 colored pegs
3. The code breaker tries to guess the pattern in both order and color in eight to twelve turns
4. For each turn, the code maker reveals which pegs are correct in color and/or position

Sound familiar? Replace 4 colored pegs with 5 letters of the English alphabet and you get the ever-so-popular, vanilla-javascript, strictly-client-side, sold-for-millions-to-the-new-york-times, word game, Wordle. 

When Mastermind was released, computer scientists, mathematicians, and hobbyists all came up with strategies to optimally "solve" Mastermind ([1](https://www.cs.uni.edu/~wallingf/teaching/cs3530/resources/knuth-mastermind.pdf), [2](https://arxiv.org/pdf/1305.1010.pdf), [3](https://doi.org/10.1007/978-3-642-44973-4_31)). Not unlike Mastermind, developers were quick to post their own "solutions" to Wordle shortly after its release. Rather than write my own algorithm to solve Wordle puzzles, I wrote a site where you (yes *you*) can write and test your own solver: [Wordle Tester](/apps/wt/index.html). Will anyone actually use the site? Probably not :) I built it just to do some greenfield development on a mostly-vanilla javascript toy project. In this post, I'll go over a bunch of things I learned along the way.

### Lessons Learned

**SWC is awesome**. SWC is a super fast JS/TS transpiler written in Rust. Even on this small project with only a handful of files, there is a *marked* difference in build times. With next.js using SWC and vite using esbuild, it seems like issues with build times for large projects are eventually going to be a non-issue in the near future.

**Javascript tooling is still annoying**. I probably spent more time messing around with SWC, eslint, prettier, jest, and typescript knobs and switches than coding up the actual site. I wish *someone* would build a framework/library agnostic create-react-app equivalent which abstracts away all the noise and gives you a project template with sensible defaults for both browser and node environments.  A simple cli that lets you pick node/browser, js/ts, and a testing library/coverage tool that would generate a base project with the best bundler/transpiler/minifier-of-the-day, consistent import/transpilation across source/tests, out-of-the-box HMR for dev, and a reasonable eslint/prettier/editorconfig setup. It seems like I'm not the only one that would benefit from the development of a project like this - as of today, there are over 11k "typescript starter" projects on Github. I'd love to take a project like this on myself but I think I'd just be adding to the pile of 11k starter projects - it *needs* to be done by a reputable company/developer for it to gather steam. It seems like vite is the closest thing that does this but it was built specifically for front end development.

**You can inject code into your site without directly calling `eval`**. I needed some way to execute code that the user enters on the tester for each wordle "guess". While I could've just opted for `eval` which probably would've been fine for my use case (the user is executing code that they wrote themselves), I snooped around jsfiddle to see how they deal with code injection/execution. Turns out they embed the code in an iframe served by a different subdomain. Since the iframe is hosted by a different domain than jsfiddle itself, the site isn't susceptible to XSS/cookie access. While I didn't host the users code on a different subdomain since my site has no cookies, I did use an iframe because they provide some other handy features.

**IFrames have a `sandbox` attribute that lets you control what it can do**. It turns out you can restrict what iframes can do on your site using the `sandbox` attribute. If you using `sandbox=""`, the iframed content is effectively treated as a static site - no javascript, no form submissions, no popups, and no browser-level site redirections are allowed. I used `allow-scripts` to embed the users solver code and `allow-same-origin` so that I can execute it from the host site.

**Vanilla Javascript is great**. While my site isn't entirely vanilla (I use ES modules and typescript), I chose not to use any view library and instead coded directly against the DOM to keep in the spirit of the original game. With template literals, the jquery-like `querySelector`, and the ability to run tests using jsdom, I rarely felt like I was missing out on features provided by react and its ilk. The entire site being 111kb uncompressed is a nice bonus (103kb is the word list, the actual code for the site is just 8kb).

**I'm still horrible at CSS**. The memes are true - after a decade as a dev, I still google how to center things. Thank goodness for Bootstrap/tailwind.