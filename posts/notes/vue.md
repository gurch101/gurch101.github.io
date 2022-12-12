---
title: Vue Notes
date: 2022-07-16
description: Udemy Course Notes
category: summary
type: notes
---

Can be used to control parts of HTML pages in a multi-page application or can be used for single page applications.

```html
<div id="app">
  <!-- setName automatically called with event object -->
  <!-- v-bind:value shorthand is :value -->
  <input type="text" v-on:input="setName" v-bind:value="name" />
  <!-- shorthand for input + value -->
  <input type="text" v-model="enteredValue" />
  <!-- event listeners are bound with v-on:<eventType>. Can be bound to a method or inline logic -->
  <!-- can pass params to the callback via methodName(param), if you also need the event object, use $event as a param -->
  <!-- v-on:click can be replaced with @click -->
  <button @click="addTodo">Add Todo</button>
  <!-- v-if controls whether element is in dom, v-show sets display:none -->
  <!-- use v-if by default, v-show if visibility toggles alot -->
  <p v-if="todos.length === 0">No goals have been added yet</p>
  <!-- v-else needs to be a direct neighbor of v-if, v-else-if can also be used -->
  <ul v-else>
    <!-- use {{ }} to interpolate between tags -->
    <!-- can also get index by v-for="(item, idx) in goals" -->
    <!-- can also iterate over objects v-for="value in obj" or (value, key) in obj -->
    <!-- can also iterate from 1 to n with v-for="num in 10" -->
    <li v-for="(todo, index) in todos" :key="todo" @click="removeGoal(index)">
      {{ todo }}
    </li>
  </ul>
  <!-- use v-bind to interpolate on an attribute -->
  <a v-bind:href="vueLink">Help</a>
  <p v-html="htmlText"></p>

  <!-- event modifiers added with .<modifier> prevent = e.preventDefault(), stop = e.stopPropagation() -->
  <!-- click.right for right mouse click -->
  <!-- keyup.enter for enter key press -->
  <form v-on:submit.prevent="submitForm">
    <input type="text" />
    <button>Sign Up</button>
  </form>

  <!-- only evaluated on creation -->
  <p v-once>Starting link {{vueLink}}</p>
  <!-- computed properties are not called with () -->
  <p>{{ fullname }}</p>
  <!-- style and class attributes can take {} on bind -->
  <!-- for class, key is className, value is boolean -->
  <!-- hardcoded classes will be merged with dynamically bound classes -->
  <p
    :style="{borderColor: 'red'}"
    class="demo"
    :class="{active: boxASelected}"
  ></p>

  <!-- can also use computed property -->
  <p :class="boxAClasses"></p>
</div>
```

```js
Vue.createApp({
  // reactive state
  data() {
    return {
      todos: [],
      enteredValue: "",
      name: "",
      vueLink: "http://www.google.com",
      htmlText: "<h1>Hello World</h1>",
      boxASelected: true,
    };
  },
  // functions accessible in vue-controlled HTML. Use for event binding
  methods: {
    addGoal() {
      this.todos.push(this.enteredValue);
      this.enteredValue = "";
    },
    removeGoal(idx) {
      this.todos.splice(idx, 1);
    }
    setName(e) {
      this.name = e.target.value;
    },
  },
  // functions that are only called when dependencies change. Use if you need to compute an output value.
  computed: {
    fullname() {
      return `Hello ${this.name}`;
    },
    boxAClasses() {
      return { active: this.boxASelected };
    },
  },
  // functions that are called only when dependencies change. Use if you need to update data.
  watch: {
    // called whenever name changes - can watch data or computed
    name(newvalue, oldvalue) {
      this.fullname = "hello " + newvalue;
    },
  },
}).mount("#app");
```

### File Structure

Use .vue files

```html
<template></template>
<script>
  export default {
    /* component def */
  };
</script>
<style scoped></style>
```

In index.js, create app with main component

```js
import App from "./App.vue";

Vue.createApp(App).mount("#app");
```

### Components

Vue files can contain just `<template>` and it works.

```html
<template>
  <section id="app">
    <ul>
      <!-- v-bind is needed for any non-string or dynamic prop -->
      <friend-contact
        v-for="friend in friends"
        :friend="friend"
        :key="friend.id"
        @toggle-favorite="toggleFavorite"
      ></friend-contact>
    </ul>
  </section>
</template>
```

```js
const app = Vue.createApp({
  data() {
    return {
      friends: [{
        id: '123',
        name: 'John Doe',
        phone: '1234567',
        isFavorite: false
      }, {
        id: '234',
        name: 'Jane Doe',
        phone: '1234568',
        isFavorite: true
      }];
    }
  },
  methods: {
    toggleFavorite(id) {
      const friend = this.friends.find(friend => friend.id === id);
      friend.isFavorite = !friend.isFavorite;
    }
  }
})

app.component('friend-contact', {
  template: `
  <li>
      <h2>{{ friend.name }}</h2>
      <button @click="toggleDetails">Toggle Details</button>
      <button @click="toggleFavorite">Toggle Favorite</button>
      <p v-show="detailsAreVisible"><strong>Phone</strong> {{ friend.phone }}</p>
    </li>
  `,
  // instead of an array of props, can use an object of the form
  /*
  dataType can be String, Number, Boolean, Array, Object, Date, Function, Symbol
  props: {
    propName: dataType,
    propName: {
      type: dataType,
      required: true/false,
      default: defaultVal || function() {},
      validator: function(val) {}
    }
  }
  */
  props: [
    // camel case as prop here but in template its dashes
    // used similarly as data
    // props should not be mutated - if you need to, set data to prop value for initial value
    'friend'
  ],
  // custom events the component emits
  /*
  can also be:
  emits: {
    'toggle-favorite': function(id){
      if(id) {
        return true;
      } else {
        return false;
      }
    }
  }
  */
  emits: ['toggle-favorite'],
  data() {
    return {
      detailsAreVisible: false
    }
  },
  methods: {
    toggleDetails() {
      this.detailsAreVisible = !this.detailsAreVisible;
    },
    toggleFavorite() {
      // always use event names with dashes
      // can emit straight from the template as well
      this.$emit('toggle-favorite', this.friend.id);
    }
  }
});

app.mount('#app');
```

### Component Registration

Global registration. All components are loaded immediately. Use for general-purpose components that are used across the app.

```js
const app = createApp(App);

app.component("user-info", UserInfo);
```

Local registration

```html
<script>
  import UserInfo from "./userInfo.vue";

  export default {
    components: {
      UserInfo, // allows use as UserInfo or user-info in template
    },
    data() {},
  };
</script>
```

### Styling

style in a component applies to entire app by default

`style scoped` only impacts styles in same .vue file - not any children

### Slots

Use component as a wrapper around another template

```html
<template>
  <section>
    <slot>
      <header>
        <!-- only render this slot if content is provided -->
        <slot name="header" v-if="$slots.header">
          <h2>
            Some default content that is overridden if parent provides header
          </h2>
        </slot>
      </header>
    </slot>
    <!-- default unnamed slot -->
    <slot></slot>
  </section>
</template>
<script>
  export default {
    mounted() {
      // has .header, .default
      console.log(this.$slots);
    },
  };
</script>
<style scoped>
  section {
    background-color: red;
  }

  header {
    background-color: blue;
  }
</style>
```

Parent component

```html
<template>
  <card>
    <!-- can also use v-slot:header -->
    <template #header>
      <h3>Hello</h3>
    </template>
    <template #default>
      <p>World</p>
    </template>
  </card>
</template>
```

### Prop Fallthrough

props added to a custom component are automatically bound to root component of the template

ie:

```html
<base-button @click="doStuff" type="submit">Hello</button>
```

This component will automatically bind click/type to the button:

```html
<button>
  <slot></slot>
</button>
```

### Bind All Props

You can bind all properties in an object using `v-bind`

```html
<user-data v-bind="person"></user-data>
```

props will be available without object notation

```html
<h2>{{firsName}} {{lastName}}</h2>
```

### Scoped Slots

Pass data from where you define the slot to the parent where the slot markup is defined

```html
<template>
  <ul>
    <li v-for="goal in goals" :key="goal"><slot :goal="goal"></slot></li>
  </ul>
</template>
```

Parent

```html
<my-list>
  <template #default="slotProps">
    <h2>{{ slotProps.goal }}</h2>
  </template>
</my-list>
```

### Dynamic Components

```html
<template>
  <button @click="setSelectedComponent('first')">First</button>
  <button @click="setSelectedComponent('second')">Second</button>
  <!-- keep-alive doesnt destroy component when its deleted -->
  <keep-alive>
    <!-- dynamic component -->
    <component :is="selectedComponent"></component>
  </keep-alive>
</template>
<script>
  export default {
    data() {
      return {
        selectedComponent: "",
      };
    },
    methods: {
      setSelectedComponent(cmp) {
        this.selectedComponent = cmp;
      },
    },
  };
</script>
```

### Teleport

Put elements anywhere in dom

```html
<template>
  <div>
    <input type="text" v-model="txt" />
    <button @click="submit">Submit</button>
    <!-- teleport is a built-in vue component. Takes a to prop that is a dom selector -->
    <teleport to="body">
      <error-alert v-if="isInvalid">
        <h2>Input is invalid</h2>
      </error-alert>
    </teleport>
  </div>
</template>
<script>
  export default {
    data() {
      return {
        txt: "",
        isInvalid: false,
      };
    },
    methods: {
      submit() {
        if (this.txt === "") {
          this.isInvalid = true;
        }
        this.isInvalid = false;
      },
    },
  };
</script>
```

### Fragments

vue 3 allows multiple top level elements

### Style Guide

Vue publishes a official style guide

### Forms

```html
<template>
  <form @submit.prevent="submitForm">
    <input name="name" type="text" v-model.trim="name" />
    <input name="age" type="number" v-model="age" />
    <select name="referrer" v-model="referrer">
      <option value="google">Google</option>
      <option value="wom">Word of Mouth</option>
    </select>
    <input type="checkbox" name="interest" value="news" v-model="interest" />
    News
    <input
      type="checkbox"
      name="interest"
      values="tutorials"
      v-model="interest"
    />
    Tutorials
    <input type="radio" name="how" value="video" v-model="how" /> Video Courses
    <input type="radio" name="how" value="blogs" v-model="how" /> Blogs
    <input type="checkbox" name="terms" v-model="confirm" />Confirm to terms?
    <rating-control v-model="rating"></rating-control>
  </form>
</template>
<script>
  import RatingControl from './RatingControl.vue';

  export default {
    components: {
      RatingControl
    },
    data() {
      return {
        name: '',
        age: null,
        referrer: 'wom',
        how: null,
        interest: [],
        confirm: false
      }
    }
    methods: {
      submitForm() {
        console.log(this.user);
        console.log(this.age);
        console.log(this.referrer);
        this.user = '';
        this.age = null;
        this.referrer = 'wom';
      }
    }
  }
</script>
```

```html
<template>
  <ul>
    <li :class="{selected: modelValue === 'poor'}" @click="activate('poor')">
      Poor
    </li>
    <li @click="activate('average')">Average</li>
    <li @click="activate('great')">Great</li>
  </ul>
</template>
<script>
  export default {
    // convention to allow parent to use v-model
    props: ["modelValue"],
    emits: ["update:modelValue"],
    methods: {
      activate(option) {
        this.$emit("update:modelValue", option);
      },
    },
  };
</script>
```

### Provide/Inject

Pattern to provide data and inject it to a component without explicitly passing it as a prop. Works for parent-child components only. Use sparingly since it can be difficult to understand where provided values are used

Parent:

```html
<script>
  export default {
    data() {
      return {
        users: [
          {
            name: "foo",
          },
        ],
      };
    },
    // can also just be an object
    provide() {
      return {
        users: this.users,
        addUser: this.addUser,
      };
    },
    methods: {
      addUser() {},
    },
  };
</script>
```

Child:

```html
<template>
  <button @click="addUser"></button>
</template>
<script>
  export default {
    inject: ["users", "addUser"],
  };
</script>
```

### Internals

Vue wraps data objects with js proxies.

```js
const data = {
  message: "Hello!",
};

const handler = {
  set(target, key, value) {
    // target is the object that was wrapped
    // key is the attribute that was changed
    // value is the new value

    target[key] = value;
  },
};

const proxy = new Proxy(data, handler);

data.message = "Greetings!";
```

You can use `Vue.createApp` to create multiple apps on one site. Each app is standalone.

Vue apps/components can also have a template property

```js
Vue.createApp({
  template: '<div>Hello World</div>
});
```

```js
export default {
  template: "<div>Hello World</div>",
};
```

Vue uses a virtual dom to detect changes and only changes are put in the browser dom.

### Lifecycle

Lifecycle methods are not put in the `method` stanza

- beforeCreate
- created
- beforeMount (template compiled)
- mounted (in browser dom)
- beforeUpdate
- updated (change is visible in dom)
- beforeUnmount
- unmounted

### Refs

```html
<template>
  <input type="text"  ref="userText" />
  </template>
  <script>
    export default {
      methods: {
        printText() {
          console.log(this.$refs.userText.value);
        }
      }
    }
```

### Composition API

options API = data/methods/watchers/computed

For bigger apps, composition API enables data/methods/watchers to be grouped together by feature in a setup method and improves re-use.

```html
<template>
  <div>
    <!-- don't need .value in the template -->
    <h1>{{ userName }} {{user.name}}</h1>
    <h3>{{ age }} {{user.age}}</h3>
    <button @click="setAge()">Update Age</button>
    <input type="text" ref="lastNameInput" />
  </div>
</template>
<script>
  // also has onBeforeMount, onMounted, onBeforeUpdate, onUpdated, onBeforeUnmount, onUnmounted
  import { ref, reactive, toRefs, computed, watch, provide } from "vue";

  export default {
    // components/props are same as options
    // setup only runs once - replaces beforeCreate/created
    // can use context.$emit
    setup(props, context) {
      const userName = ref("default"); // create a reactive value
      const age = ref(31);
      // value is retrieved by lastNameInput.value.value
      const lastNameInput = ref(null);

      const user = ref({
        name: "default",
        age: 31,
      });

      // reactive only works with objects; doesn't require .value to access
      const rUser = reactive({
        name: "default",
        age: 31,
      });

      // makes the object AND each value inside the object a ref
      const trUser = toRefs({
        name: "default",
        age: 31,
      });

      // treated as a readonly ref - can use via label.value
      const label = computed(function () {
        return `${userName.value} - ${age.value}`;
      });

      // can pass an array of dependencies to watch
      // function will get get array of newVals, oldVals
      watch(userName, function (newVal, oldVal) {
        console.log("new username", newVal);
      });

      // to watch for changes in a prop
      const { user } = toRefs(props);
      watch(user, function (newVal, oldVal) {});

      setTimeout(() => {
        userName.value = "some new value";
        user.value.name = "some other value";
        rUser.name = "some other value";
        rUser.age = 41;
      }, 2000);

      function setAge() {
        age = 42;
      }

      // use inject function in consumer, ie const age = inject('userAge');
      // you should only change injected values where you provide them
      provide("userAge", age);

      // anything returned here is available in the template
      return {
        userName,
        age,
        user,
        setAge,
        label,
        lastNameInput,
      };
    },
  };
</script>
```

Alternative syntax:

```html
<script setup>
  const userName = ref("default"); // create a reactive value

  setTimeout(() => {
    userName.value = "some new value";
  }, 2000);
</script>
```

### Mixins

take the component code config (data, methods, watchers) and put it in a js file.

mixin:

```js
// options exposed by mixin are merged with options exposed with component. Component option overrides the value set by the mixin. If lifecycle methods exist in both, the mixin one runs first, then the component.
export default {
  data() {
    return {
      alertIsVisible: false,
    };
  },
  methods: {
    showAlert() {
      this.alertIsVisible = true;
    },
    hideAlert() {
      this.alertIsVisible = false;
    },
  },
};
```

component:

```js
export default {
  components: {
    UserAlert,
  },
  mixins: [myMixin],
};
```

Mixins can be registered globally via `app.mixin(myMixin)`. This adds the mixin to every component.

### Mixins using Composition API

```js
import { ref } from "vue";

export default function useAlert() {
  const alertIsvisible = ref(false);

  function showAlert() {
    alertIsVisible.value = true;
  }

  function hideAlert() {
    alertIsVisible.value = false;
  }

  return {
    alertIsVisible,
    showAlert,
    hideAlert,
  };
}
```

In component:

```js
export default {
  setup() {
    const { alertIsVisible, showAlert, hideAlert } = useAlert();
    return {
      alertIsVisible,
      showAlert,
      hideAlert,
    };
  },
};
```

### Http Requests

post/submit data. Add `@submit.prevent` to any form.

```js
fetch(url, {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    // someJson
  }),
});
```

get data. Fetch inside of mounted/created.

```js
try {
  // response.ok - if false, server-side error
  const response = await fetch(url);
  if(response.ok) {
    const json = await response.json();
  } else {
    throw new Error('could not get data!');
  }
} catch(error => {
  // this will be a client-side error
  console.log(error);
})
```
