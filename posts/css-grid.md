# CSS Grid

2d grid layout system (flexbox is 1d)

### Terminology

Grid container - direct parent of all grid items. 

`display: grid/inline-grid`

`grid-template-columns: [lineName(s)] size [lineName(s)] size;`

`grid-template-rows: [lineName(s)] size [lineName(s)] size`

`grid-template-area: "name" | . (empty space) | none`

```css
.container {
    display: grid;
}
```

Grid item - direct descendants of the grid container. Has `grid-column/row-start/end: number/name/auto`

