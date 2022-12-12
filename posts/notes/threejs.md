---
title: Threejs Summary
date: 2020-03-02
description: Threejs summary
category: summary
type: notes
---

3d library that uses webgl to draw scenes, lights, shadows, materials, textures, 3d math.

Renderer has a scene and camera

scene is composed of objects and lights

mesh objects draw a specific geometry with a specific material.

geometry represents the vertex data of objects

material represents the surface properties (color, shine, texture) to draw geometry. Main difference between three.js material types is how they react to light.

```css
html,
body {
  margin: 0;
  height: 100%;
}

#c {
  width: 100%;
  height: 100%;
  display: block;
}
```

```js
import * as THREE from "./three.module.js";

function main() {
  const canvas = document.querySelector("#c");
  const renderer = new THREE.WebGLRenderer({ canvas });

  // frustrum - 3d shape like a pyramid with the tip sliced off
  // anything inside the frustrum is drawn, everything outside is not
  const fieldOfViewInDegrees = 75;
  const aspectRatio = 2;
  const near = 0.1;
  const far = 5;

  // camera defaults to looking down the -Z access with +Y up
  const camera = new THREE.PerspectiveCamera(
    fielOfViewInDegrees,
    aspectRatio,
    near,
    far
  );
  camera.position.z = 2;

  const scene = new THREE.Scene();

  const boxWidth = 1;
  const boxHeight = 1;
  const boxDepth = 1;
  const geometry = new THREE.BoxGeometry(boxWidth, boxHeight, boxDepth);
  const material = new THREE.MeshPhongMaterial({ color: 0x44aa88 });
  const cube = new THREE.Mesh(geometry, material);

  scene.add(cube);

  const color = 0xffffff;
  const intensity = 1;
  const light = new THREE.DirectionalLight(color, intensity);
  light.position.set(-1, 2, 4);

  scene.add(light);

  renderer.render(scene, camera);

  function resizeRendererToDisplaySize(renderer) {
    const canvas = renderer.domElement;
    const pixelRatio = window.devicePixelRatio;
    const width = (canvas.clientWidth * pixelRatio) | 0;
    const height = (canvas.clientHeight * pixelRatio) | 0;
    const needResize = canvas.width !== width || canvas.height !== height;
    if (needResize) {
      renderer.setSize(width, height, false);
    }
    return needResize;
  }

  function render(time) {
    time * -0.001; // convert to seconds;

    if (resizeRendererToDisplaySize(renderer)) {
      const canvas = renderer.domElement;
      camera.aspect = canvas.clientWidth / canvas.clientHeight;
      camera.updateProjectionMatrix();
    }

    // rotation in radians
    cube.rotation.x = time;
    cube.rotation.y = time;

    renderer.render(scene, camera);

    requestAnimationFrame(render);
  }

  requestAnimationFrame(render);
}

main();
```
