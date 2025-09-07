---
layout: default
permalink: /blogs/
title: blogs
nav: true
nav_order: 1
pagination:
  enabled: true
  collection: blogs
---

<!-- <h1 class="text-4xl font-semibold text-center mb-8">Latest Blog Posts</h1>

<!-- KaTeX (fast, no layout shift) -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/katex.min.css" integrity="sha384-IiFML8y2kC6K7m0gK2b0N1vF6wthYq2Hh1d9JmE6s2d55eV7Yy1m9Q6u+z0WkQf4" crossorigin="anonymous">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/katex.min.js" integrity="sha384-W8rN2a6g0Hgc6G5j1X0p7j1j7xv7oK2o+GXl6gAvE1sJk0aNw0K0gYo3nHk4q9kF" crossorigin="anonymous"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/contrib/auto-render.min.js" integrity="sha384-mll67bS5pQmB3X9XnN3s7f1U7xQm9q7m6pWwTg5lZ7x3q0dsz2zTq3O2q3w3s8yN" crossorigin="anonymous"></script>
<script>
  document.addEventListener("DOMContentLoaded", function() {
    renderMathInElement(document.body, {
      delimiters: [
        {left: "$$", right: "$$", display: true},
        {left: "\\[", right: "\\]", display: true},
        {left: "$",  right: "$",  display: false},
        {left: "\\(", right: "\\)", display: false}
      ],
      throwOnError: false
    });
  });
</script>


<div class="container mx-auto px-4">
  <div class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-8">
    {% assign blogs = site.blogs | sort: "date" | reverse %}
    {% for post in blogs %}
      <div class="bg-white p-4 rounded-lg shadow-lg">
        <h2 class="text-xl font-semibold text-blue-600">{{ post.title }}</h2>
        <p class="text-gray-600">{{ post.excerpt | truncatewords: 20 }}</p>
        <a href="{{ post.url }}" class="text-blue-500 hover:underline mt-2 inline-block">Read More</a>
        {% if post.tags %}
          <p class="mt-2 text-sm text-gray-500">Tags: {{ post.tags | join: ", " }}</p>
        {% endif %}
      </div>
    {% endfor %}
  </div>
</div> -->
