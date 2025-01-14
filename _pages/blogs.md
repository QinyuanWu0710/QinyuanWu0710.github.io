---
layout: default
permalink: /blogs/
title: blogs
nav: true
nav_order: 1
pagination:
  enabled: true
  collection: posts
  permalink: /page/:num/
  per_page: 5
  sort_field: date
  sort_reverse: true
  trail:
    before: 1 # The number of links before the current page
    after: 3 # The number of links after the current page
---

{% for blog in site.blogs %}
  <h2><a href="{{ blog.url }}">{{ blog.title }}</a></h2>
  <p>{{ blog.excerpt }}</p>
{% endfor %}

