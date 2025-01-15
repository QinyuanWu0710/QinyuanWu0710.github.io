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

<h1>Latest Blog Posts</h1>
<ul>
  {% assign blogs = site.blogs %}
  {% for post in blogs %}
    <li><a href="{{ post.url }}">{{ post.title }}</a></li>
  {% endfor %}
</ul>
