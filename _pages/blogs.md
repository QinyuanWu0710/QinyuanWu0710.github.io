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

# Latest Blog Posts

{% for post in site.blogs %}
  - [{{ post.title }}]({{ post.url }})
{% endfor %}