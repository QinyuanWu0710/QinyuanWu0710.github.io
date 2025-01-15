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

<h1 class="text-4xl font-semibold text-center mb-8">Latest Blog Posts</h1>
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
</div>
