---
title: 'Home'
date: 2023-10-24
type: landing

design:
  # Default section spacing
  spacing: "6rem"

sections:
  - block: hero
    content:
      title: OpenVQA
      text: The easy, well- structured code technical documentation solution your users to learn Quantum Computing for Quantum Chemistry
      primary_action:
        text: Get Started
        url: https://github.com/OpenVQE/OpenVQE
        icon: rocket-launch
      secondary_action:
        text: Read the docs
        url: /docs/
      announcement:
        text: "Announcing the release of version 2."
        link:
          text: "Read more"
          url: "/blog/"
    design:
      spacing:
        padding: [0, 0, 0, 0]
        margin: [0, 0, 0, 0]
      # For full-screen, add `min-h-screen` below
      css_class: ""
      background:
        color: ""
        image:
          # Add your image background to `assets/media/`.
          filename: ""
          filters:
            brightness: 0.5
  - block: stats
    content:
      items:
        - statistic: "1"
          description: |
              Open Source specified for Quantum Computing for Quantum Chemistry
        - statistic: "2"
          description: |
            GitHub stars  
            since 2021
        - statistic: "3"
          description: |
            3 published papers
    design:
      # Section background color (CSS class)
      css_class: "bg-gray-100 dark:bg-gray-800"
      # Reduce spacing
      spacing:
        padding: ["1rem", 0, "1rem", 0]
  - block: features
    id: features
    content:
      title: Features
      text: Collaborate, publish, and maintain technical knowledge with an all-in-one documentation site. Used by startups, enterprises, and researchers.
      items:
        - name: Optimized search
          icon: magnifying-glass
          description: Automatic sitemaps, easy to find the function 
        - name: Fast
          icon: bolt
          description: Super fast download with git-based platform 
        - name: Easy
          icon: sparkles
          description: Easy for people who want to learn quantum computing with the background in Chemistry
        - name: Code Structure
          icon: code-bracket
          description: Well-structured code, people are highly advice to contribute to the package
        - name: Highly Rated
          icon: star
          description: Rated 5-stars by the community.
        - name: Interprobability
          icon: rectangle-group
          description: Code with myQLM language can swap with Qiskit, Pennylane, ...
  - block: cta-card
    content:
      title: "Start Exploring with the OpenVQE Package"
      text: OpenVQE is an Open Source Variational Quantum Eigensolver package. It is an extension of the Quantum Learning Machine to Quantum Chemistry based on the tools provided in myQLM-fermion package.
      button:
        text: Get Started
        url: https://github.com/OpenVQE/OpenVQE
    design:
      card:
        # Card background color (CSS class)
        css_class: "bg-primary-700"
        css_style: ""
---
