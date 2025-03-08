---
title: Showcase
description: "Docs websites powered by Hugo Blox."
type: landing

sections:
  - block: hero
    content:
      title: Get Inspirated
      text: 'Get inspired by exploring conferences with OpenVQE'
      gallery:
        - album: showcase
          image: "x4.jpeg"
          caption: "Conference Showcase 1"
        - album: showcase
          image: "x5.jpeg"
          caption: "Conference Showcase 2"
        - album: showcase
          image: "x6.jpeg"
          caption: "Conference Showcase 3"
        - album: showcase
          image: "x7.jpeg"
          caption: "Conference Showcase 4"
      primary_action:
        icon: brands/linkedin
        text: Check our Author LinkedIn site
        url: "https://www.linkedin.com/in/mohammad-haidar-3930041a4/recent-activity/all/"
      secondary_action:
        icon: brands/git
        text: Explore the github OpenVQE
        url: https://github.com/OpenVQE/OpenVQE
    design:
      background:
        gradient_end: '#1976d2'
        gradient_start: '#004ba0'
        text_color_light: true
      spacing:
        padding: [2rem, 0, 2rem, 0]
        margin: [0, 0, 0, 0]
      view: showcase
      columns: '2'
      css_style: 'background-position: center; background-size: cover;'

  - block: collection
    content:
      filters:
        folders:
          - showcase
    design:
      view: card
      columns: '2'
      spacing:
        padding: ['3rem', 0, '6rem', 0]
---