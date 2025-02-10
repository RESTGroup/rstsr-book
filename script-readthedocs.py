with open("docusaurus.config.js", "r") as f:
    token = f.read()

token = token.replace("url: 'https://ajz34.github.io'", "url: 'https://rstsr-book.readthedocs.io'")
token = token.replace("baseUrl: '/rstsr-book/'", "baseUrl: '/v0.0.1/'")

with open("docusaurus.config.js", "w") as f:
    f.write(token)
