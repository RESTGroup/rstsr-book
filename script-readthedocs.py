with open("docusaurus.config.js", "r") as f:
    token = f.read()

token = token.replace("baseUrl: '/rstsr-book/'", "baseUrl: '/latest/'")

with open("docusaurus.config.js", "w") as f:
    f.write(token)
