# Project Greenhouse website

Build and preview from the repository root.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r website/requirements.txt
python website/build.py
python -m http.server 8000 --directory website/_site
```

Open [the local preview](http://localhost:8000/). To remove pages after deleting or renaming source files, build into a fresh output directory.

## Publish a research update

Add a Markdown file in `website/updates/` named `YYYY-MM-DD-slug.md`. The date controls ordering and the slug is the permanent URL at `/updates/slug/`.

```markdown
Title: Release title
Summary: A one-sentence summary for the update index.
Model: 0.9b

Explain what was released, how it was trained, and what the evaluation showed.
```

`Model` is optional and must match an ID in `models.json`. It adds checkpoint links to the post and links the post from that model family in the zoo. Update the catalog when publishing new checkpoints. Link each newly released checkpoint directly to its Hugging Face model card when you introduce it in the post. For updates developed in the private repo, copy over the text and links cleared for public release.

## Deploy to project-greenhouse.ai

In the repository’s **Settings → Pages**, set the source to **GitHub Actions**. The website workflow builds on pull requests and deploys from `main`. You can also run it manually from Actions.

Set the custom domain to `project-greenhouse.ai` in **Settings → Pages**. Point the apex DNS records to GitHub Pages using [GitHub’s custom domain instructions](https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site/managing-a-custom-domain-for-your-github-pages-site). Enable **Enforce HTTPS** when GitHub has provisioned the certificate. The custom domain for an Actions deployment is managed in Pages settings.
