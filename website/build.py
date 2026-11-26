"""Build the public site from Markdown updates and the model catalog."""

import argparse
from datetime import date
from html import escape
import json
from pathlib import Path
import re
import shutil

import markdown


SOURCE = Path(__file__).resolve().parent
REPO = "https://github.com/castorini/project-greenhouse"
PAPER = "https://arxiv.org/abs/2610.11922"
MODELS = "https://huggingface.co/castorini"


def layout(title, description, section, root, body):
    nav = []
    for label, destination in (("Home", ""), ("Research updates", "updates/"), ("Model zoo", "models/")):
        current = ' aria-current="page"' if section == label else ""
        nav.append(f'<a href="{root}{destination}"{current}>{label}</a>')
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="{escape(description)}">
  <title>{escape(title)}</title>
  <link rel="stylesheet" href="{root}style.css">
</head>
<body>
  <a class="skip" href="#main">Skip to content</a>
  <header class="frame top">
    <a class="brand" href="{root}">Project Greenhouse</a>
    <nav aria-label="Main navigation">{"".join(nav)}<a href="{REPO}">GitHub</a></nav>
  </header>
  <main class="frame" id="main">{body}</main>
  <footer class="frame">
    <span>Project Greenhouse · University of Waterloo</span>
    <a href="https://github.com/castorini">Castorini</a>
    <a href="{MODELS}">Hugging Face</a>
  </footer>
</body>
</html>
"""


def read_updates(models):
    updates = []
    slugs = set()
    for source in sorted((SOURCE / "updates").glob("*.md"), reverse=True):
        match = re.fullmatch(r"(\d{4}-\d{2}-\d{2})-([a-z0-9]+(?:-[a-z0-9]+)*)", source.stem)
        if not match:
            raise ValueError(f"{source.name}: expected YYYY-MM-DD-slug.md")
        day, slug = match.groups()
        if slug in slugs:
            raise ValueError(f"{source.name}: duplicate slug {slug}")
        slugs.add(slug)
        document = markdown.Markdown(extensions=["meta", "tables", "toc"])
        body = document.convert(source.read_text(encoding="utf-8"))
        metadata = document.Meta
        title = " ".join(metadata.get("title", []))
        summary = " ".join(metadata.get("summary", []))
        model = " ".join(metadata.get("model", []))
        if not title or not summary or not body or (model and model not in models):
            raise ValueError(f"{source.name}: requires Title, Summary, body and a valid optional Model")
        updates.append(dict(day=date.fromisoformat(day), slug=slug, title=title,
                            summary=summary, model=model, body=body))
    return updates


def date_label(day):
    return f'<time datetime="{day.isoformat()}">{day:%b} {day.day}, {day.year}</time>'


def update_list(updates, prefix, heading="h3"):
    items = []
    for update in updates:
        items.append(f"""<li>{date_label(update["day"])}
          <div><{heading}><a href="{prefix}{update["slug"]}/">{escape(update["title"])}</a></{heading}>
          <p>{escape(update["summary"])}</p></div></li>""")
    return f'<ul class="updates">{"".join(items)}</ul>'


def checkpoints(model):
    links = [
        f'<li><a href="{MODELS}/{escape(checkpoint)}">{escape(label)}</a></li>'
        for label, checkpoint in model["checkpoints"].items()
    ]
    return f'<ul class="checkpoints">{"".join(links)}</ul>'


def build(output):
    catalog = json.loads((SOURCE / "models.json").read_text(encoding="utf-8"))
    models = {model["id"]: model for model in catalog}
    updates = read_updates(models)

    def page(path, title, description, section, root, body):
        target = output / path / "index.html"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(layout(title, description, section, root, body), encoding="utf-8")

    description = "Fully open models for search and agentic search, trained from scratch."
    page("", "Project Greenhouse", description, "Home", "./", f"""
    <section class="hero">
      <div>
        <p class="eyebrow">Open research from Castorini</p>
        <h1>Project<br>Greenhouse</h1>
        <p class="intro">{description} Explore Gaggle’s training data, code, configurations, and checkpoints.</p>
        <div class="actions"><a class="button" href="./models/">Explore the model zoo</a>
        <a href="{PAPER}">Read the paper</a></div>
      </div>
      <div class="hero-art" aria-hidden="true"><div class="roof"></div><div class="glass"></div></div>
    </section>
    <section class="section"><h2>Latest research updates</h2>{update_list(updates[:3], "./updates/")}
      <a href="./updates/">All research updates</a>
    </section>
    <section class="section resources"><h2>Reproduce the research</h2>
      <p>Find the code, experiment configurations, and checks behind the first Gaggle paper.</p>
      <a href="{REPO}/blob/main/REPRODUCE.md">Reproduction guide</a>
      <a href="{REPO}/blob/main/ONBOARDING.md">Getting started</a>
    </section>""")
    page("updates", "Research updates · Project Greenhouse", "Dated releases and research notes.",
         "Research updates", "../", f"""
    <header class="page-head"><h1>Research updates</h1><p>Model releases and research notes from Project Greenhouse.</p></header>
    <section aria-label="Updates by date">{update_list(updates, "./", heading="h2")}</section>""")

    model_sections = []
    for model in catalog:
        family_updates = [update for update in updates if update["model"] == model["id"]]
        notes = "".join(
            f'<li><a href="../updates/{update["slug"]}/">{escape(update["title"])}</a></li>'
            for update in family_updates
        )
        model_sections.append(f"""<section class="model" id="gaggle-{escape(model["id"].replace(".", ""))}">
          <h2>{escape(model["name"])}</h2>{checkpoints(model)}
          {f'<ul class="model-notes">{notes}</ul>' if notes else ""}
        </section>""")
    page("models", "Model zoo · Project Greenhouse", "Public Gaggle checkpoints and release notes.",
         "Model zoo", "../", f"""
    <header class="page-head"><h1>Model zoo</h1><p>Public checkpoints, grouped by Gaggle backbone.</p></header>
    {"".join(model_sections)}""")

    for update in updates:
        model_links = ""
        if update["model"]:
            model = models[update["model"]]
            anchor = model["id"].replace(".", "")
            model_links = f"""<section class="release-models"><h2>Checkpoints</h2>{checkpoints(model)}
              <p><a href="../../models/#gaggle-{anchor}">Find {escape(model["name"])} in the model zoo</a></p></section>"""
        page(f'updates/{update["slug"]}', f'{update["title"]} · Project Greenhouse',
             update["summary"], "", "../../", f"""
        <article class="article">
          <a href="../">All research updates</a>
          <header class="page-head">{date_label(update["day"])}<h1>{escape(update["title"])}</h1>
            <p>{escape(update["summary"])}</p></header>
          {update["body"]}{model_links}
        </article>""")

    shutil.copyfile(SOURCE / "style.css", output / "style.css")
    (output / ".nojekyll").touch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SOURCE / "_site")
    build(parser.parse_args().output)
