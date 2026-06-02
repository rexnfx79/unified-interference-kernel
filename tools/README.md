# LLM Wiki Desktop App (Submodule)

Git submodule: [nashsu/llm_wiki](https://github.com/nashsu/llm_wiki)

Use this to run the cross-platform LLM Wiki application against the parent repo's knowledge base.

## Quick Start

```bash
cd tools/llm_wiki
npm install
npm run tauri dev    # development
npm run tauri build  # production binary
```

In the app: **Open Project** → select `../../knowledge/` (relative to this directory).

## Update Submodule

```bash
git submodule update --remote tools/llm_wiki
```

## Agent Skill (optional)

```bash
npx skills add https://github.com/nashsu/llm_wiki_skill.git --skill llm_wiki_skill
```

Requires LLM Wiki app running with API enabled (Settings → API Server).

## License

LLM Wiki is GPL-3.0. This submodule is optional tooling; the `knowledge/` markdown wiki is standard project content.
