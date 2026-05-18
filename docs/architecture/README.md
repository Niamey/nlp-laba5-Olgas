# Діаграми архітектури

У цій теці лежать **джерела Mermaid** (`.mmd`) і згенеровані **SVG** та **PNG** (растрові «картинки» для слайдів).

| Файл | Зміст |
|------|--------|
| **`00-general-flow`** `.mmd` / `.svg` / `.png` | **Флоу 0:** загальна архітектура end-to-end (вхід → точка входу → `run_triage` → LangGraph + MCP → вихід) |
| **`05-langchain-flow`** `.mmd` / `.svg` / `.png` | **Флоу LangChain/LangGraph:** LLM, вузли графа, draft → grounding → completion → HIL/END |
| **`06-mcp-flow`** `.mmd` / `.svg` / `.png` | **Флоу MCP:** `mcp_config` → два stdio-процеси → `MultiServerMCPClient` → `set_mcp_tools` → вузли |
| `01-overview` … | Детальна загальна схема (шари клієнт / API / ядро / MCP / зовнішні сервіси) |
| `02-langgraph` … | Детальний граф з усіма розвилками |
| `03-mcp-github-triage` … | П’ять тулів власного MCP |
| `04-mcp-fetch` … | Сторонній MCP `fetch` |

**Текстовий опис кожного флоу (для захисту):** [`../FLOW_ARCHITECTURE_UA.md`](../FLOW_ARCHITECTURE_UA.md)

## Як згенерувати PNG / SVG на диск

З кореня репозиторія (потрібні [Node.js](https://nodejs.org/) і `npx`):

```bash
cd docs/architecture
# SVG (вектор)
npx --yes @mermaid-js/mermaid-cli -i 01-overview.mmd -o 01-overview.svg
# PNG (малюнок для Word / слайдів)
npx --yes @mermaid-js/mermaid-cli -i 01-overview.mmd -o 01-overview.png
```

Повторити для інших `.mmd` у цій теці (наприклад `00-general-flow`, `05-langchain-flow`, `06-mcp-flow`, `02-langgraph`, …).

Альтернатива без Node: відкрити вміст `.mmd` на [mermaid.live](https://mermaid.live) і експортувати PNG/SVG вручну.
