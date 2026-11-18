# Prepare a knowledge bundle

Use one folder containing `index.md`, `manifest.json`, and UTF-8 Markdown instructions. The index describes when to search each section and links to the instruction files.

```text
knowledge/
  index.md
  manifest.json
  access/password-reset.md
```

Example `index.md`:

```markdown
# Support playbook

## Restore account access

Search here for sign-in problems and questions about password reset links.

- [Password reset](access/password-reset.md)
```

Example `manifest.json`:

```json
{
  "documents": [
    {
      "path": "access/password-reset.md",
      "document": "password-reset",
      "version": 1,
      "title": "Password reset",
      "language": "en",
      "product": "general",
      "category": "access",
      "effective_from": "2026-01-01T00:00:00Z",
      "available_from": "2026-01-01T00:00:00Z"
    }
  ]
}
```

Example `access/password-reset.md`:

```markdown
# Password reset

## Link expiry

A password reset link remains valid for 30 minutes after it is issued.
If it expires, send a new link after verifying the customer's identity.
```

| Field | How CHECK uses it |
| --- | --- |
| `path` | Relative path from the bundle root. Link to every manifest file from the index. |
| `document`, `version`, `title` | Keep the document ID stable, start the integer version at 1, and use a short title. Increment the version when its text changes. |
| `language`, `product`, `category` | Match ticket metadata. `general` applies to all products or categories in its respective field. |
| `effective_from`, `effective_to` | When the policy was in force. The optional upper bound is exclusive. |
| `available_from`, `available_to` | When the policy was available to support. The optional upper bound is exclusive. Include `available_from` even when it matches `effective_from`. |
| `audience` | Optional array of allowed `actor_id` values. Omit it or use an empty array for organization-wide availability. |

Use timezone-aware ISO 8601 dates. Document IDs, titles, language, product, and category accept up to 120 characters each. Each nonempty Markdown section becomes a searchable fragment with its heading path. Keep each fragment under 5,000 characters including its heading path, and use up to 30 fragments per instruction version. Put the rule, key conditions, and exceptions together so each fragment can be read independently.

In **Knowledge**, select the folder, review the preview, and choose **Publish bundle**. The bundle accepts up to 100 instruction files, and API requests are limited to 5 MB. CHECK publishes new versions and replaces the organization's index together. An existing version can be linked again if its metadata and contents match.

For API clients, send `index` with the index Markdown and `documents` with the manifest fields plus a `markdown` property for each file. Use `POST /api/knowledge/bundle/preview` for validation and `POST /api/knowledge/bundle` for publication. Individual versions can also be published through `POST /api/knowledge/markdown` or `POST /api/knowledge`, the latter with explicit `fragments`. See the local OpenAPI documentation for request schemas.

## Prompt for converting an existing knowledge base

```text
Convert the supplied support policies to a CHECK knowledge bundle.

Create index.md, manifest.json, and UTF-8 Markdown instruction files. Group
instructions by topic or product. In the index, explain when to search each
section using language found in support conversations. Add a relative link to
each instruction and ensure that it matches the exact manifest path.

Give each policy a stable document ID and a version number. Use the supplied
language, product, category, audience, and validity dates. Report missing
metadata for a human to supply. Preserve policy wording, meaning, conditions,
exceptions, and dates. Split long pages into headings. Keep each searchable
fragment under 5,000 characters and each version at no more than 30 fragments.

Return the folder tree, file contents, and any metadata that needs review.
```
