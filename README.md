# Seyedhamidreza Alaie — Personal Academic Website

Source for the GitHub Pages site at `seyeala.github.io`.

The site's academic homepage is `about.html`; `index.html` redirects visitors there. Other HTML files contain research, laboratory, publications, team, service, opportunities, calendar, outreach, chatbot, and news pages.

## Repository maintenance

The TensorFlow.js webcam demo that previously occupied the root landing page is preserved in Git history and in the archival branch `archive/pre-housekeeping-2026-09-26`.

When editing the site, keep navigation links relative so they continue to work under GitHub Pages.

## Preservation-first visual refresh

The site remains dependency-free static HTML on GitHub Pages. Existing `.html` URLs and the `index.html` redirect are unchanged. Root HTML pages are generated and tracked, so GitHub Pages does not need a custom build service.

- Edit page material in `content/pages.json`.
- Edit shared header/footer structure in `scripts/build-pages.mjs`.
- Edit the visual system in `assets/css/site.css`.
- Navigation and portrait fallback behavior live in `assets/js/site.js`.
- Run `npm run build` after changing content or shared templates, then commit the generated pages as well as the sources.
- Run `npm run check` (Python 3, standard library only) for structural, internal-link, and content-preservation checks.

`scripts/content-baseline.json` records content from commit `4a3c3b51749773d766f5f491d00625a98d1fa646`. It verifies 70 original text blocks, publication placements, member/equipment counts, and original links. It is a review guardrail, not a source of new factual claims.

All original image paths, the legacy webcam script, and the TensorFlow model files remain in the repository. At the owner's request, `CV.pdf` is removed from the review branch and all publishable website files; it is not linked or embedded. This does not erase copies in the original branch or Git history. The legacy script is no longer loaded by academic pages. `style.css` remains as a compatibility entrypoint to the new shared CSS. Chatbot is restored to the header's More menu between Outreach and News. Machine Learning retains its standalone URL without an added global footer link. The repeated personal name is removed from the footer.

Two failed external Team portraits use accessible initials instead of broken-image placeholders. Other portraits retain a fallback if loading fails. Empty email anchors are removed without inventing contact addresses.

See `docs/CONTENT-REVIEW.md` for factual inconsistencies that need owner confirmation. Dates, roles, citation details, and grant wording were not silently changed. At the owner's request, the personal Google calendar embed was replaced with official NMSU academic-calendar and campus-event links.
