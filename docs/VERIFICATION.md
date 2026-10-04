# Review checks

## Completed

- Shared templates rebuilt all 12 academic content pages plus a new 404 page.
- Existing `index.html` redirect and every academic page URL retained.
- Standard-library HTML parser checks passed: balanced tags, one main/title/H1 per page, semantic landmarks, unique IDs, image descriptions, iframe title, and active navigation.
- 69 original content blocks preserved after documented spelling/grammar corrections. The owner explicitly requested replacing the remaining “To be updated” Chatbot placeholder with a front end; that one exception is checked separately.
- All original content links preserved; malformed and empty email links repaired or removed.
- Publication placements: 12 across five categories, including the intentional cross-listing.
- Laboratory equipment: 11 entries across six categories.
- Team: five members, preserved order and factual descriptions.
- News: three original announcements, presented latest first.
- At the owner's request, the personal Google calendar embed was removed and replaced with verified official NMSU academic-calendar and campus-event links. No calendar permissions were changed.
- Chatbot restored to the header navigation between Outreach and News. Added global Machine Learning and CV links removed; the standalone Machine Learning page is retained. Original copyright wording restored; repeated personal name removed from the footer.
- At the owner's request, `CV.pdf` is removed from the review branch and private Site source/build files, not merely unlinked. Original-branch and historical Git copies are unaffected.
- Site JavaScript and build script syntax checks passed. Legacy webcam script is not loaded by academic pages.
- Contact the webmaster now opens a mail draft to `alaie@nmsu.edu` on every page. This changes a link, not email-account forwarding rules.
- Chatbot front-end controls, safe text rendering, keyboard shortcuts, local draft behavior, future connector success/error/cancellation, and reset behavior are covered by dependency-free DOM-mock tests. No real backend, API key, or GPT connection is installed.
- Original image assets and model shards remain in the GitHub branch through the unchanged baseline tree. The CV is explicitly excluded at the owner's request.

## Review limitations

Browser-based rendering, keyboard interaction, 200% zoom, and phone/tablet/desktop screenshots were not available in this execution environment. They have not been claimed as passed. The CSS includes responsive breakpoints, no fixed-width layout tables, and reduced-motion/print support; these still need visual review in the private Site.

External publication/profile links have not been independently availability-tested. The official NMSU calendar destinations were verified from NMSU's public pages. No Google-account-dependent calendar iframe remains. Crimson Connection's events page requires JavaScript on its own website.

The private Sites copy references the existing GitHub Pages URLs for the two large original images (`Lab_Layout_V02.jpg` and `DSC01028.jpg`) because their binary contents could not be retrieved through the connected reader. The production GitHub branch retains the actual original image blobs; they are not removed or replaced. All other displayed local assets are included in the review copy.

## Before production release

Review the private Site, especially the Home reading order, navigation menu, Team image fallbacks, publication layout, and Calendar at narrow widths. Confirm any factual changes separately. Merge the review pull request only after visual review; GitHub Pages production remains unchanged until then.
