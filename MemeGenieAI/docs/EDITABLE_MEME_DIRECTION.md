# Editable meme direction checkpoint

The next development direction is a supplied-background meme editor with AI-assisted text layout. Preserve the existing editor, exports, and optional local image-generation tab. No training or model download is required for the first milestone.

## Verified current state

The Qt/Pillow editor loads backgrounds and renders one caption with font, size, color, alignment, and shadow. PNG/JPEG and landscape/portrait exports work. Text is flattened into the preview/export; there are no independent text layers, editable project saves, drag positioning, or undo stack yet. See LINUX_QUALITY_TESTS.md for existing image-generation validation and limitations.

## Next milestone

1. Add a versioned project model containing the background reference and independent text objects: exact text, normalized position/width, font/style, alignment, and stacking order.
2. Make text objects selectable and movable in the Qt scene. Use the same project model for deterministic export. Keep the source image unchanged.
3. Add title/subtitle, top/bottom caption and split-caption layouts, safe margins, and undo/redo.
4. Add project save/reopen with missing-background handling. Test reopening, cancellation, interrupted saves, and export clipping.
5. Then add optional Suggest Layout with three previews. Preserve user wording by default and apply only the selected suggestion. Manual editing must remain available.

## AI integration boundaries

Start with CPU brightness/contrast and quiet-area analysis, then optionally let a local text model choose a validated layout specification. Do not execute model output as code. Bound text sizes and coordinates, whitelist available fonts, and reject invalid proposals. A vision model may improve subject avoidance, but image-input support and performance must be verified first. Installed small local models are candidates, not yet integrated or benchmarked for this task.

Re-layout text for the selected output aspect ratio before flattening; cropping an already-captioned square image can remove text. Keep image-generation features intact rather than replacing them.

## Checkpoint limitations

This document records the plan, not implemented text layers or AI layout. Desktop test-launcher settings and generated diagnostic images live outside tracked source; the frozen package predates current source fixes. No merge, publication, or new model installation is part of this checkpoint.
