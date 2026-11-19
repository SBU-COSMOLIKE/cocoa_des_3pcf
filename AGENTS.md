# DES 3PCF development

Read [.claude/skills/des-3pcf/SKILL.md](.claude/skills/des-3pcf/SKILL.md)
before editing this project. Consistency with LSST Y1 and DES Y3 is a
requirement: use their likelihood, example, notebook and directory patterns.

Persistent project instruction from the user (19 November 2026): follow the
LSST Y1 likelihood base's exact structure and coding style. Put native
initialization directly in `initialize`, retain the established class-method
sequence, and put numerical checks in the C++ initializers. No standalone
two-point helper layer, field-name registries or Python likelihood-state
framework. Use one named argument per line. See the project's skill for the
full instruction and the production IA convention.
