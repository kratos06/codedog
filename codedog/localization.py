from codedog.templates import grimoire_en, template_en


class Localization:
    templates = {
        "en": template_en,
    }

    grimoires = {
        "en": grimoire_en,
    }

    def __init__(self, language="en"):
        if language not in self.templates or language not in self.grimoires:
            raise ValueError(f"Unsupported Language: {language}")
        self._language = language

    @property
    def language(self):
        return self._language

    @property
    def template(self):
        return self.templates[self.language]

    @property
    def grimoire(self):
        return self.grimoires[self.language]
